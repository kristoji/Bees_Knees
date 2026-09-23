from engine.board import Board
from engine.game import Move
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from ai.graph_network import GraphClassifier
from ai.oracle import Oracle
from ai.loader import GraphDataset
from engine.enums import BugType, Direction
from engine.game import NEIGHBOR_INDICES
from collections import defaultdict
from ai.log_utils import log_header, log_subheader
import os
from engine.enums import GameState, PlayerColor


SHORT = 50
LONG = 100
SUPERLONG = 150
TURN_LIMIT = 100

# Node feature layout, built once instead of on every _data_from_board call.
# NOTE: column 1 is never written. The one-hot index starts at 1 and is then offset
# by another 1, so the types land in columns 2..9 and one of the 13 features is
# always zero. Left as is on purpose: changing it changes the network's input
# representation, which is a modelling decision, not an optimization.
_NUM_FEATURES = 1 + len(list(BugType)) + 1 + 3
_TYPE_COLUMN = {bug_type: 1 + i + 1 for i, bug_type in enumerate(BugType)}

class OracleGNN(Oracle):
    """
    Oracle that uses a neural network to predict the value and policy of a board state.
    """
    def __init__(self, device: Optional[str] = None, hidden_dim: int = 64, **kwargs_network) -> None:
        # self.device = torch.device(torch.environ.get("TORCH_DEVICE", "cpu")) if hasattr(torch, 'environ') else torch.device("cpu")
        # device to gpu
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs_network = kwargs_network
        self.network = GraphClassifier(in_dim=13, hidden_dim=hidden_dim, num_classes=1, **self.kwargs_network)
        self.network.to(self.device)
        self.path: Optional[str] = None
        if self.device.type == "cuda":
            # Global flags: these were being re-set on every single batch predict.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._amp_dtype = None

        if self.device.type == 'cpu':
            os.environ["OMP_NUM_THREADS"] = "8"     # scegli in base ai core fisici
            os.environ["MKL_NUM_THREADS"] = "8"
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)        # evita oversubscription
            
        try:
            self.network = torch.compile(self.network, mode="reduce-overhead")  # or "reduce-overhead"
        except Exception:
            print("sei ghey, torch.compile not supported on this device")
            pass  # fall back if PyG op not supported
        
    def training(self, train_data_path:str, epochs:int) -> None:
        """
        Train the neural network with the provided training data.
        T is a tuple of (in_mats, out_mats, values)
        """
        self.path = train_data_path

        log_header(f"STARTING DATA LOADING")

        self.dataset =  GraphDataset(folder_path=self.path) # ------------> DA METTERE co dataloader

        if not self.network:
            raise ValueError("Neural network is not initialized.")
        
        if self.device.type == 'cuda':
            print(f"Pre-loading dataset to GPU: {self.device}")
            self.dataset = self._preload_to_gpu(self.dataset)
        
        batch_size = 1024 #128
        if self.device.type == 'cuda':
            self.train_loader, self.test_loader = self.dataset.get_dataloader(batch_size=batch_size, train_size=0.8, shuffle=True, num_workers=0)
        else: #if we are on CPU
            # No pin_memory here: page-locked staging buffers only pay off for a host
            # to device copy, and there is no device. (The CUDA branch above uses
            # num_workers=0 on purpose: the dataset is already resident on the GPU.)
            self.train_loader, self.test_loader = self.dataset.get_dataloader(batch_size=batch_size, train_size=0.8, shuffle=True, num_workers=6, persistent_workers=True, prefetch_factor=4)

        if not self.network:
            raise ValueError("Neural network is not initialized.")
        log_subheader("Data loading ended!")
        log_header("STARTING PRE-TRAINING")
        self.network.train_network(
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            epochs=epochs,
        )

    def _preload_to_gpu(self, dataset):
        """Pre-move dataset to GPU to avoid repeated transfers."""
        for i in range(len(dataset)):
            if dataset.data[i] is not None:
                dataset.data[i] = dataset.data[i].to(self.device)
        return dataset
    
    def save(self, path: str) -> None:
        """
        Save weights
        """
        self.path = path
        self.network.save(path)

    def load(self, path: str) -> None:
        """
        Load weights
        """
        self.path = path
        self.network.load(path)

    def copy(self) -> 'OracleGNN':
        """
        Create a copy of the Oracle instance.
        """
        #TODO: probabilmente sbagliata!!!
        if not self.path:
            # self.save("temp.pth") # save in a temp file just to perform the copy
            raise ValueError("Path is not set. Cannot copy without a path.")
        new_oracle = OracleGNN()
        new_oracle.network.load(self.path) 
        return new_oracle
   
    # -------------------------
    #  Helpers
    # -------------------------
    def flatten_nodes(self, x: List[List[float]]) -> List[List[float]]:
        temp: List[List[float]] = []
        for el in x:
            flattened: List[float] = []
            for sub_el in el:
                if isinstance(sub_el, list):
                    flattened.extend(sub_el)
                else:
                    flattened.append(sub_el)
            temp.append(flattened)
        return temp

    def _to_float(self, x) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().view(-1)[0].item())
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        return float(x)

    @torch.no_grad()
    def predict_values_batch_from_data(self, data_list, use_sigmoid=True):
        if not data_list: return []
        batch = Batch.from_data_list(data_list)
        if self.device.type == "cuda":
            # torch.autocast takes device_type as a required positional argument; the
            # previous call omitted it and raised TypeError, so this whole CUDA branch
            # was dead.
            batch = batch.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=self._amp_dtype):
                out = self.network.predict(batch, use_sigmoid=use_sigmoid)
        else:
            out = self.network.predict(batch, use_sigmoid=use_sigmoid)
        return out.detach().cpu().view(-1).tolist() if isinstance(out, torch.Tensor) \
            else np.asarray(out).reshape(-1).tolist()
    

    @torch.no_grad()
    def predict_values_batch_from_data_with_gpu(self, data_list, use_sigmoid: bool = True):
        if not data_list:
            return []

        # Build Batch on CPU first (fast), then move in one shot
        batch = Batch.from_data_list(data_list)

        if self.device.type == "cuda":
            # allow TF32 on Ampere+; good speed, same or near-same accuracy
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # autocast: prefer bf16 (more robust) if supported; else fp16
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            # Non-blocking copy: effective if tensors were allocated in pinned memory
            batch = batch.to(self.device, non_blocking=True)

            with torch.autocast("cuda", dtype=amp_dtype):
                out = self.network.predict(batch, use_sigmoid=use_sigmoid)
        else:
            out = self.network.predict(batch, use_sigmoid=use_sigmoid)

        if isinstance(out, torch.Tensor):
            vals = out.detach().cpu().view(-1).tolist()
        else:
            vals = np.asarray(out).reshape(-1).tolist()
        return [float(v) for v in vals]


    # Keep single-item API for backwards compatibility
    def compute_heuristic(self, board: Board, game: bool = True) -> float:
        if board.state != GameState.IN_PROGRESS:
            if board.state == GameState.DRAW:
                v = 0.5
            else:
                v = 1.0 if (
                    (board.state == GameState.WHITE_WINS and board.current_player_color == PlayerColor.WHITE) or
                    (board.state == GameState.BLACK_WINS and board.current_player_color == PlayerColor.BLACK)
                ) else 0.0
        else:
            d = self._data_from_board(board)
            if d is None:
                v = 0.5
            else:
                v = self.predict_values_batch_from_data([d], use_sigmoid=True)[0]
            v = self._to_float(v)  # convert to float
            # v = 1 - v  # inverted net
        return v

    def predict(self, board: Board, game: bool = True) -> Tuple[float, Dict[Move, float]]:
        """Single-state path kept for compatibility; internally uses batch helpers.
        This still computes π by looking one ply ahead (batched under the hood).
        """
        # Value for the leaf
        d_leaf = self._data_from_board(board)
        #print(d_leaf.x, d_leaf.edge_index, d_leaf.batch)
        if d_leaf is None:
            v = 0.5
        else:
            v = self.predict_values_batch_from_data([d_leaf], use_sigmoid=True)[0]
        v = self._to_float(v)
        #v = 1 - self._to_float(v)  # inverted net

        # Policy via one-ply lookahead
        pi: Dict[Move, float] = {}
        valid_moves = list(board.get_valid_moves())
        next_datas: List[Data] = []
        map_indices: List[Tuple[int, Optional[float]]] = []  # (idx in next_datas or -1, terminal V if any)

        for m in valid_moves:
            board.safe_play(m)
            if board.state != GameState.IN_PROGRESS:
                if board.state == GameState.DRAW:
                    V = 0.5
                else:
                    V = 1.0 if (
                        (board.state == GameState.WHITE_WINS and board.current_player_color == PlayerColor.WHITE) or
                        (board.state == GameState.BLACK_WINS and board.current_player_color == PlayerColor.BLACK)
                    ) else 0.0
                map_indices.append((-1, float(V)))
            else:
                d = self._data_from_board(board)
                if d is None:
                    map_indices.append((-1, 0.5))
                else:
                    map_indices.append((len(next_datas), None))
                    next_datas.append(d)
            board.undo()

        if next_datas:
            preds = self.predict_values_batch_from_data(next_datas, use_sigmoid=True)
        else:
            preds = []

        # Assemble π
        probs: List[float] = []
        for idx, termV in map_indices:
            if idx == -1:
                V = termV
            else:
                V = float(preds[idx])
                # V = 1 - V  # inverted net for next state
            probs.append(1 - float(V))
        if probs:
            arr = np.array(probs, dtype=np.float32)
            arr = np.exp(arr - np.max(arr))
            arr /= np.sum(arr)
            pi = {m: float(p) for m, p in zip(valid_moves, arr.tolist())}
        return v, pi

    # Alternative: If you still need the separate functions for compatibility
    def _data_from_board(self, board: Board) -> Optional[Data]:
        """Build the PyG graph for a board state.

        This runs once per legal move per expanded leaf, so it is the hottest Python
        in a GNN-guided search. Changes from the previous version, all
        output-preserving: the type table is a module constant instead of being
        rebuilt per call; pos_bug_to_index (a dict keyed on (Position, Bug), so every
        insert hashed a Bug) was built and never read; the per-height grouping is a
        list of dicts keyed on dense cell indices rather than a defaultdict keyed on
        (Position, height) tuples; edges go straight into a numpy array instead of a
        list of Python tuples; and the tensors are no longer pinned individually.
        """
        pos_to_bug = board._pos_to_bug
        if not pos_to_bug:
            return None

        current_player = board.current_player_color
        art_pos_set = board._art_pos

        # Nodes, grouped by stack height as we go.
        x_rows = []
        height_maps: List[dict] = []
        vertical: List[tuple] = []
        node_idx = 0
        for pos, bugs in pos_to_bug.items():
            if not bugs:
                continue
            is_art = pos.index in art_pos_set
            num_bugs = len(bugs)
            pos_index = pos.index
            first_idx = node_idx
            for h, bug in enumerate(bugs):
                while len(height_maps) <= h:
                    height_maps.append({})
                height_maps[h][pos_index] = node_idx
                x_rows.append((
                    1.0 if bug.color == current_player else 0.0,
                    _TYPE_COLUMN[bug.type],
                    1.0 if h < num_bugs - 1 else 0.0,   # pinned
                    1.0 if h > 0 else 0.0,              # pinning
                    1.0 if h == 0 and is_art else 0.0,  # articulation
                ))
                node_idx += 1
            for h in range(num_bugs - 1):
                i = first_idx + h
                vertical.append((i, i + 1))
                vertical.append((i + 1, i))

        total_nodes = node_idx
        if total_nodes == 0:
            return None

        x = np.zeros((total_nodes, _NUM_FEATURES), dtype=np.float32)
        for i, (color, type_col, pinned, pinning, art) in enumerate(x_rows):
            x[i, 0] = color
            x[i, type_col] = 1.0
            x[i, -3] = pinned
            x[i, -2] = pinning
            x[i, -1] = art

        # Flat edges: same height, adjacent cells. Both directions appear because both
        # endpoints are visited.
        edges: List[tuple] = vertical
        for pos_map in height_maps:
            for pos_index, i in pos_map.items():
                for neighbor_index in NEIGHBOR_INDICES[pos_index]:
                    j = pos_map.get(neighbor_index)
                    if j is not None:
                        edges.append((i, j))

        if edges:
            edge_index = torch.from_numpy(
                np.asarray(edges, dtype=np.int64).T.copy()
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        batch = torch.zeros(total_nodes, dtype=torch.long)
        # pin_memory() used to be called on each of these three tiny tensors. It is a
        # page-locking allocation with a device synchronisation: on ~30-node graphs it
        # costs far more than the asynchronous copy it enables. If pinning is wanted it
        # belongs on the aggregated Batch in predict_values_batch_from_data.
        return Data(x=torch.from_numpy(x), edge_index=edge_index, batch=batch)
