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
from collections import defaultdict
from ai.log_utils import log_header, log_subheader
import os
from engine.enums import GameState, PlayerColor


SHORT = 50
LONG = 100
SUPERLONG = 150
TURN_LIMIT = 100

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
        self.cache: Dict[int, float] = {}
        self.path: Optional[str] = None
        self.pin = (self.device.type == "cuda")

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

        self.train_loader =  GraphDataset(folder_path=self.path) # ------------> DA METTERE co dataloader

        if not self.network:
            raise ValueError("Neural network is not initialized.")
        
        if self.device.type == 'cuda':
            print(f"Pre-loading dataset to GPU: {self.device}")
            self.train_loader = self._preload_to_gpu(self.train_loader)
        
        batch_size = 1024 #128
        if self.device.type == 'cuda':
            self.train_loader = self.train_loader.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=0)
        else: #if we are on CPU
            self.train_loader = self.train_loader.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)

        if not self.network:
            raise ValueError("Neural network is not initialized.")
        log_subheader("Data loading ended!")
        log_header("STARTING PRE-TRAINING")
        self.network.train_network(
            train_loader=self.train_loader,
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
        batch_cpu = Batch.from_data_list(data_list)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            batch = batch_cpu.to(self.device, non_blocking=True)
            with torch.autocast(dtype=amp_dtype):
                out = self.network.predict(batch, use_sigmoid=use_sigmoid)
        else:
            out = self.network.predict(batch_cpu, use_sigmoid=use_sigmoid)
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

            with torch.cuda.amp.autocast(dtype=amp_dtype):
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
        """
        Even faster version using numpy throughout and minimal Python loops.
        """
        pos_to_bug = board._pos_to_bug
        if not pos_to_bug:
            return None
        
        # Count total nodes first
        total_nodes = sum(len(bugs) for bugs in pos_to_bug.values())
        if total_nodes == 0:
            return None
        
        # Setup
        types = list(BugType)
        type_to_index = {bug_type: (i + 1) for i, bug_type in enumerate(types)}
        num_features = 1 + len(types) + 1 + 3  # color + one-hot type + pinned + pinning + art
        
        # Pre-allocate feature matrix
        x = np.zeros((total_nodes, num_features), dtype=np.float32)
        
        # Tracking
        pos_bug_to_index = {}
        pos_height_to_idx = {}
        current_player = board.current_player_color
        art_pos_set = board._art_pos
        
        # Build nodes with vectorized operations where possible
        node_idx = 0
        for pos, bugs in pos_to_bug.items():
            is_art = pos in art_pos_set
            num_bugs = len(bugs)
            
            for h, bug in enumerate(bugs):
                # Set features directly in pre-allocated array
                x[node_idx, 0] = 1.0 if bug.color == current_player else 0.0
                x[node_idx, 1 + type_to_index[bug.type]] = 1.0
                x[node_idx, -3] = 1.0 if h < num_bugs - 1 else 0.0  # pinned
                x[node_idx, -2] = 1.0 if h > 0 else 0.0  # pinning
                x[node_idx, -1] = 1.0 if h == 0 and is_art else 0.0  # articulation
                
                pos_bug_to_index[(pos, bug)] = node_idx
                pos_height_to_idx[(pos, h)] = node_idx
                node_idx += 1
        
        # Build edges using numpy for better performance
        edge_list = []
        
        # Flat edges - batch process by height
        by_height = defaultdict(dict)
        for (pos, h), idx in pos_height_to_idx.items():
            by_height[h][pos] = idx
        
        for h, pos_map in by_height.items():
            for pos, i in pos_map.items():
                for d in Direction.flat():
                    npos = pos.get_neighbor(d)
                    if npos in pos_map:
                        edge_list.append((i, pos_map[npos]))
        
        # Vertical edges
        for pos, bugs in pos_to_bug.items():
            for h in range(len(bugs) - 1):
                i = pos_height_to_idx[(pos, h)]
                j = pos_height_to_idx[(pos, h + 1)]
                edge_list.extend([(i, j), (j, i)])
        
        # Convert to tensor
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Apply pinning if needed
        if self.pin:
            x_tensor = torch.from_numpy(x).pin_memory()
            edge_index = edge_index.pin_memory()
            batch = torch.zeros(total_nodes, dtype=torch.long).pin_memory()
        else:
            x_tensor = torch.from_numpy(x)
            batch = torch.zeros(total_nodes, dtype=torch.long)
        # return x_tensor, edge_index, batch
        return Data(x=x_tensor, edge_index=edge_index, batch=batch)
    

