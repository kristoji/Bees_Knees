from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np
import torch
from torch_geometric.data import Data
from ai.graph_honored_network import GraphClassifier
from ai.improved_honored_GNN import GraphClassifierImproved
from ai.training import Training
from ai.oracle import Oracle
from ai.loader import GraphDataset
from gpt import board_to_simple_honored_graph
from engineer import Engine
from ai.log_utils import log_header, log_subheader
import os
from ai.brains import MCTS
from engine.enums import GameState, PlayerColor
from gpt import save_simple_honored_graph
import json
from typing import List, Optional, Tuple
from torch_geometric.data import Batch


# --------------------------------------------------------------
# OracleGNN – add batched helpers while keeping existing API
# --------------------------------------------------------------
class OracleGNN_BATCH(Oracle):
    """
    Oracle that uses a neural network to predict the value and policy of a board state.
    This version adds batched helpers to reduce per-state overhead.
    """
    def __init__(self):
        self.device = torch.device(torch.environ.get("TORCH_DEVICE", "cpu")) if hasattr(torch, 'environ') else torch.device("cpu")
        self.kwargs_network = { 'conv_type': 'GAT' }  # 'GIN', 'GAT', 'GCN'
        self.network = GraphClassifierImproved(in_dim=13, hidden_dim=64, num_classes=1, **self.kwargs_network)
        self.network.to(self.device)
        self.cache: Dict[int, float] = {}
        self.path: Optional[str] = None

    # -------------------------
    #  I/O and training (unchanged API)
    # -------------------------
    def training(self, train_data_path: str, epochs: int) -> None:
        from ai.loader import GraphDataset
        if not self.network:
            raise ValueError("Neural network is not initialized.")

        dataset = GraphDataset(folder_path=train_data_path)
        if self.device.type == 'cuda':
            dataset = self._preload_to_gpu(dataset)

        batch_size = 128
        train_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=0)
        self.network.train_network(train_loader=train_loader, epochs=epochs)

    def _preload_to_gpu(self, dataset):
        for i in range(len(dataset)):
            if dataset.data[i] is not None:
                dataset.data[i] = dataset.data[i].to(self.device)
        return dataset

    def save(self, path: str) -> None:
        self.path = path
        self.network.save(path)

    def load(self, path: str) -> None:
        self.path = path
        self.network.load(path)

    def copy(self) -> 'OracleGNN':
        if not self.path:
            raise ValueError("Path is not set. Cannot copy without a path.")
        new_oracle = OracleGNN()
        new_oracle.load(self.path)
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

    def _data_from_board(self, board: Board) -> Optional[Data]:
        """Make a PyG Data from a board. Returns None for empty graph.
        The `batch` field is set for single-graph usage; Batch.from_data_list will fix it.
        """
        x, edge_index, _ = board_to_simple_honored_graph(board)
        x = self.flatten_nodes(x)
        if len(x) == 0:
            return None
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).contiguous(),
            batch=torch.zeros(len(x), dtype=torch.long),
        )
        return data

    # -------------------------
    #  Batched forward helpers
    # -------------------------
    @torch.no_grad()
    def predict_values_batch_from_data(self, data_list: List[Data], use_sigmoid: bool = True) -> List[float]:
        """Fast path: pass a list of PyG Data and get a list of floats (in [0,1] if use_sigmoid).
        Assumes the model accepts a batched Data (via torch_geometric.data.Batch).
        """
        if not data_list:
            return []
        batch = Batch.from_data_list([d.to(self.device) for d in data_list])
        out = self.network.predict(batch, use_sigmoid=use_sigmoid)  # should support batched Data
        if isinstance(out, torch.Tensor):
            vals = out.detach().cpu().view(-1).tolist()
        elif isinstance(out, np.ndarray):
            vals = out.reshape(-1).tolist()
        else:
            # Fallback: try to float() each item
            vals = [float(o) for o in out]
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
            v = 1 - self._to_float(v)  # inverted net
        return v

    def predict(self, board: Board, game: bool = True) -> Tuple[float, Dict[Move, float]]:
        """Single-state path kept for compatibility; internally uses batch helpers.
        This still computes π by looking one ply ahead (batched under the hood).
        """
        # Value for the leaf
        d_leaf = self._data_from_board(board)
        if d_leaf is None:
            v = 0.5
        else:
            v = self.predict_values_batch_from_data([d_leaf], use_sigmoid=True)[0]
        v = 1 - self._to_float(v)  # inverted net

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
                V = 1 - V  # inverted net for next state
            probs.append(1 - float(V))
        if probs:
            arr = np.array(probs, dtype=np.float32)
            arr = np.exp(arr - np.max(arr))
            arr /= np.sum(arr)
            pi = {m: float(p) for m, p in zip(valid_moves, arr.tolist())}
        return v, pi

