from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np
import torch
from torch_geometric.data import Data
from ai.graph_honored_network import GraphClassifier
from ai.training import Training
from ai.oracle import Oracle
from ai.loader import GraphDataset
from gpt import board_to_simple_honored_graph

class OracleGNN(Oracle):
    """
    Oracle that uses a neural network to predict the value and policy of a board state.
    """
    def __init__(self):
        self.network = GraphClassifier(in_dim=13, hidden_dim=64, num_classes=1)
        self.path = "pro_matches/GNN_Apr-3-2024/graphs"
        self.train_loader =  GraphDataset(folder_path=self.path) # ------------> DA METTERE co dataloader

    def training(self, ts: str, iteration: int) -> None:
        """
        Train the neural network with the provided training data.
        T is a tuple of (in_mats, out_mats, values)
        """
        if not self.network:
            raise ValueError("Neural network is not initialized.")
        self.network.train_network(
            train_loader=self.train_loader,
            epochs=3,
        )

    def save(self, path: str) -> None:
        """
        Save weights
        """
        self.path = path
        self.network.save(path)


    def copy(self) -> 'OracleGNN':
        """
        Create a copy of the Oracle instance.
        """
        if not self.path:
            # self.save("temp.pth") # save in a temp file just to perform the copy
            raise ValueError("Path is not set. Cannot copy without a path.")
        new_oracle = OracleGNN()
        new_oracle.network.load(self.path) 
        return new_oracle
    
    def compute_heuristic(self, board) -> float:
        v, _ = self.predict(board)
        return v

    def predict(self, board: Board) -> tuple[float, Dict[Move, float]]:
        """
        Predict the value and policy for the given board state.
        """

        # TODO : convert the board into a graph representation
        x, edge_index, pos_bug_to_index = board_to_simple_honored_graph(board)

        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            batch=torch.zeros(len(x), dtype=torch.long)  # Fixed: should be zeros for single graph        
            )
        
        v = self.network.predict(data) # PREDICT THE STATE

        
        valid_moves = list(board.get_valid_moves())
        pi = {}
        for m in valid_moves:
            board.safe_play(m) # safe to optimize
            x, edge_index, pos_bug_to_index = board_to_simple_honored_graph(board)
            data = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                batch=torch.zeros(len(x), dtype=torch.long)  # Fixed: should be zeros for single graph            
                )
            pi[m] = 1 - self.network.predict(data)
            board.undo(m)
            
        # Softmax the probabilities
        if pi:
            probs = np.array(list(pi.values()))
            probs = np.exp(probs - np.max(probs))
            probs /= np.sum(probs)
            pi = {move: prob for move, prob in zip(pi.keys(), probs)}
        else:
            raise ValueError("No valid moves found in the board state.")

        return v, pi
