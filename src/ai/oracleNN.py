from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np
from ai.network import NeuralNetwork
from ai.training import Training
from ai.oracle import Oracle
    

class OracleNN(Oracle):
    """
    Oracle that uses a neural network to predict the value and policy of a board state.
    """
    def __init__(self):
        self.network = NeuralNetwork()
        self.path = ""    

    def training(self, ts: str, iteration: int) -> None:
        """
        Train the neural network with the provided training data.
        T is a tuple of (in_mats, out_mats, values)
        """
        if not self.network:
            raise ValueError("Neural network is not initialized.")
        self.network.train_network(
            ts=ts, 
            iteration=iteration,
            num_epochs=15, 
            batch_size=32, 
            learning_rate=0.001,
            value_loss_weight=0.5 
        )

    def save(self, path: str) -> None:
        """
        Save weights
        """
        self.path = path
        self.network.save(path)


    def copy(self) -> 'OracleNN':
        """
        Create a copy of the Oracle instance.
        """
        if not self.path:
            # self.save("temp.pth") # save in a temp file just to perform the copy
            raise ValueError("Path is not set. Cannot copy without a path.")
        new_oracle = OracleNN()
        new_oracle.network.load(self.path) 
        return new_oracle
    
    def compute_heuristic(self, board) -> float:
        v, _ = self.predict(board)
        return v

    def predict(self, board: Board) -> tuple[float, Dict[Move, float]]:
        """
        Predict the value and policy for the given board state.
        """
        T = Training.get_in_mat_from_board(board)
        T = np.array(T, dtype=np.float32).reshape((1, *Training.INPUT_SHAPE))
        v, pi_mat = self.network.predict(T)

        # print(pi_mat.shape) # (1, 109760)

        pi = Training.get_dict_from_matrix(pi_mat[0], board)
        valid_moves = list(board.get_valid_moves())
        # Filter pi to only include valid moves
        pi = {move: prob for move, prob in pi.items() if move in valid_moves}
        # Softmax the probabilities
        if pi:
            probs = np.array(list(pi.values()))
            probs = np.exp(probs - np.max(probs))
            probs /= np.sum(probs)
            pi = {move: prob for move, prob in zip(pi.keys(), probs)}
        # else:
        #     raise ValueError("No valid moves found in the board state.")

        return v, pi
