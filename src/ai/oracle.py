from engine.board import Board
from engine.enums import PlayerColor
from engine.game import Move
from typing import Optional, List, Dict
import numpy as np
from ai.network import NeuralNetwork
from ai.training import Training

# -------------------------------- FROM CURRENT BOARD TO CENTERED BOARD ---------------------------------------
# origin = Training.get_wQ_pos()
# if not origin:
#   origin = (0,0)
# pos_to_bug_centered = Training.center_pos(curr_board.pos_to_bug, origin)
# matrice_in = Training.to_in_mat(pos_to_bug_centered , curr_board.current_player_color)

# -------------------------- USING NN TO GET MOVE PROBABILITIES AND V VALUE ----------------------------------
# output_rete = chiamo_la_rete()
# matrice_out, v = output_rete

# ------------------------- TRANSFORMING THE OUT MATRIX IN A DICT[MOVE, FLOAT] -------------------------------
# dict_mov_prob = Training.get_dict_from_matrix(matrice_out, origin, curr_board)


class Oracle:
    def __init__(self, nn: bool = False):
        pass

    def compute_heuristic(self, board: Board) -> float:
        "Compute a heuristic value for the board state in [0, 1], where 1 is a win for white and 0 is a win for black"
        score_queen =  (6 - board.count_queen_neighbors(board.current_player_color) + board.count_queen_neighbors(board.other_player_color)) / 12.0
        # # score = board.count_queen_neighbors(board.other_player_color) / 6.0
        # moves_curr = len(board.get_valid_moves(board.current_player_color))
        # moves_opp = len(board.get_valid_moves(board.other_player_color))
        # diff =  moves_curr - moves_opp
        # score_mobility = diff / max(moves_curr, moves_opp, 1)  # Avoid division by zero ???

        # if board.current_player_color == PlayerColor.BLACK:
        #     k1 = 100
        #     k2 = 20
        # else:
        #     k1 = 50
        #     k2 = 50

        # score = (k1*score_queen + k2*score_mobility) / (k1 + k2)
        return score_queen
    
    def predict(self, board: Board) -> tuple[float, Dict[Move, float]]:
        """
        returns a tuple of (v, pi)  
        where v is the value of the board and pi is a dictionary of moves with their probabilities
        """

        v = self.compute_heuristic(board)
        valid_moves = list(board.get_valid_moves())
        pi = {}
        h_arr = np.zeros(len(valid_moves), dtype=float)
        for i, move in enumerate(valid_moves):
            board.safe_play(move)
            h = 1 - self.compute_heuristic(board)
            board.undo()
            h_arr[i] = h

        h_arr /= h_arr.sum()
        # print("h_arr:", h_arr)
        # softmax = np.exp(h_arr - np.max(h_arr))
        # if softmax.sum() > 0:
        #     h_arr = softmax / softmax.sum()
        # else:
        #     h_arr = np.zeros_like(h_arr)
        # print("softmax:", softmax)
        # if len(set(h_arr)) != 1:
        #     print("h_arr has more than one unique value")
        #     exit()

        pi = {move: prob for move, prob in zip(valid_moves, h_arr)} if h_arr.size > 0 else {}
        return v, pi
    

class OracleNN(Oracle):
    """
    Oracle that uses a neural network to predict the value and policy of a board state.
    """
    def __init__(self):
        self.network = NeuralNetwork()
        self.path = ""    

    def training(self, T: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Train the neural network with the provided training data.
        T is a tuple of (in_mats, out_mats, values)
        """
        if not self.network:
            raise ValueError("Neural network is not initialized.")
        self.network.train_network(
            T, 
            num_epochs=10, 
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
        return v, pi
