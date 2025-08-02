from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np

class Oracle:
    def __init__(self, nn: bool = False):
        pass

    def compute_heuristic(self, board: Board) -> float:
        "Compute a heuristic value for the board state in [0, 1], where 1 is a win for white and 0 is a win for black"
        score_queen  =  (6 - board.count_queen_neighbors(board.current_player_color) + board.count_queen_neighbors(board.other_player_color)) / 12.0
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
        pi = {move: prob for move, prob in zip(valid_moves, h_arr)} if h_arr.size > 0 else {}
        return v, pi
    