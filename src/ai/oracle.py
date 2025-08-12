from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np
from engine.enums import PlayerColor, GameState

class Oracle:
    def __init__(self, nn: bool = False):
        pass

    def compute_heuristic(self, board: Board) -> float:
        "Compute a heuristic value for the board state in [0, 1], where 1 is a win for white and 0 is a win for black"
        if board.state != GameState.IN_PROGRESS and board.state != GameState.DRAW:
            score_queen = 1.0 if board.current_player_color == PlayerColor.WHITE and board.state == GameState.WHITE_WINS or board.current_player_color == PlayerColor.BLACK and board.state == GameState.BLACK_WINS else 0.0
            return score_queen
        
        if board.state == GameState.DRAW:
            return 0.5
        
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
            if h == 1.0:
                h = 7
            elif h == 0.0:
                h = -7
            h_arr[i] = h

        #softmax
        h_arr = np.exp(h_arr - np.max(h_arr))
        h_arr /= np.sum(h_arr)
        pi = {move: prob for move, prob in zip(valid_moves, h_arr)} if h_arr.size > 0 else {}
        return v, pi

    def training(self, train_data_path:str, epochs:int) -> None:
        pass

    def generate_matches(self, iteration_folder: str, n_games: int = 500, n_rollouts: int = 1500, verbose: bool = False, perc_allowed_draws: float = float('inf')) -> None:
        pass

    