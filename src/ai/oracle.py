from engine.board import Board
from engine.enums import PlayerColor
from engine.game import Move
from typing import Optional, List, Dict

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
    def __init__(self):
        pass

    def predict(self, board: Board) -> tuple[float, Dict[Move, float]]:
        """
        returns a tuple of (v, pi)  
        where v is the value of the board and pi is a dictionary of moves with their probabilities
        """

        def compute_heuristic(board: Board) -> float:
            "Compute a heuristic value for the board state in [0, 1], where 1 is a win for white and 0 is a win for black"
            score =  (6 - board.count_queen_neighbors(board.current_player_color) + board.count_queen_neighbors(board.other_player_color)) / 12.0
            # score = board.count_queen_neighbors(board.other_player_color) / 6.0
            return score
        # For simplicity, let's assume the oracle always returns a uniform distribution over valid moves
        valid_moves = board.get_valid_moves()
        pi = {move: 1.0 / len(valid_moves) for move in valid_moves} if valid_moves else {}
        v = compute_heuristic(board)
        return v, pi