from board import Board
from enums import BugType, PlayerColor, BugName
from game import Bug, Position, Move
from typing import Any, Optional, TypeAlias

# each bug has its own matrix of size SIZE x SIZE x LAYERS 
Bug_Matrix: TypeAlias = list[list[list[list[float | None]]]]

class Training:
    SIZE: int = 10
    LAYERS: int = 3
    CENTER: int = SIZE // 2 - 1
    NUM_PIECES: int = BugName.NumPieceNames.value

    @staticmethod
    def get_wQ_pos(board: Board) -> Optional[Position]:
        bug_pos: Position = board._pos_from_bug(Bug(PlayerColor.WHITE, BugType.QUEEN_BEE))

        return bug_pos

    @staticmethod
    def center_pos(bug_pos: Position, pos_to_bug: dict[Position, list[Bug]]) -> dict[Position, list[Bug]]:
        
        new_pos_to_bug: dict[Position, list[Bug]]= {}

        for pos, bugs in pos_to_bug.items():
            new_pos_to_bug[pos - bug_pos] = bugs

        return new_pos_to_bug
    
    @staticmethod
    def rotate_pos(pos_to_bug: dict[Position, Any]) -> dict[Position, Any]:
        temp_pos_to_bug: dict[Position, Any] = {}
        for key, value in pos_to_bug.items():
            rotated_key = key.rotate_cw()
            temp_pos_to_bug[rotated_key] = value
        pos_to_bug.clear()
        pos_to_bug.update(temp_pos_to_bug)
        return pos_to_bug

    @staticmethod
    def center_pi(bug_pos: Position, pi: dict[Move, float]) -> dict[Move, float]:
        
        new_pi: dict[Move, float]= {}

        for move, prob in pi.items():
            new_pi[move.center(bug_pos)] = prob

        return new_pi

    @staticmethod
    def rotate_pi(pi: dict[Move, float]) -> dict[Move, float]:
        temp_pi: dict[Move, float] = {}
        for move, prob in pi.items():
            rotated_move = move.rotate_cw()
            temp_pi[rotated_move] = prob
        pi.clear()
        pi.update(temp_pi)
        return pi
    
    @staticmethod
    def to_in_mat(pos_to_bug: dict[Position, list[Bug]], curr_player_color: PlayerColor) -> Bug_Matrix:
        if not pos_to_bug:
            return []

        # Initialize matrix
        matrix: Bug_Matrix = [[[[None for _ in range(Training.SIZE)] for _ in range(Training.SIZE)] for _ in range(Training.LAYERS)] for _ in range(Training.NUM_PIECES)]

        # Populate matrix
        for pos, bugs in pos_to_bug.items():
            col, row = pos.to_oddr()
            adj_col = col + Training.CENTER
            adj_row = row + Training.CENTER
            for i, bug in enumerate(bugs):
                matrix[BugName[str(bug)].value][i][adj_row][adj_col] = 1 if bug.color == curr_player_color else -1

        return matrix
    
    @staticmethod
    def to_out_mat(pos_to_bug: dict[Position, list[Bug]], curr_player_color: PlayerColor, pi: dict[Move, float] = None) -> Bug_Matrix:
        if not pos_to_bug:
            return []

        # Initialize matrix
        matrix: Bug_Matrix = [[[[None for _ in range(Training.SIZE)] for _ in range(Training.SIZE)] for _ in range(Training.LAYERS)] for _ in range(Training.NUM_PIECES)]

        # Populate matrix
        for move, prob in pi.items():
            col, row = move.destination.to_oddr()
            adj_col = col + Training.CENTER
            adj_row = row + Training.CENTER
            layer = len(pos_to_bug[move.destination])
            matrix[BugName[str(move.bug)].value][layer][adj_row][adj_col] = prob if move.bug.color == curr_player_color else -prob

        return matrix
    
    
    @staticmethod
    def get_matrices_from_center(s: Board, pi: dict[Move, float], center: Position) -> list[tuple[Bug_Matrix, Bug_Matrix]]:

        # we want as output a list of (in_mat, out_mat) to pass to the NN
        tp_out: list[tuple[Bug_Matrix, Bug_Matrix]] = []

        # center the input board pos_to_bug
        in_pos_to_bug_centered: dict[Position, list[Bug]] = Training.center_pos(center, s._pos_to_bug)

        # we need Move to distinguish same bugs
        pi_centered: dict[Move, float] = Training.center_pi(Training.get_wQ_pos(s), pi)

        # as out we only want moved bugs probabilities
        out_pos_to_bug: dict[Position, list[Bug]] = {}
        for move, prob in pi.items():
            out_pos_to_bug[move.destination] = [None] * len(s._pos_to_bug[move.destination]) + [move.bug]
        
        out_pos_to_bug_centered: dict[Position, list[Bug]] = Training.center_pos(Training.get_wQ_pos(s), out_pos_to_bug)

        # for each rotation, compute in_mat and out_mat
        for _ in range(6):
            in_pos_to_bug_centered: dict[Position, list[Bug]] = Training.rotate_pos(in_pos_to_bug_centered) #rotation
            in_mat = Training.to_in_mat(in_pos_to_bug_centered, s.current_player_color) #input matrix conversion

            Training.log_matrix(in_mat) #log

            out_pos_to_bug_centered: dict[Position, list[Bug]] = Training.rotate_pos(out_pos_to_bug_centered) #rotation
            pi_centered: dict[Move, float] = Training.rotate_pi(pi_centered) #probabilities rotation
            out_mat = Training.to_out_mat(out_pos_to_bug_centered, s.current_player_color, pi_centered) #output matrix

            Training.log_matrix(out_mat) #log

            tp_out.append((in_mat, out_mat)) #saving the tuple input-output matrix

        return tp_out
    
    @staticmethod
    def get_matrices_from_board(s: Board, pi: dict[Move, float]) -> list[tuple[Bug_Matrix, Bug_Matrix]]:
        # if there is no wQ, we center the input on each bug

        tp_out: list[tuple[Bug_Matrix, Bug_Matrix]] = []

        wQ_pos: Optional[Position] = Training.get_wQ_pos(s)
        if wQ_pos:
            return Training.get_matrices_from_center(s, pi, wQ_pos)
        else:
            for pos in s._bug_to_pos.values():
                tp_out += Training.get_matrices_from_center(s,pi,pos)
            return tp_out

    @staticmethod
    def log_matrix(matrix: Bug_Matrix) -> None:
        # Log the matrix to a file
        
        with open("test/log.txt", "a") as f:
            # f.write(f"matrix: {matrix}\n")
            f.write(f"DIM:{len(matrix)}x{len(matrix[0])}x{len(matrix[0][0])}\n\n ")
            for p, piece in enumerate(matrix):
                f.write(f"PIECE {BugName(p).name}\n")
                for i,width in enumerate(piece):
                    f.write(f"LAYER {i+1}\n")
                    for row in width:
                        f.write("["+", ".join([str(bug) if bug else "" for bug in row]) + "]\n")
            f.write("\nsesso e samba\n\n")
    
    # @staticmethod
    # def get_dict_from_matrix(matrix: Bug_Matrix) -> dict[Move, float]:
    #     # Convert the matrix back to a dictionary
    #     move_to_prob: dict[Move, float] = {}

    #     # Bug_Matrix = [[[[None for _ in range(Training.SIZE)] for _ in range(Training.SIZE)] for _ in range(Training.LAYERS)] for _ in range(BugName.NumPieceNames.value)]

    #     for p, piece in enumerate(matrix):
    #         bug = BugName[p]
    #         for i, layer in enumerate(piece):
    #             for r in row:
    #                 for c in col:
    #                     if matrix[p][i][r][c]>0: #if the probabiliti is not 0, we save it in the dict
    #                         color : PlayerColor = PlayerColor.WHITE if p<14 else PlayerColor.BLACK
    #                         bug_type : BugType = 
    #                         move_to_prob[Move(Bug(color, ), None, Position())] = matrix[p][i][r][c]



    @staticmethod
    def get_dict_from_matrix(matrix: Bug_Matrix, board: Board, abs_origin: Position) -> dict[Move, float]:

        move_to_prob: dict[Move, float] = {}

        # Check matrix dimensions safely
        if not matrix or not matrix[0] or not matrix[0][0] or not matrix[0][0][0]:
            return move_to_prob # Return empty dict for empty or improperly shaped matrices

        absolute_center_pos = Training.get_wQ_pos(board) or Position(0, 0)

        for p in range(Training.NUM_PIECES):
            bug: Bug = Bug.parse(BugName(p).name)
            origin_pos_abs: Position = board._pos_from_bug(bug)

            for i in range(Training.LAYERS):
                for r in range(Training.SIZE):
                    for c in range(Training.SIZE):

                        prob = matrix[p][i][r][c]
                        
                        # Check if the cell contains a probability value (is a float and not zero)
                        if isinstance(prob, float) and prob != 0.0:

                            dest_pos_rel = Position.from_oddr(c - Training.CENTER, r - Training.CENTER)
                            dest_pos_abs = absolute_center_pos + dest_pos_rel

                            move = Move(bug, origin_pos_abs, dest_pos_abs)
                            move_to_prob[move] = prob

        return move_to_prob



# class ZeroSumPerfInfoGame:
    
#     """
#     ZerosumPerfinfGame
#     .initial_state()->s
#     .next_state(s, a)->s’
#     .game_phase(s)-> winner
#     """
    
#     def __init__(self) -> None:
#         # TODO: generalizzare?
#         self._board = None

#     def initial_state(self):
#         self._board = Board("Base+MLP")
#         return self._board

#     def next_state(self, board: Board, action: Position) -> Board:
#         return board.play(action)