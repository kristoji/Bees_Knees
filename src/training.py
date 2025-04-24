from board import Board
from enums import BugType, PlayerColor, BugName
from game import Bug, Position, Move
from typing import Any


class Training:
    Matrix = list[list[list[list[Bug]]]]

    @staticmethod
    def get_wQ_pos(board: Board) -> Position:
        bug_pos: Position = board._pos_from_bug(Bug(PlayerColor.WHITE, BugType.QUEEN_BEE))

        return bug_pos or Position(0, 0)

    @staticmethod
    def center_pos(bug_pos: Position, pos_to_bug: dict[Position, list[Bug]]) -> dict[Position, list[Bug]]:
        
        new_pos_to_bug: dict[Position, list[Bug]]= {}

        for pos, bugs in pos_to_bug.items():
            new_pos_to_bug[pos - bug_pos] = bugs

        return new_pos_to_bug
    
    @staticmethod
    def rotate_pos(pos_to_bug: dict[Position, Any]) -> dict[Position, Any]:
        # TODO: prova a non fare la copia
        new_pos_to_bug: dict[Position, Any]= {}

        for pos, bugs in pos_to_bug.items():
            new_pos_to_bug[pos.rotate_cw()] = bugs

        return new_pos_to_bug

    @staticmethod
    def center_pi(bug_pos: Position, pi: dict[Move, float]) -> dict[Move, float]:
        
        new_pi: dict[Move, float]= {}

        for move, prob in pi.items():
            new_pi[move.center(bug_pos)] = prob

        return new_pi

    @staticmethod
    def rotate_pi(pi: dict[Move, float]) -> dict[Move, float]:
        # TODO: prova a non fare la copia
        new_pi: dict[Move, float]= {}

        for move, prob in pi.items():
            new_pi[move.rotate_cw()] = prob

        return new_pi
    
    @staticmethod
    def to_matrix(pos_to_bug: dict[Position, list[Bug]], curr_player_color: PlayerColor, pi: dict[Move, float] = None) -> list[list[list[Bug]]]:
        # def to_matrix(pos_to_bug)
        # così modifichiamo il pos_to_bug centrando o ruotando
        """Converts the board into a matrix using odd-r offset coordinates."""
        if not pos_to_bug:
            return []
        
        # TODO: 
        # matrice a 4 dim: 64x64x7xnum_pezzi
        # tenere in considerazione pi: se c'è come arg allora invece di +/- 1 bisogna mettere le prob

        # Initialize matrix
        SIZE = 10
        LAYERS = 3
        # TODO: inizializzare a 0
        matrix: list[list[list[list[Bug]]]] = [[[[None for _ in range(SIZE)] for _ in range(SIZE)] for _ in range(LAYERS)] for _ in range(BugName.NumPieceNames.value)]

        # Populate matrix
        # TODO: FIX HERE
        # ciclando nel pos_to_bug prendiamo solo le pos esistenti e non quelle in pi. 
        # se c'è pi, forse conviene ciclare su pi e basta!
        if pi is None:

            for pos, bugs in pos_to_bug.items():
                col, row = pos.to_oddr()
                adj_col = col + SIZE//2 - 1
                adj_row = row + SIZE//2 - 1
                for i, bug in enumerate(bugs):
                    matrix[BugName[str(bug)].value][i][adj_row][adj_col] = 1 if bug.color == curr_player_color else -1

        else:
            for move, prob in pi.items():
                col, row = move.destination.to_oddr()
                adj_col = col + SIZE//2 - 1
                adj_row = row + SIZE//2 - 1
                layer = len(pos_to_bug[move.destination])
                matrix[BugName[str(move.bug)].value][layer][adj_row][adj_col] = prob if move.bug.color == curr_player_color else -prob


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

        return matrix
    
    @staticmethod
    def get_matrices(s: Board, pi: dict[Move, float]) -> list[tuple[list[list[list[list[Bug]]]], list[list[list[list[Bug]]]]]]:

        # we want as output a list of (in_mat, out_mat) to pass to the NN
        tp_out: list[tuple[list[list[list[list[Bug]]]], list[list[list[list[Bug]]]]]] = []

        # center the input board pos_to_bug
        in_pos_to_bug_centered: dict[Position, list[Bug]] = Training.center_pos(Training.get_wQ_pos(s), s._pos_to_bug)

        # we need Move to distinguish same bugs
        pi_centered: dict[Move, float] = Training.center_pi(Training.get_wQ_pos(s), pi)

        # as out we only want moved bugs probabilities
        out_pos_to_bug: dict[Position, list[Bug]] = {}
        for move, prob in pi.items():
            out_pos_to_bug[move.destination] = [None] * len(s._pos_to_bug[move.destination]) + [move.bug]
        
        out_pos_to_bug_centered: dict[Position, list[Bug]] = Training.center_pos(Training.get_wQ_pos(s), out_pos_to_bug)

        # for each rotation, compute in_mat and out_mat
        for _ in range(1):
            in_pos_to_bug_centered: dict[Position, list[Bug]] = Training.rotate_pos(in_pos_to_bug_centered)
            in_mat = Training.to_matrix(in_pos_to_bug_centered, s.current_player_color)

            out_pos_to_bug_centered: dict[Position, list[Bug]] = Training.rotate_pos(out_pos_to_bug_centered)
            pi_centered: dict[Move, float] = Training.rotate_pi(pi_centered)
            for move, prob in pi_centered.items():
                print(f"{move} : {prob}")
            out_mat = Training.to_matrix(out_pos_to_bug_centered, s.current_player_color, pi_centered)
            tp_out.append((in_mat, out_mat))

        return tp_out

    
    

class ZeroSumPerfInfoGame:
    
    """
    ZerosumPerfinfGame
    .initial_state()->s
    .next_state(s, a)->s’
    .game_phase(s)-> winner
    """
    
    def __init__(self) -> None:
        # TODO: generalizzare?
        self._board = None

    def initial_state(self):
        self._board = Board("Base+MLP")
        return self._board

    def next_state(self, board: Board, action: Position) -> Board:
        return board.play(action)