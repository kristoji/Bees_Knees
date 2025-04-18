from board import Board
from enums import BugType, PlayerColor
from game import Bug, Position, Move


class Training:
    Matrix = list[list[list[Bug]]]

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
    def rotate_pos(pos_to_bug: dict[Position, list[Bug]]) -> dict[Position, list[Bug]]:
        # TODO: prova a non fare la copia
        new_pos_to_bug: dict[Position, list[Bug]]= {}

        for pos, bugs in pos_to_bug.items():
            new_pos_to_bug[pos.rotate_cw()] = bugs

        return new_pos_to_bug
    
    @staticmethod
    def to_matrix(pos_to_bug: dict[Position, list[Bug]], pi: dict[Move, float] =None) -> list[list[list[Bug]]]:
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
        matrix: list[list[list[Bug]]] = [[[None for _ in range(SIZE)] for _ in range(SIZE)] for _ in range(LAYERS)]

        # Populate matrix
        for pos, bugs in pos_to_bug.items():
            col, row = pos.to_oddr()
            adj_col = col + SIZE//2 - 1            
            adj_row = row + SIZE//2 - 1
            for i, bug in enumerate(bugs):
                matrix[i][adj_row][adj_col] = bug

        with open("test/log.txt", "a") as f:
            # f.write(f"matrix: {matrix}\n")
            f.write(f"DIM:{len(matrix)}x{len(matrix[0])}x{len(matrix[0][0])}\n\n ")
            for i,width in enumerate(matrix):
                f.write(f"LAYER {i+1}\n")
                for row in width:
                    f.write("["+", ".join([str(bug) if bug else "" for bug in row]) + "]\n")
            f.write("\nsesso e samba\n\n")

        return matrix
    
    @staticmethod
    def get_matrices(T_game_out: list[Matrix], s: Board, pi: dict[Move, float]) -> None:

        # in_mat = Training.to_matrix(s._pos_to_bug)
        # for move, prob in pi.items():

        #     s.safe_play(move)
            
        #     pos_to_bug: dict[Position, list[Bug]] = Training.center_pos(Training.get_wQ_pos(s), s._pos_to_bug)

        #     for _ in range(6):

        #         rotated_pos_to_bug: dict[Position, list[Bug]] = Training.rotate_pos(pos_to_bug)
        #         out_mat = Training.to_matrix(pos_to_bug, moved_bug, prob) #insert prob in out_mat in qualche modo

        #         T_game_out.append((in_mat, out_mat))

        in_mat = []

        in_pos_to_bug_centered: dict[Position, list[Bug]] = Training.center_pos(Training.get_wQ_pos(s), s._pos_to_bug)

        for i in range(6):
            
            rotated_pos_to_bug: dict[Position, list[Bug]] = Training.rotate_pos(in_pos_to_bug_centered)
            in_mat[i] = Training.to_matrix(rotated_pos_to_bug)


        # ----------------
        # for move, prob in pi.items():
        #     play
        #     center
        #     for _ in range(6)
        #         rotate
        #         out_mat[i] final_pos.append((bug, get_pos(bug)))
        #     undo 

        out_mats: list[Matrix] = []

        
        out_pos_to_bug: dict[Position, list[Bug]] = {}
        for move, prob in pi.items():
            out_pos_to_bug[move.destination] = [0] * len(in_pos_to_bug_centered[move.destination]) + [move.bug]
        
        out_pos_to_bug_centered: dict[Position, list[Bug]] = Training.center_pos(Training.get_wQ_pos(s), out_pos_to_bug)

        for _ in range(6):
            rotated_pos_to_bug: dict[Position, list[Bug]] = Training.rotate_pos(out_pos_to_bug_centered)
            out_mat = Training.to_matrix(rotated_pos_to_bug, pi)
            out_mats.append((out_mat, prob))
    
    

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