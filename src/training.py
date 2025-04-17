from board import Board
from enums import BugType, PlayerColor
from game import Bug, Position


class Training:

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
    def to_matrix(pos_to_bug: dict[Position, list[Bug]]) -> list[list[list[Bug]]]:
        # def to_matrix(pos_to_bug)
        # così modifichiamo il pos_to_bug centrando o ruotando
        """Converts the board into a matrix using odd-r offset coordinates."""
        if not pos_to_bug:
            return []

        # TODO:
        # rotations
        # center in wQ if present 
        # else (origini corrispondenti + ogni pezzo come centro + rotazioni)

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