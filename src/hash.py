from random import getrandbits
from typing import Final
from game import Position

class ZobristHash:

    _NUM_PIECES: Final[int] = 28
    _BOARD_SIZE: Final[int] = 128
    _HALF_BOARD_SIZE: Final[int] = _BOARD_SIZE // 2
    _BOARD_STACK_SIZE: Final[int] = 8

    def __init__(self) -> None:
        self.value = 0
        self._hashPartByTurnColor: int = self._rand64()
        self._hashPartByLastMovedPiece: list[int] = [self._rand64() for _ in range(self._NUM_PIECES)]
        self._hashPartByPosition: list[list[list[int]]] = [[[[self._rand64() for _ in range(self._BOARD_STACK_SIZE+1)] for _ in range(self._BOARD_SIZE)] for _ in range(self._BOARD_SIZE)] for _ in range(self._NUM_PIECES)]

    def _rand64(self) -> int:
        return getrandbits(64)
    
    def toggle_turn_color(self) -> None:
        self.value ^= self._hashPartByTurnColor
    
    def toggle_last_moved_piece(self, piece_idx: int) -> None:
        self.value ^= self._hashPartByLastMovedPiece[piece_idx]
    
    def toggle_piece(self, piece_idx: int, pos: Position, stack_pos: int) -> None:
        self.value ^= self._hashPartByPosition[piece_idx][self._HALF_BOARD_SIZE + pos.q][self._HALF_BOARD_SIZE + pos.r][stack_pos+1]

    def __str__(self) -> str:
        return hex(self.value)