from random import Random
from typing import Final
from engine.game import Position

# --- table geometry -------------------------------------------------------
_NUM_PIECES: Final[int] = 28
# Matches the range of Position.POSITIONS: q and r both live in [-32, 32).
_BOARD_SIZE: Final[int] = 64
_HALF_BOARD_SIZE: Final[int] = _BOARD_SIZE // 2
_BOARD_STACK_SIZE: Final[int] = 8
_STACK_SLOTS: Final[int] = _BOARD_STACK_SIZE + 1
_SEED: Final[int] = 0x9E3779B97F4A7C15

_R_STRIDE: Final[int] = _STACK_SLOTS
_Q_STRIDE: Final[int] = _BOARD_SIZE * _STACK_SLOTS
_PIECE_STRIDE: Final[int] = _BOARD_SIZE * _BOARD_SIZE * _STACK_SLOTS

# The tables are built once for the whole process. They used to be rebuilt per
# ZobristHash instance as a 4-level nested list (28 x 128 x 128 x 9 = 4.1M ints),
# which cost ~0.9s and ~237MB for every Board() and — worse — gave each board its
# own random values, so keys from two different boards were not comparable.
# A single flat tuple with a fixed seed builds in ~50ms and is shared and stable.
_rng = Random(_SEED)
_TURN_COLOR: Final[int] = _rng.getrandbits(64)
_LAST_MOVED: Final[tuple[int, ...]] = tuple(_rng.getrandbits(64) for _ in range(_NUM_PIECES))
_POSITION: Final[tuple[int, ...]] = tuple(
    _rng.getrandbits(64) for _ in range(_NUM_PIECES * _PIECE_STRIDE)
)
# Base offset of each piece index, so toggle_piece adds instead of multiplying.
_PIECE_BASE: Final[tuple[int, ...]] = tuple(i * _PIECE_STRIDE for i in range(_NUM_PIECES))
del _rng

# Every Position is interned in Position.POSITIONS, so its contribution to the table
# index can be baked in once instead of being recomputed on every toggle.
for _coords, _pos in Position.POSITIONS.items():
    _pos.zobrist_base = (
        (_HALF_BOARD_SIZE + _coords[0]) * _Q_STRIDE
        + (_HALF_BOARD_SIZE + _coords[1]) * _R_STRIDE
    )
del _coords, _pos


class ZobristHash:
    """Incremental Zobrist key for a board state."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0

    def toggle_turn_color(self) -> None:
        self.value ^= _TURN_COLOR

    def toggle_last_moved_piece(self, piece_idx: int) -> None:
        self.value ^= _LAST_MOVED[piece_idx]

    def toggle_piece(self, piece_idx: int, pos: Position, stack_pos: int) -> None:
        self.value ^= _POSITION[_PIECE_BASE[piece_idx] + pos.zobrist_base + stack_pos + 1]

    def __str__(self) -> str:
        return hex(self.value)
