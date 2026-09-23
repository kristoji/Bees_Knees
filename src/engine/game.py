from typing import Final, Optional
from engine.enums import Error, PlayerColor, BugType, BugName, Direction
import re


class Position:
    # Positions are interned in POSITIONS and compared by identity, so the per-instance
    # data can be precomputed once: `flat_neighbors` removes a match statement, a dict
    # lookup and a Position.__add__ from every neighbour walk, and `zobrist_base` (filled
    # in by engine.hash) removes the index arithmetic from every hash toggle.
    __slots__ = ("q", "r", "hash", "flat_neighbors", "zobrist_base")

    def __init__(self, q: int, r: int):
        self.r = r
        self.q = q
        self.hash = (q << 32) ^ (r & 0xffffffff)
        self.flat_neighbors: tuple = ()
        self.zobrist_base: int = 0

    def __str__(self) -> str:
        return f"({self.q}, {self.r})"

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: object) -> bool:
        return self is other

    def __add__(self, other: object):
        return Position.POSITIONS[(self.q + other.q, self.r + other.r)] if isinstance(other, Position) else NotImplemented

    def __sub__(self, other: object):
        return Position.POSITIONS[(self.q - other.q, self.r - other.r)] if isinstance(other, Position) else NotImplemented

    def get_neighbor(self, direction: Direction) -> "Position":
        idx = _DELTA_INDEX[direction]
        return self if idx > 5 else self.flat_neighbors[idx]


Position.POSITIONS = {
    (q, r): Position(q, r) for q in range(-32, 32) for r in range(-32, 32)
}

# Index order matches Direction.delta_index.
_DELTAS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
_DELTA_INDEX = {d: d.delta_index for d in Direction}

# A hive of 28 pieces grown from the origin can never reach |q| or |r| = 32, so the
# None at the boundary is unreachable; it is there only to keep the tuples uniform.
for _coords, _p in Position.POSITIONS.items():
    _q, _r = _coords
    _p.flat_neighbors = tuple(
        Position.POSITIONS.get((_q + _dq, _r + _dr)) for _dq, _dr in _DELTAS
    )
del _coords, _p, _q, _r

# class Position:
#     @countit
#     def __init__(self, q: int, r: int):
#         self.q: Final[int] = q
#         self.r: Final[int] = r
#         # self.hash = (q << 32) ^ (r & 0xffffffff)

#     def rotate_cw(self) -> "Position":
#         return Position(-self.r, self.q + self.r)

#     def to_oddr(self) -> tuple[int, int]:
#         col = self.q + (self.r - (self.r & 1)) // 2
#         row = self.r
#         return col, row
    
#     @staticmethod
#     def from_oddr(col: int, row: int) -> "Position":
#         q = col - (row - (row & 1)) // 2
#         r = row
#         return Position(q, r)

#     def __str__(self) -> str:
#         return f"({self.q}, {self.r})"

#     def __hash__(self) -> int:
#         return (self.q << 32) ^ (self.r & 0xffffffff)

#     def __eq__(self, other: object) -> bool:
#         return self is other or (isinstance(other, Position) and self.q == other.q and self.r == other.r)

#     def __add__(self, other: object):
#         return Position(self.q + other.q, self.r + other.r) if isinstance(other, Position) else NotImplemented

#     def __sub__(self, other: object):
#         return Position(self.q - other.q, self.r - other.r) if isinstance(other, Position) else NotImplemented
    
#     def get_neighbor(self, direction: Direction) -> "Position":
#         match direction:
#             case Direction.RIGHT:
#                 p = Position(1, 0)
#             case Direction.UP_RIGHT:
#                 p = Position(0, 1)
#             case Direction.UP_LEFT:
#                 p = Position(-1, 1)
#             case Direction.LEFT:
#                 p = Position(-1, 0)
#             case Direction.DOWN_LEFT:
#                 p = Position(0, -1)
#             case Direction.DOWN_RIGHT:
#                 p = Position(1, -1)
#             case Direction.BELOW:
#                 p = Position(0, 0)
#             case Direction.ABOVE:
#                 p = Position(0, 0)
#         return self + p

#     @staticmethod
#     def neighbor_delta(direction: Direction) -> "Position":
#         match direction:
#             case Direction.RIGHT:
#                 return Position(1, 0)
#             case Direction.UP_RIGHT:
#                 return Position(0, 1)
#             case Direction.UP_LEFT:
#                 return Position(-1, 1)
#             case Direction.LEFT:
#                 return Position(-1, 0)
#             case Direction.DOWN_LEFT:
#                 return Position(0, -1)
#             case Direction.DOWN_RIGHT:
#                 return Position(1, -1)
#             case Direction.BELOW:
#                 return Position(0, 0)
#             case Direction.ABOVE:
#                 return Position(0, 0)
            

class Bug:
    COLORS: Final[dict[str, PlayerColor]] = {color.code: color for color in PlayerColor}
    REGEX: Final[str] = (
        f"({'|'.join(COLORS.keys())})"
        f"({'|'.join(str(b) for b in BugType)})"
        f"(1|2|3)?"
    )

    @classmethod
    def parse(cls, bug_str: str) -> "Bug":
        if (match := re.fullmatch(cls.REGEX, bug_str)):
            color_code, type_str, bug_id = match.groups()
            return Bug(cls.COLORS[color_code], BugType(type_str), int(bug_id or 0))
        raise Error(f"'{bug_str}' is not a valid BugString")

    __slots__ = ("color", "type", "id", "name", "index")

    def __init__(self, color: PlayerColor, bug_type: BugType, bug_id: int = 0) -> None:
        self.color: Final[PlayerColor] = color
        self.type: Final[BugType] = bug_type
        self.id: Final[int] = bug_id
        # There are only 28 distinct bugs, so render the name and resolve its index
        # once. The hot path used to do BugName[str(bug)].value three times per
        # safe_play, and __hash__ used to build a string on every dict lookup.
        self.name: Final[str] = f"{color.code}{bug_type}{bug_id if bug_id else ''}"
        self.index: Final[int] = BugName[self.name].value

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other: object) -> bool:
        return (
            self is other
            or (isinstance(other, Bug)
                and self.color is other.color
                and self.type is other.type
                and self.id == other.id)
        )


class Move:
    PASS: Final[str] = "pass"
    REGEX: str = (
        f"({Bug.REGEX})"
        "( ?(" + '|'.join('\\' + str(d) for d in Direction.flat_left()) + ")?"
        f"({Bug.REGEX})"
        "(" + '|'.join('\\' + str(d) for d in Direction.flat_right()) + ")?)?"
    )

    @classmethod
    def stringify(
        cls, moved: Bug, relative: Optional[Bug] = None, direction: Optional[Direction] = None
    ) -> str:
        if relative:
            left_part = f"{direction}" if direction and direction.is_left else ""
            right_part = f"{direction}" if direction and direction.is_right else ""
            return f"{moved} {left_part}{relative}{right_part}"
        return f"{moved}"

    def __init__(self, bug: Bug, origin: Optional[Position], destination: "Position") -> None:
        self.bug: Final[Bug] = bug
        self.origin: Final[Optional[Position]] = origin
        self.destination: Final[Position] = destination

    def __str__(self) -> str:
        return f"{self.bug}, {self.origin}, {self.destination}"

    def __hash__(self) -> int:
        return hash((self.bug, self.origin, self.destination))

    def __eq__(self, other: object) -> bool:
        return (
            self is other
            or (isinstance(other, Move)
                and self.bug == other.bug
                and self.origin == other.origin
                and self.destination == other.destination)
        )
    
    def rotate_cw(self) -> "Move":
        return Move(self.bug, self.origin.rotate_cw() if self.origin else None, self.destination.rotate_cw())
    
    def center(self, pos: Position) -> "Move":
        # we dont care about origin bc we'll use this only to check where the bug will be placed
        return Move(self.bug, None, self.destination - pos)
    