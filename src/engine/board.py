from collections import defaultdict
from random import choice, Random as _Random
import re
from engine.hash import ZobristHash
from engine.game import Position, Bug, Move, NEIGHBOR_INDICES
from typing import Final, Optional, Set
from engine.enums import GameType, GameState, PlayerColor, BugName, BugType, Direction, Error, InvalidMoveError
import inspect

# Distinguishes the plies where the opening rules still change which moves are legal
# (see get_valid_moves). Fixed values keep cache keys stable across processes.
_OPENING_TURN_SALT: Final[tuple[int, ...]] = (
    0x1d8e4e27c47d124f, 0x9c3a7f1b6e5d2a83, 0x4f6b8d2e1a9c5730, 0xe7215c93bd0a48f6,
    0x3a5e9071cf2b8d46, 0x82c4d6a3195f7be0, 0x6d039b5ea87c142f, 0xb51782fc4e6d09a3,
)


_QUEEN: Final[dict] = {color: Bug(color, BugType.QUEEN_BEE) for color in PlayerColor}

# Articulation points depend only on which cells are occupied, not on which bug sits
# where, how tall the stacks are or whose turn it is. Keying their cache on a hash of
# the occupied set therefore hits far more often than keying it on the zobrist.
_shape_rng = _Random(0x5DEECE66D)
_SHAPE_RANDOM: Final[tuple[int, ...]] = tuple(_shape_rng.getrandbits(64) for _ in range(4096))
del _shape_rng


class Board():
    # ORIGIN: Final[Position] = Position(0, 0)
    ORIGIN: Final[Position] = Position.POSITIONS[(0, 0)]

    def __init__(self, gamestring: str = "") -> None:
        type_, state, turn, moves = self._parse_gamestring(gamestring)
        self.type: Final[GameType] = type_
        self.state: GameState = state
        self.turn: int = turn
        self.move_strings: list[Optional[str]] = []
        self.moves: list[Optional[Move]] = []
        self._zobrist_hash: ZobristHash = ZobristHash()
        self._pos_to_bug: dict[Position, list[Bug]] = {}
        self._bug_to_pos: dict[Bug, Optional[Position]] = {}
        for color in PlayerColor:
            for expansion in self.type:
                if expansion is GameType.Base:
                    self._bug_to_pos[Bug(color, BugType.QUEEN_BEE)] = None
                    for i in range(1, 3):
                        self._bug_to_pos[Bug(color, BugType.SPIDER, i)] = None
                        self._bug_to_pos[Bug(color, BugType.BEETLE, i)] = None
                        self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, i)] = None
                        self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, i)] = None
                    self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, 3)] = None
                    self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, 3)] = None
                else:
                    self._bug_to_pos[Bug(color, BugType(expansion.name))] = None
        self._draw_counter: dict[int, int] = defaultdict(lambda: 0)
        # Indices of the currently occupied cells, and an incremental hash of that
        # set. Both are maintained by safe_play/undo.
        self._occupied: set[int] = set()
        self._shape_key: int = 0
        self._art_pos: set[int] = set()
        self._snapshots: dict[int, set[Move]] = {}
        self._snapshots_art_pos: dict[int, set[int]] = {}
        self._play_initial_moves(moves)

    def __str__(self) -> str:
        moves_part = ";".join(m if m is not None else "?" for m in self.move_strings) if self.moves else ""
        return (
            f"{self.type};{self.state};{self.current_player_color}[{self.current_player_turn}]"
            f"{';' if moves_part else ''}{moves_part}"
            # f"\n[DBG] hash:{self._zobrist_hash}"
        )

    @property
    def current_player_color(self) -> PlayerColor:
        return PlayerColor.WHITE if self.turn % 2 == 0 else PlayerColor.BLACK
    
    @property 
    def other_player_color(self) -> PlayerColor:
        return PlayerColor.WHITE if self.turn % 2 == 1 else PlayerColor.BLACK

    @property
    def current_player_turn(self) -> int:
        return 1 + self.turn // 2

    @property
    def current_player_queen_in_play(self) -> bool:
        return bool(self._bug_to_pos.get(Bug(self.current_player_color, BugType.QUEEN_BEE)))

    @property
    def valid_moves(self) -> str:
        return ";".join(self.stringify_move(m) for m in self.get_valid_moves()) or Move.PASS

    @property
    def zobrist_key(self) -> int:
        return self._zobrist_hash.value

    def safe_play(self, move: Move, update_hash: bool = True, move_string: str = None) -> None:
        if self.state is GameState.NOT_STARTED:
            self.state = GameState.IN_PROGRESS

        if self.state is GameState.IN_PROGRESS:
            if update_hash:
                if len(self.moves) and self.moves[-1]:
                    self._zobrist_hash.toggle_last_moved_piece(self.moves[-1].bug.index)
                self._zobrist_hash.toggle_turn_color()

            self.turn += 1
            # move_string is None only on the search path (safe_play with a Move), and
            # nothing reads move_strings there; play() always passes the real string.
            self.move_strings.append(move_string)
            self.moves.append(move)
            
            if move:

                self._bug_to_pos[move.bug] = move.destination
                dest_stack = self._pos_to_bug.get(move.destination)
                if dest_stack:
                    dest_stack.append(move.bug)
                else:
                    if dest_stack is None:
                        self._pos_to_bug[move.destination] = [move.bug]
                    else:
                        dest_stack.append(move.bug)
                    dest_index = move.destination.index
                    self._occupied.add(dest_index)
                    self._shape_key ^= _SHAPE_RANDOM[dest_index]
                if move.origin:
                    if update_hash:
                        self._zobrist_hash.toggle_piece(move.bug.index, move.origin, len(self._bugs_from_pos(move.origin)))
                    origin_stack = self._pos_to_bug[move.origin]
                    origin_stack.pop()
                    if not origin_stack:
                        origin_index = move.origin.index
                        self._occupied.discard(origin_index)
                        self._shape_key ^= _SHAPE_RANDOM[origin_index]

                if update_hash:
                    self._zobrist_hash.toggle_last_moved_piece(move.bug.index)
                    self._zobrist_hash.toggle_piece(move.bug.index, move.destination, len(self._bugs_from_pos(move.destination)))
                
                self._update_cut_pos()

                black_queen_surrounded = self.count_queen_neighbors(PlayerColor.BLACK) == 6
                white_queen_surrounded = self.count_queen_neighbors(PlayerColor.WHITE) == 6
                if black_queen_surrounded and white_queen_surrounded:
                    self.state = GameState.DRAW
                elif black_queen_surrounded:
                    self.state = GameState.WHITE_WINS
                elif white_queen_surrounded:
                    self.state = GameState.BLACK_WINS

                self._draw_counter[self.zobrist_key] += 1
                if self._draw_counter[self.zobrist_key] > 2:
                    self.state = GameState.DRAW
        else:
            raise InvalidMoveError(
                f"You can't {'play' if move else Move.PASS} when the game is over"
            )

    def copy(self) -> "Board":
        """An independent board sharing the read-only caches.

        Cheap now that the Zobrist tables are global: it used to be unusable because
        every board carried its own 237MB table. The snapshot caches are keyed on
        values that are global too (the zobrist key and the hive shape hash), so the
        copy shares them instead of starting cold; neither is ever mutated in place.
        """
        other = Board.__new__(Board)
        other.type = self.type
        other.state = self.state
        other.turn = self.turn
        other.move_strings = list(self.move_strings)
        other.moves = list(self.moves)
        other._zobrist_hash = ZobristHash()
        other._zobrist_hash.value = self._zobrist_hash.value
        other._pos_to_bug = {pos: list(bugs) for pos, bugs in self._pos_to_bug.items() if bugs}
        other._bug_to_pos = dict(self._bug_to_pos)
        other._draw_counter = defaultdict(int, self._draw_counter)
        other._occupied = set(self._occupied)
        other._shape_key = self._shape_key
        other._art_pos = self._art_pos
        other._snapshots = self._snapshots
        other._snapshots_art_pos = self._snapshots_art_pos
        return other

    def play(self, move_string: str, update_hash: bool = True) -> None:
        move = self._parse_move(move_string)
        self.safe_play(move, update_hash, move_string) 

    def undo(self, amount: int = 1, update_hash: bool = True) -> None:
        if self.state is not GameState.NOT_STARTED and len(self.moves) >= amount:
            if self.state is not GameState.IN_PROGRESS:
                self.state = GameState.IN_PROGRESS
            for _ in range(amount):
                self.turn -= 1
                self.move_strings.pop()
                move = self.moves.pop()
                if move:
                    # safe_play only counts real moves, so a pass must not decrement here
                    # (the key is still the post-move one: the hash is untoggled below).
                    self._draw_counter[self.zobrist_key] -= 1

                if update_hash:
                    if len(self.moves) and self.moves[-1]:
                        self._zobrist_hash.toggle_last_moved_piece(self.moves[-1].bug.index)
                    self._zobrist_hash.toggle_turn_color()
                if move:
                    if update_hash:
                        self._zobrist_hash.toggle_last_moved_piece(move.bug.index)
                        self._zobrist_hash.toggle_piece(move.bug.index, move.destination, len(self._bugs_from_pos(move.destination)))
                    
                    dest_stack = self._pos_to_bug[move.destination]
                    dest_stack.pop()
                    if not dest_stack:
                        dest_index = move.destination.index
                        self._occupied.discard(dest_index)
                        self._shape_key ^= _SHAPE_RANDOM[dest_index]
                    self._bug_to_pos[move.bug] = move.origin
                    if move.origin:
                        origin_stack = self._pos_to_bug.get(move.origin)
                        if origin_stack:
                            origin_stack.append(move.bug)
                        else:
                            if origin_stack is None:
                                self._pos_to_bug[move.origin] = [move.bug]
                            else:
                                origin_stack.append(move.bug)
                            origin_index = move.origin.index
                            self._occupied.add(origin_index)
                            self._shape_key ^= _SHAPE_RANDOM[origin_index]
                        if update_hash:
                            self._zobrist_hash.toggle_piece(move.bug.index, move.origin, len(self._bugs_from_pos(move.origin)))
            self._update_cut_pos()
            if self.turn == 0:
                self.state = GameState.NOT_STARTED
        else:
            raise Error(f"Unable to undo {amount} moves")

    def stringify_move(self, move: Optional[Move]) -> str:
        if move:
            moved = move.bug
            relative = None
            direction = None
            dest_bugs = self._bugs_from_pos(move.destination)
            if dest_bugs:
                relative = dest_bugs[-1]
            else:
                for d in Direction.flat():
                    neighbor_bugs = self._bugs_from_pos(move.destination.get_neighbor(d))
                    if neighbor_bugs and neighbor_bugs[0] != moved:
                        relative = neighbor_bugs[0]
                        direction = d.opposite
                        break
            return Move.stringify(moved, relative, direction)
        return Move.PASS


    def _update_cut_pos(self) -> None:
        """Recompute the articulation points of the hive.

        Cached on the shape of the hive rather than on the zobrist key: which bug sits
        where, how tall the stacks are and whose turn it is do not change which cells
        are cut vertices, so many distinct zobrist keys share one answer.

        Tarjan's algorithm runs iteratively over dense cell indices; the recursive
        version over Position objects was the single hottest thing in a search, and
        every set/dict probe went through a Python-level Position.__hash__.
        """
        cached = self._snapshots_art_pos.get(self._shape_key)
        if cached is not None:
            self._art_pos = cached
            return

        occupied = self._occupied
        if not occupied:
            return

        art: set[int] = set()
        disc: dict[int, int] = {}
        low: dict[int, int] = {}
        parent: dict[int, int] = {}
        neighbors = NEIGHBOR_INDICES

        root = next(iter(occupied))
        disc[root] = low[root] = 0
        timer = 1
        root_children = 0
        stack = [(root, iter(neighbors[root]))]

        while stack:
            u, it = stack[-1]
            descended = False
            for v in it:
                if v not in occupied:
                    continue
                if v not in disc:
                    parent[v] = u
                    if u == root:
                        root_children += 1
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, iter(neighbors[v])))
                    descended = True
                    break
                if v != parent.get(u):
                    dv = disc[v]
                    if dv < low[u]:
                        low[u] = dv
            if not descended:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    lu = low[u]
                    if lu < low[p]:
                        low[p] = lu
                    if p != root and lu >= disc[p]:
                        art.add(p)
        if root_children > 1:
            art.add(root)

        self._art_pos = art
        self._snapshots_art_pos[self._shape_key] = art

    def count_queen_neighbors(self, color: PlayerColor) -> int:
        queen_pos = self._bug_to_pos.get(_QUEEN[color])
        if queen_pos is None:
            return 0
        occupied = self._occupied
        n = 0
        for idx in queen_pos.neighbor_indices:
            if idx in occupied:
                n += 1
        return n

    def _parse_turn(self, turn: str) -> int:
        if (match := re.fullmatch(f"({PlayerColor.WHITE}|{PlayerColor.BLACK})\\[(\\d+)\\]", turn)):
            color, player_turn = match.groups()
            turn_number = int(player_turn)
            if turn_number > 0:
                return 2 * turn_number - 2 + list(PlayerColor).index(PlayerColor(color))
            raise Error("The turn number must be greater than 0")
        raise Error(f"'{turn}' is not a valid TurnString")

    def _parse_gamestring(self, gamestring: str) -> tuple[GameType, GameState, int, list[str]]:
        values = gamestring.split(";") if gamestring else ["", "", f"{PlayerColor.WHITE}[1]"]
        if len(values) == 1:
            values += ["", f"{PlayerColor.WHITE}[1]"]
        elif len(values) < 3:
            raise Error(f"'{gamestring}' is not a valid GameString")
        type_str, state_str, turn_str, *moves = values
        return GameType.parse(type_str), GameState.parse(state_str), self._parse_turn(turn_str), moves

    def _play_initial_moves(self, moves: list[str]) -> None:
        if self.turn == len(moves):
            old_turn, old_state = self.turn, self.state
            self.turn, self.state = 0, GameState.NOT_STARTED
            for move in moves:
                self.play(move)
            if old_turn != self.turn:
                raise Error(
                    f"TurnString is not correct, should be {self.current_player_color}[{self.current_player_turn}]"
                )
            if old_state != self.state:
                raise Error(f"GameStateString is not correct, should be {self.state}")
        else:
            raise Error(f"Expected {self.turn} moves but got {len(moves)}")

    def get_valid_moves(self) -> Set[Move]:
        # Zobrist alone is not enough while the opening rules still apply: the first
        # two plies have their own placement rules and the queen must be down by the
        # fourth turn of each player (turn <= 7), but the key only encodes the parity
        # of the turn. Salt the key over that window so a transposition back to an
        # early position cannot reuse a move set generated under different rules.
        turn = self.turn
        cache_key = self._zobrist_hash.value
        if turn <= 7:
            cache_key ^= _OPENING_TURN_SALT[turn]
        if cache_key not in self._snapshots:
            moves = set()
            if self.state in (GameState.NOT_STARTED, GameState.IN_PROGRESS):
                # self._update_cut_pos()
                for bug, pos in self._bug_to_pos.items():
                    if bug.color is self.current_player_color:
                        if self.turn == 0:
                            if bug.type is not BugType.QUEEN_BEE and self._can_bug_be_played(bug):
                                moves.add(Move(bug, None, self.ORIGIN))
                        elif self.turn == 1:
                            if bug.type is not BugType.QUEEN_BEE and self._can_bug_be_played(bug):
                                moves.update(
                                    Move(bug, None, self.ORIGIN.get_neighbor(d))
                                    for d in Direction.flat()
                                )
                        elif pos is None:
                            if self._can_bug_be_played(bug) and (
                                self.current_player_turn != 4
                                or self.current_player_queen_in_play
                                or bug.type is BugType.QUEEN_BEE
                            ):
                                moves.update(
                                    Move(bug, None, placement) for placement in self._get_valid_placements(self.current_player_color)
                                )
                        elif self.current_player_queen_in_play and self._bugs_from_pos(pos)[-1] == bug and self._was_not_last_moved(bug):
                            if len(self._bugs_from_pos(pos)) > 1 or self._can_move_without_breaking_hive(pos):
                                match bug.type:
                                    case BugType.QUEEN_BEE:
                                        moves.update(self._get_sliding_moves(bug, pos, 1))
                                    case BugType.SPIDER:
                                        moves.update(self._get_sliding_moves(bug, pos, 3))
                                    case BugType.BEETLE:
                                        moves.update(self._get_beetle_moves(bug, pos))
                                    case BugType.GRASSHOPPER:
                                        moves.update(self._get_grasshopper_moves(bug, pos))
                                    case BugType.SOLDIER_ANT:
                                        moves.update(self._get_sliding_moves(bug, pos))
                                    case BugType.MOSQUITO:
                                        moves.update(self._get_mosquito_moves(bug, pos))
                                    case BugType.LADYBUG:
                                        moves.update(self._get_ladybug_moves(bug, pos))
                                    case BugType.PILLBUG:
                                        moves.update(self._get_sliding_moves(bug, pos, 1))
                                        moves.update(self._get_pillbug_special_moves(pos))
                            else:
                                match bug.type:
                                    case BugType.MOSQUITO:
                                        moves.update(self._get_mosquito_moves(bug, pos, True))
                                    case BugType.PILLBUG:
                                        moves.update(self._get_pillbug_special_moves(pos))
            self._snapshots[cache_key] = moves
        return self._snapshots[cache_key]

    def _get_valid_placements(self, color: PlayerColor) -> Set[Position]:
        pos_to_bug = self._pos_to_bug
        placements: set[Position] = set()
        for bug, pos in self._bug_to_pos.items():
            if bug.color is not color or pos is None or not self._is_bug_on_top(bug):
                continue
            for i, neighbor in enumerate(pos.flat_neighbors):
                if neighbor in placements or pos_to_bug.get(neighbor):
                    continue
                # The direction back towards `pos` is skipped: that is the bug we are
                # placing next to, and it is ours by construction.
                back = i - 3 if i >= 3 else i + 3
                ok = True
                for j, around in enumerate(neighbor.flat_neighbors):
                    if j == back:
                        continue
                    bugs = pos_to_bug.get(around)
                    if bugs and bugs[-1].color is not color:
                        ok = False
                        break
                if ok:
                    placements.add(neighbor)
        return placements

    def _check_no_door(self, origin: Position, position: Position, direction: Direction) -> bool:
        return (origin in ((right := position.get_neighbor(direction.right_of)), (left := position.get_neighbor(direction.left_of)))) == (bool(self._bugs_from_pos(right)) == bool(self._bugs_from_pos(left)))

    def _get_sliding_moves(self, bug: Bug, origin: Position, depth: int = 0) -> Set[Move]:
        occupied = self._occupied
        destinations: set[Position] = set()
        # Kept as a set, like the original: two different paths reaching the same cell
        # at the same depth are distinct states (the path blocks backtracking), so only
        # exactly identical triples may be collapsed.
        stack: set[tuple[Position, int, frozenset[Position]]] = {(origin, 0, frozenset({origin}))}
        unlimited_depth = depth == 0
        while stack:
            current, current_depth, path = stack.pop()
            if unlimited_depth or current_depth == depth:
                destinations.add(current)
            if not (unlimited_depth or current_depth < depth):
                continue
            neighbors = current.flat_neighbors
            next_depth = current_depth + 1
            for i in range(6):
                neighbor = neighbors[i]
                if neighbor in path or neighbor.index in occupied:
                    continue
                # "no door": the gate formed by the two cells flanking the step must
                # not be closed. right_of(i) is i-1 and left_of(i) is i+1, mod 6.
                right = neighbors[i - 1 if i else 5]
                left = neighbors[i + 1 if i < 5 else 0]
                if (origin is right or origin is left) != (
                    (right.index in occupied) == (left.index in occupied)
                ):
                    continue
                stack.add((neighbor, next_depth, path | {neighbor}))
        return {Move(bug, origin, destination) for destination in destinations if destination is not origin}


    def _get_beetle_moves(self, bug: Bug, origin: Position, virtual: bool = False) -> Set[Move]:
        pos_to_bug = self._pos_to_bug
        neighbors = origin.flat_neighbors
        # Stack heights of the six neighbours, computed once: the gate check reads each
        # of them twice (as the left of one direction and the right of the next).
        heights = [len(bugs) if (bugs := pos_to_bug.get(n)) else 0 for n in neighbors]
        origin_bugs = pos_to_bug.get(origin)
        height = (len(origin_bugs) if origin_bugs else 0) - 1 + virtual
        moves: Set[Move] = set()
        for i in range(6):
            dest_height = heights[i]
            left_height = heights[i + 1 if i < 5 else 0]
            right_height = heights[i - 1 if i else 5]
            if not ((height == 0 and dest_height == 0 and left_height == 0 and right_height == 0)
                    or (dest_height < left_height and dest_height < right_height and height < left_height and height < right_height)):
                moves.add(Move(bug, origin, neighbors[i]))
        return moves

    def _get_grasshopper_moves(self, bug: Bug, origin: Position) -> Set[Move]:
        pos_to_bug = self._pos_to_bug
        moves: Set[Move] = set()
        for i, destination in enumerate(origin.flat_neighbors):
            distance = 0
            while pos_to_bug.get(destination):
                destination = destination.flat_neighbors[i]
                distance += 1
            if distance > 0:
                moves.add(Move(bug, origin, destination))
        return moves

    def _get_mosquito_moves(self, bug: Bug, origin: Position, special_only: bool = False) -> Set[Move]:
        if len(self._bugs_from_pos(origin)) > 1:
            return self._get_beetle_moves(bug, origin)
        moves: Set[Move] = set()
        bugs_copied: set[BugType] = set()
        for d in Direction.flat():
            neighbor_pos = origin.get_neighbor(d)
            bugs = self._bugs_from_pos(neighbor_pos)
            if bugs and (neighbor := bugs[-1]).type not in bugs_copied:
                bugs_copied.add(neighbor.type)
                if special_only:
                    if neighbor.type == BugType.PILLBUG:
                        moves.update(self._get_pillbug_special_moves(origin))
                else:
                    match neighbor.type:
                        case BugType.QUEEN_BEE:
                            moves.update(self._get_sliding_moves(bug, origin, 1))
                        case BugType.SPIDER:
                            moves.update(self._get_sliding_moves(bug, origin, 3))
                        case BugType.BEETLE:
                            moves.update(self._get_beetle_moves(bug, origin))
                        case BugType.GRASSHOPPER:
                            moves.update(self._get_grasshopper_moves(bug, origin))
                        case BugType.SOLDIER_ANT:
                            moves.update(self._get_sliding_moves(bug, origin))
                        case BugType.LADYBUG:
                            moves.update(self._get_ladybug_moves(bug, origin))
                        case BugType.PILLBUG:
                            moves.update(self._get_sliding_moves(bug, origin, 1))
                            moves.update(self._get_pillbug_special_moves(origin))
                        case BugType.MOSQUITO:
                            pass
        return moves

    def _get_ladybug_moves(self, bug: Bug, origin: Position) -> Set[Move]:
        return {
            Move(bug, origin, final_move.destination)
            for first_move in self._get_beetle_moves(bug, origin, True)
            if self._bugs_from_pos(first_move.destination)
            for second_move in self._get_beetle_moves(bug, first_move.destination, True)
            if self._bugs_from_pos(second_move.destination) and second_move.destination != origin
            for final_move in self._get_beetle_moves(bug, second_move.destination, True)
            if not self._bugs_from_pos(final_move.destination) and final_move.destination != origin
        }

    def _get_pillbug_special_moves(self, origin: Position) -> Set[Move]:
        pos_to_bug = self._pos_to_bug
        occupied = self._occupied
        neighbors = origin.flat_neighbors
        empty_positions = [n for n in neighbors if n.index not in occupied]
        moves: Set[Move] = set()
        if empty_positions:
            for source in neighbors:
                bugs = pos_to_bug.get(source) or ()  # .get gives None, the old helper gave []
                if (len(bugs) == 1 
                    and self._was_not_last_moved(move_bug := bugs[-1]) 
                    and self._can_move_without_breaking_hive(source) 
                    and Move(move_bug, source, origin) in self._get_beetle_moves(move_bug, source)
                ):
                    moves.update(
                        Move(move_bug, source, m.destination)
                        # changes made here: he was using from source instead of origin
                        for m in self._get_beetle_moves(move_bug, origin, True)
                        if m.destination in empty_positions
                    )
        return moves

    def _can_move_without_breaking_hive(self, position: Position) -> bool:
        # # assert self._bugs_from_pos(position)
        # neighbors = [self._bugs_from_pos(position.get_neighbor(d)) for d in Direction.flat()]
        # if sum(bool(neighbors[i] and not neighbors[i - 1]) for i in range(len(neighbors))) > 1:
        #     visited: set[Position] = set()
        #     # neighbors_pos = [self._pos_from_bug(bugs[-1]) for bugs in neighbors if bugs]
        #     neighbors_pos = [self._pos_from_bug(bugs[-1]) for bugs in neighbors if bugs and (self._pos_from_bug(bugs[-1]) is not None)]
        #     stack: set[Position] = {neighbors_pos[0]} if neighbors_pos else set()
        #     while stack:
        #         current = stack.pop()
        #         visited.add(current)
        #         for d in Direction.flat():
        #             neighbor = current.get_neighbor(d)
        #             if neighbor != position and self._bugs_from_pos(neighbor) and neighbor not in visited:
        #                 stack.add(neighbor)
        #     return all(pos in visited for pos in neighbors_pos)
        # return True
        return position.index not in self._art_pos

    def _can_bug_be_played(self, piece: Bug) -> bool:
        # assert piece.pos is None
        return all(
            bug.id >= piece.id
            for bug, pos in self._bug_to_pos.items()
            if pos is None and bug.type is piece.type and bug.color is piece.color
        )

    def _was_not_last_moved(self, bug: Bug) -> bool:
        return not self.moves[-1] or self.moves[-1].bug != bug

    def _parse_move(self, move_string: str) -> Optional[Move]:
        if move_string == Move.PASS:
            if not self.get_valid_moves():
                return None
            raise InvalidMoveError("You can't pass when you have valid moves")
        if (match := re.fullmatch(Move.REGEX, move_string)):
            bug_string_1, _, _, _, _, left_dir, bug_string_2, _, _, _, right_dir = match.groups()
            if not left_dir or not right_dir:
                moved = Bug.parse(bug_string_1)
                if (relative_pos := self._pos_from_bug(Bug.parse(bug_string_2)) if bug_string_2 else self.ORIGIN):
                    # Fix the f-string syntax issue by constructing the direction string separately
                    if left_dir:
                        direction_str = f"{left_dir}|"
                    else:
                        direction_str = f"|{right_dir or ''}"
                    move = Move(moved, self._pos_from_bug(moved), relative_pos.get_neighbor(Direction(direction_str)))
                    if move in self.get_valid_moves():
                        return move
                    print("VALID_MOVES", self.get_valid_moves())
                    raise InvalidMoveError(f"'{move_string}' is not a valid move for the current board state")
                raise InvalidMoveError(f"'{bug_string_2}' has not been played yet")
            raise InvalidMoveError("Only one direction at a time can be specified")
        raise InvalidMoveError(f"'{move_string}' is not a valid MoveString")

    def _is_bug_on_top(self, bug: Bug) -> bool:
        pos = self._pos_from_bug(bug)
        return pos is not None and self._bugs_from_pos(pos)[-1] == bug

    def _bugs_from_pos(self, position: Position) -> list[Bug]:
        return self._pos_to_bug.get(position, [])

    def _pos_from_bug(self, bug: Bug) -> Optional[Position]:
        return self._bug_to_pos.get(bug)


    def get_neighbor(self, position: Position, direction: Direction) -> Position:
        
        return position + Position.neighbor_delta(direction)

    def __hash__(self):
        return self.zobrist_key

    def __eq__(node1, node2):
        return node1.zobrist_key == node2.zobrist_key