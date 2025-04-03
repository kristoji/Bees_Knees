from random import choice
import re
from hash import ZobristHash
from game import Position, Bug, Move
from typing import Final, Optional, Set
from enums import GameType, GameState, PlayerColor, BugName, BugType, Direction, Error, InvalidMoveError


class Board():
    ORIGIN: Final[Position] = Position(0, 0)
    NEIGHBOR_DELTAS: Final[
        dict[Direction, Position]
    ] = {
        Direction.RIGHT: Position(1, 0),
        Direction.UP_RIGHT: Position(1, -1),
        Direction.UP_LEFT: Position(0, -1),
        Direction.LEFT: Position(-1, 0),
        Direction.DOWN_LEFT: Position(-1, 1),
        Direction.DOWN_RIGHT: Position(0, 1),
        Direction.BELOW: Position(0, 0),
        Direction.ABOVE: Position(0, 0),
    }

    def __init__(self, gamestring: str = "") -> None:
        type_, state, turn, moves = self._parse_gamestring(gamestring)
        self.type: Final[GameType] = type_
        self.state: GameState = state
        self.turn: int = turn
        self.move_strings: list[str] = []
        self.moves: list[Optional[Move]] = []
        self._zobrist_hash: ZobristHash = ZobristHash()
        self._valid_moves_cache: dict[PlayerColor, Optional[Set[Move]]] = {PlayerColor.WHITE: None, PlayerColor.BLACK: None}
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
        self._play_initial_moves(moves)

    def __str__(self) -> str:
        moves_part = ";".join(self.move_strings) if self.moves else ""
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
                    self._zobrist_hash.toggle_last_moved_piece(BugName[str(self.moves[-1].bug)].value)
                self._zobrist_hash.toggle_turn_color()

            self.turn += 1
            if move_string is None:
                move_string = self.stringify_move(move)
            self.move_strings.append(move_string)
            self.moves.append(move)
            self._valid_moves_cache[self.other_player_color] = None
            
            if move:

                self._bug_to_pos[move.bug] = move.destination
                self._pos_to_bug.setdefault(move.destination, []).append(move.bug)
                if move.origin:
                    if update_hash:
                        self._zobrist_hash.toggle_piece(BugName[str(move.bug)].value, move.origin, len(self._bugs_from_pos(move.origin)))
                    self._pos_to_bug[move.origin].pop()
                
                if update_hash:
                    self._zobrist_hash.toggle_last_moved_piece(BugName[str(move.bug)].value)
                    self._zobrist_hash.toggle_piece(BugName[str(move.bug)].value, move.destination, len(self._bugs_from_pos(move.destination)))
                
                black_queen_surrounded = self.count_queen_neighbors(PlayerColor.BLACK) == 6
                white_queen_surrounded = self.count_queen_neighbors(PlayerColor.WHITE) == 6
                if black_queen_surrounded and white_queen_surrounded:
                    self.state = GameState.DRAW
                elif black_queen_surrounded:
                    self.state = GameState.WHITE_WINS
                elif white_queen_surrounded:
                    self.state = GameState.BLACK_WINS
        else:
            raise InvalidMoveError(
                f"You can't {'play' if move else Move.PASS} when the game is over"
            )

    def play(self, move_string: str, update_hash: bool = True) -> None:
        move = self._parse_move(move_string)
        self.safe_play(move, update_hash, move_string) 

    def undo(self, amount: int = 1, update_hash: bool = True) -> None:
        if self.state is not GameState.NOT_STARTED and len(self.moves) >= amount:
            if self.state is not GameState.IN_PROGRESS:
                self.state = GameState.IN_PROGRESS
            self._valid_moves_cache[self.current_player_color] = None
            if amount > 1:
                self._valid_moves_cache[self.other_player_color] = None
            for _ in range(amount):
                self.turn -= 1
                self.move_strings.pop()
                move = self.moves.pop()

                if update_hash:
                    if len(self.moves) and self.moves[-1]:
                        self._zobrist_hash.toggle_last_moved_piece(BugName[str(self.moves[-1].bug)].value)
                    self._zobrist_hash.toggle_turn_color()
                if move:
                    if update_hash:
                        self._zobrist_hash.toggle_last_moved_piece(BugName[str(move.bug)].value)
                        self._zobrist_hash.toggle_piece(BugName[str(move.bug)].value, move.destination, len(self._bugs_from_pos(move.destination)))
                    
                    self._pos_to_bug[move.destination].pop()
                    self._bug_to_pos[move.bug] = move.origin
                    if move.origin:
                        self._pos_to_bug[move.origin].append(move.bug)
                        if update_hash:
                            self._zobrist_hash.toggle_piece(BugName[str(move.bug)].value, move.origin, len(self._bugs_from_pos(move.origin)))
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
                    neighbor_bugs = self._bugs_from_pos(self._get_neighbor(move.destination, d))
                    if neighbor_bugs and neighbor_bugs[0] != moved:
                        relative = neighbor_bugs[0]
                        direction = d.opposite
                        break
            return Move.stringify(moved, relative, direction)
        return Move.PASS

    def count_queen_neighbors(self, color: PlayerColor = current_player_color) -> int:
        return sum(
            bool(self._bugs_from_pos(self._get_neighbor(queen_pos, d)))
            for d in Direction.flat() 
        ) if (queen_pos := self._pos_from_bug(Bug(color, BugType.QUEEN_BEE))) else 0

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

    def get_valid_moves(self, color: PlayerColor = None) -> Set[Move]:
        if color is None:
            color = self.current_player_color
        if self._valid_moves_cache[color] is None:
            moves = set()
            if self.state in (GameState.NOT_STARTED, GameState.IN_PROGRESS):
                for bug, pos in self._bug_to_pos.items():
                    if bug.color is color:
                        if self.turn == 0:
                            if bug.type is not BugType.QUEEN_BEE and self._can_bug_be_played(bug):
                                moves.add(Move(bug, None, self.ORIGIN))
                        elif self.turn == 1:
                            if bug.type is not BugType.QUEEN_BEE and self._can_bug_be_played(bug):
                                moves.update(
                                    Move(bug, None, self._get_neighbor(self.ORIGIN, d))
                                    for d in Direction.flat()
                                )
                        elif pos is None:
                            if self._can_bug_be_played(bug) and (
                                self.current_player_turn != 4
                                or self.current_player_queen_in_play
                                or bug.type is BugType.QUEEN_BEE
                            ):
                                moves.update(
                                    Move(bug, None, placement) for placement in self._get_valid_placements(color)
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
            self._valid_moves_cache[color] = moves
        return self._valid_moves_cache[color] or set()

    def _get_valid_placements(self, color: PlayerColor = current_player_color) -> Set[Position]:
        return {
            neighbor
            for bug, pos in self._bug_to_pos.items()
            if bug.color is color and pos and self._is_bug_on_top(bug)
            for direction in Direction.flat()
            for neighbor in [self._get_neighbor(pos, direction)]
            if not self._bugs_from_pos(neighbor)
            and all(
                not self._bugs_from_pos(self._get_neighbor(neighbor, d))
                or self._bugs_from_pos(self._get_neighbor(neighbor, d))[-1].color is color
                for d in Direction.flat()
                if d != direction.opposite
            )
        }

    def _get_sliding_moves(self, bug: Bug, origin: Position, depth: int = 0) -> Set[Move]:
        destinations: Set[Position] = set()
        visited: Set[Position] = set()
        stack: Set[tuple[Position, int]] = {(origin, 0)}
        unlimited_depth = depth == 0
        while stack:
            current, current_depth = stack.pop()
            visited.add(current)
            if unlimited_depth or current_depth == depth:
                destinations.add(current)
            if unlimited_depth or current_depth < depth:
                for d in Direction.flat():
                    neighbor = self._get_neighbor(current, d)
                    if neighbor not in visited and not self._bugs_from_pos(neighbor):
                        right = self._get_neighbor(current, d.right_of)
                        left = self._get_neighbor(current, d.left_of)
                        # changes made here: he was not considering some moves (right/left != origin was outside)
                        if bool(right != origin and self._bugs_from_pos(right)) != bool(left != origin and self._bugs_from_pos(left)):
                            stack.add((neighbor, current_depth + 1))
        return {Move(bug, origin, dest) for dest in destinations if dest != origin}

    def _get_beetle_moves(self, bug: Bug, origin: Position, virtual: bool = False) -> Set[Move]:
        moves: Set[Move] = set()
        height = len(self._bugs_from_pos(origin)) - 1 + int(virtual)
        for d in Direction.flat():
            destination = self._get_neighbor(origin, d)
            dest_height = len(self._bugs_from_pos(destination))
            left_height = len(self._bugs_from_pos(self._get_neighbor(origin, d.left_of)))
            right_height = len(self._bugs_from_pos(self._get_neighbor(origin, d.right_of)))
            if not ((height == 0 and dest_height == 0 and left_height == 0 and right_height == 0)
                    or (dest_height < left_height and dest_height < right_height and height < left_height and height < right_height)):
                moves.add(Move(bug, origin, destination))
        return moves

    def _get_grasshopper_moves(self, bug: Bug, origin: Position) -> Set[Move]:
        moves: Set[Move] = set()
        for d in Direction.flat():
            destination = self._get_neighbor(origin, d)
            distance = 0
            while self._bugs_from_pos(destination):
                destination = self._get_neighbor(destination, d)
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
            neighbor_pos = self._get_neighbor(origin, d)
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
        empty_positions = [
            self._get_neighbor(origin, d)
            for d in Direction.flat()
            if not self._bugs_from_pos(self._get_neighbor(origin, d))
        ]
        moves: Set[Move] = set()
        if empty_positions:
            for d in Direction.flat():
                source = self._get_neighbor(origin, d)
                bugs = self._bugs_from_pos(source)
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
        # assert self._bugs_from_pos(position)
        neighbors = [self._bugs_from_pos(self._get_neighbor(position, d)) for d in Direction.flat()]
        if sum(bool(neighbors[i] and not neighbors[i - 1]) for i in range(len(neighbors))) > 1:
            visited: set[Position] = set()
            # neighbors_pos = [self._pos_from_bug(bugs[-1]) for bugs in neighbors if bugs]
            neighbors_pos = [self._pos_from_bug(bugs[-1]) for bugs in neighbors if bugs and (self._pos_from_bug(bugs[-1]) is not None)]
            stack: set[Position] = {neighbors_pos[0]} if neighbors_pos else set()
            while stack:
                current = stack.pop()
                visited.add(current)
                for d in Direction.flat():
                    neighbor = self._get_neighbor(current, d)
                    if neighbor != position and self._bugs_from_pos(neighbor) and neighbor not in visited:
                        stack.add(neighbor)
            return all(pos in visited for pos in neighbors_pos)
        return True

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
                    move = Move(moved, self._pos_from_bug(moved), self._get_neighbor(relative_pos, Direction(f"{left_dir}|") if left_dir else Direction(f"|{right_dir or ""}")))
                    if move in self.get_valid_moves():
                        return move
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

    def _get_neighbor(self, position: Position, direction: Direction) -> Position:
        return position + self.NEIGHBOR_DELTAS[direction]



    # '''
    # STARTING FUNCTIONS FOR MCTS

    # TODO: 
    # spostare i metodi: troppa memoria sprecata copiando le boards
    # si potrebbe copiare solo quella iniziale e da lì si gioca quando espandiamo
    # '''
    # def find_children(self):
    #     "All possible successors of this board state"
    #     if self.is_terminal():  # If the game is finished then no moves can be made
    #         return set()
    #     # Otherwise, you can make a move in each of the empty spots
    #     children: Set[Board] = set()
    #     for move in self.valid_moves.split(";"):
    #         new_board = deepcopy(self)
    #         new_board.play(move)
    #         children.add(new_board)
    #     return children

    # def find_random_child(self):
    #     moves = self.valid_moves.split(";")
    #     rnd_move_str = choice(list(moves))
    #     new_board = deepcopy(self)
    #     new_board.play(rnd_move_str)
    #     return new_board

    # def is_terminal(self):
    #     "Returns True if the node has no children"
    #     if self.state == GameState.DRAW or self.state == GameState.BLACK_WINS or self.state == GameState.WHITE_WINS:
    #         return True
    #     else:
    #         return False

    # def reward(self):
    #     "Use the NeuralNetwork to get v (probability value)"
    #     if not self.is_terminal():
    #         # Use the Neural network to compute v
    #         v = 0.5
    #         return v
    #     if self.state == GameState.DRAW:
    #         return 0.5
    #     elif self.state.BLACK_WINS and self.other_player_color==PlayerColor.BLACK or self.state.WHITE_WINS and self.other_player_color==PlayerColor.WHITE:
    #         return 1
    #     else:
    #         return 0
        

    def __hash__(self):
        return self.zobrist_key

    def __eq__(node1, node2):
        return node1.zobrist_key == node2.zobrist_key