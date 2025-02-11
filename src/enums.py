from enum import StrEnum, Flag, auto
from functools import reduce


class Command(StrEnum):
    INFO = "info"
    HELP = "help"
    OPTIONS = "options"
    NEWGAME = "newgame"
    VALIDMOVES = "validmoves"
    BESTMOVE = "bestmove"
    PLAY = "play"
    PASS = "pass"
    UNDO = "undo"
    EXIT = "exit"

    @classmethod
    def help_details(cls, command: "Command") -> str:
        help_detail = {
            Command.INFO: (
                f"  {Command.INFO}\n\n"
                "  Displays the engine's identifier and capabilities.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#info."
            ),
            Command.HELP: (
                f"  {Command.HELP}\n  {Command.HELP} [Command]\n\n"
                "  Lists available commands. If a command is specified, shows its help text."
            ),
            Command.OPTIONS: (
                f"  {Command.OPTIONS}\n  {Command.OPTIONS} get OptionName\n  {Command.OPTIONS} set OptionName OptionValue\n\n"
                "  Shows available engine options. Use 'get' or 'set' as needed.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#options."
            ),
            Command.NEWGAME: (
                f"  {Command.NEWGAME} [GameTypeString|GameString]\n\n"
                "  Starts a new Base game. Provide a game type or game string to load a game.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#newgame."
            ),
            Command.VALIDMOVES: (
                f"  {Command.VALIDMOVES}\n\n"
                "  Displays every valid move in the current game.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#validmoves."
            ),
            Command.BESTMOVE: (
                f"  {Command.BESTMOVE} time MaxTime\n  {Command.BESTMOVE} depth MaxTime\n\n"
                "  Searches for the best move. Limit by time (hh:mm:ss) or depth.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#bestmove."
            ),
            Command.PLAY: (
                f"  {Command.PLAY} MoveString\n\n"
                "  Plays the specified move in the current game.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#play."
            ),
            Command.PASS: (
                f"  {Command.PASS}\n\n"
                "  Plays a passing move.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#pass."
            ),
            Command.UNDO: (
                f"  {Command.UNDO} [MovesToUndo]\n\n"
                "  Undoes the last move. Optionally, specify how many moves to undo.\n"
                "  See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#undo."
            ),
            Command.EXIT: (
                f"  {Command.EXIT}\n\n"
                "  Exits the engine."
            ),
        }
        if command not in help_detail:
            return ""
        return help_detail[command]



class PlayerColor(StrEnum):
    WHITE = "White"
    BLACK = "Black"

    @property
    def code(self) -> str:
        return self[0].lower()


class GameState(StrEnum):
    NOT_STARTED = "NotStarted"
    IN_PROGRESS = "InProgress"
    DRAW = "Draw"
    WHITE_WINS = "WhiteWins"
    BLACK_WINS = "BlackWins"

    @classmethod
    def parse(cls, state: str) -> "GameState":
        return cls(state) if state else cls.NOT_STARTED


class GameType(Flag):
    Base = auto()
    M = auto()
    L = auto()
    P = auto()

    @classmethod
    def parse(cls, s: str) -> "GameType":
        if s:
            parts = s.split("+")
            try:
                base = cls[parts[0]]
            except KeyError:
                raise Error(f"'{s}' is not a valid GameType")
            if base != cls.Base or (len(parts) > 1 and parts[1] == "") or len(parts) > 2:
                raise Error(f"'{s}' is not a valid GameType")
            expansions = parts[1] if len(parts) == 2 else ""
            try:
                return reduce(lambda acc, ch: acc | cls[ch], expansions, base)
            except KeyError:
                raise Error(f"'{s}' is not a valid GameType")
        return cls.Base

    def __str__(self) -> str:
        return "".join(
            f"{gametype.name}{'+' if gametype is GameType.Base and len(self) > 1 else ''}"
            for gametype in self
        )


class BugType(StrEnum):
    QUEEN_BEE = "Q"
    SPIDER = "S"
    BEETLE = "B"
    GRASSHOPPER = "G"
    SOLDIER_ANT = "A"
    MOSQUITO = "M"
    LADYBUG = "L"
    PILLBUG = "P"


class Direction(StrEnum):
    RIGHT = "|-"
    UP_RIGHT = "|/"
    UP_LEFT = "\\|"
    LEFT = "-|"
    DOWN_LEFT = "/|"
    DOWN_RIGHT = '|\\'
    BELOW = ""
    ABOVE = "|"

    @classmethod
    def flat(cls) -> list["Direction"]:
        return [d for d in cls if d not in (cls.ABOVE, cls.BELOW)]

    @classmethod
    def flat_left(cls) -> list["Direction"]:
        return [d for d in cls if d.is_left]

    @classmethod
    def flat_right(cls) -> list["Direction"]:
        return [d for d in cls if d.is_right]

    def __str__(self) -> str:
        return self.replace("|", "")

    @property
    def opposite(self) -> "Direction":
        if self in (Direction.BELOW, Direction.ABOVE):
            return list(Direction)[(self.delta_index - 5) % 2 + 6]
        return list(Direction)[(self.delta_index + 3) % 6]

    @property
    def left_of(self) -> "Direction":
        if self in (Direction.BELOW, Direction.ABOVE):
            return self
        return list(Direction)[(self.delta_index + 1) % 6]

    @property
    def right_of(self) -> "Direction":
        if self in (Direction.BELOW, Direction.ABOVE):
            return self
        return list(Direction)[(self.delta_index + 5) % 6]

    @property
    def delta_index(self) -> int:
        return list(Direction).index(self)

    @property
    def is_right(self) -> bool:
        return self in (Direction.RIGHT, Direction.UP_RIGHT, Direction.DOWN_RIGHT)

    @property
    def is_left(self) -> bool:
        return self in (Direction.LEFT, Direction.UP_LEFT, Direction.DOWN_LEFT)

class Error(Exception):
    def __init__(self, message: str ="An error occurred"):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"err {self.message}."

class InvalidMoveError(Error):
    def __init__(self, message: str ="Invalid move attempted"):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"invalidmove {self.message}."