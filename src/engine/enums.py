from enum import StrEnum, Flag, auto, Enum
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
                f"  {Command.BESTMOVE} time MaxTime\n  {Command.BESTMOVE} depth MaxDepth\n\n"
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

class BugName(Enum):
    wQ = 0
    wS1 = auto()
    wS2 = auto()
    wB1 = auto()
    wB2 = auto()
    wG1 = auto()
    wG2 = auto()
    wG3 = auto()
    wA1 = auto()
    wA2 = auto()
    wA3 = auto()
    wM = auto()
    wL = auto()
    wP = auto()
    bQ = auto()
    bS1 = auto()
    bS2 = auto()
    bB1 = auto()
    bB2 = auto()
    bG1 = auto()
    bG2 = auto()
    bG3 = auto()
    bA1 = auto()
    bA2 = auto()
    bA3 = auto()
    bM = auto()
    bL = auto()
    bP = auto()
    NumPieceNames = auto()


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
        # return [d for d in cls if d not in (cls.ABOVE, cls.BELOW)]
        return [Direction.RIGHT, Direction.UP_RIGHT, Direction.UP_LEFT, Direction.LEFT, Direction.DOWN_LEFT, Direction.DOWN_RIGHT]

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
        match self:
            case Direction.RIGHT:
                return Direction.LEFT
            case Direction.UP_RIGHT:
                return Direction.DOWN_LEFT
            case Direction.UP_LEFT:
                return Direction.DOWN_RIGHT
            case Direction.LEFT:
                return Direction.RIGHT
            case Direction.DOWN_LEFT:
                return Direction.UP_RIGHT
            case Direction.DOWN_RIGHT:
                return Direction.UP_LEFT
            case Direction.BELOW:
                return Direction.ABOVE
            case Direction.ABOVE:
                return Direction.BELOW

    @property
    def left_of(self) -> "Direction":
        match self:
            case Direction.RIGHT:
                return Direction.UP_RIGHT
            case Direction.UP_RIGHT:
                return Direction.UP_LEFT
            case Direction.UP_LEFT:
                return Direction.LEFT
            case Direction.LEFT:
                return Direction.DOWN_LEFT
            case Direction.DOWN_LEFT:
                return Direction.DOWN_RIGHT
            case Direction.DOWN_RIGHT:
                return Direction.RIGHT
            case Direction.BELOW:
                return Direction.BELOW
            case Direction.ABOVE:
                return Direction.ABOVE

    @property
    def right_of(self) -> "Direction":
        match self:
            case Direction.RIGHT:
                return Direction.DOWN_RIGHT
            case Direction.UP_RIGHT:
                return Direction.RIGHT
            case Direction.UP_LEFT:
                return Direction.UP_RIGHT
            case Direction.LEFT:
                return Direction.UP_LEFT
            case Direction.DOWN_LEFT:
                return Direction.LEFT
            case Direction.DOWN_RIGHT:
                return Direction.DOWN_LEFT
            case Direction.BELOW:
                return Direction.BELOW
            case Direction.ABOVE:
                return Direction.ABOVE

    @property
    def delta_index(self) -> int:
        match self:
            case Direction.RIGHT:
                return 0
            case Direction.UP_RIGHT:
                return 1
            case Direction.UP_LEFT:
                return 2
            case Direction.LEFT:
                return 3
            case Direction.DOWN_LEFT:
                return 4
            case Direction.DOWN_RIGHT:
                return 5
            case Direction.BELOW:
                return 6
            case Direction.ABOVE:
                return 7
        raise ValueError(f"Invalid direction: {self}")

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
    
if __name__ == "__main__":
    b = "wQ"
    print(BugName[b].value)
    print(BugName.wQ.name)
    print(BugName.NumPieceNames.value)