import re
from typing import TypeGuard, Final, Optional
from engine.enums import Command, InvalidMoveError, Error
from engine.board import Board
from engine.game import Move
# from ai.training import Training
from ai.brains import Brain, Random, AlphaBetaPruner, MCTS
from copy import deepcopy

class Engine:
    VERSION: Final[str] = "0.0.0"

    def __init__(self) -> None:
        self.board: Optional[Board] = None
        # self.brain: Brain = Random()
        # self.brain: Brain = AlphaBetaPruner()
        # self.brain: Brain = MCTS()
        self.brain = None

    @property
    def is_active(self) -> TypeGuard[Board]:
        return self.board is not None

    def start(self) -> None:
        self.info()
        while True:
            print("ok")
            parts = input().strip().split()
            try:
                match parts:
                    case [Command.INFO]:
                        self.info()
                    case [Command.HELP, *args]:
                        self.help(args)
                    case [Command.OPTIONS]:
                        pass
                    case [Command.NEWGAME, *args]:
                        self.newgame(args)
                    case [Command.VALIDMOVES]:
                        self.validmoves()
                    case [Command.BESTMOVE, restriction, value]:
                        self.bestmove(restriction, value)
                    case [Command.PLAY, move]:
                        self.play(move)
                    case [Command.PLAY, part1, part2]:
                        self.play(f"{part1} {part2}")
                    case [Command.PASS]:
                        self.play(Move.PASS)
                    case [Command.UNDO, *args]:
                        self.undo(args)
                    case [Command.EXIT]:
                        print("ok")
                        break
                    case _:
                        raise Error("Invalid command. Try 'help' to see a list of valid commands and usage")
            
            except InvalidMoveError as e:
                print(e)
            except Error as e:
                print(e)

    def info(self) -> None:
        print(f"id BeesKneesEngine v{self.VERSION}")
        print("Mosquito;Ladybug;Pillbug")

    def help(self, args: list[str]) -> None:
        if args:
            if len(args) > 1:
                raise Error(f"Too many arguments for '{Command.HELP}'")
            else:
                cmd = args[0]
                if details := Command.help_details(cmd):
                    print(details)
                else:
                    raise Error(f"Unknown command '{cmd}'")
        else:
            print("Available commands:")
            print("\n".join(f"  {cmd}" for cmd in Command))
            print(f"Try '{Command.HELP} <command>' for details.")

    def newgame(self, args: list[str]) -> None:
        self.board = Board(" ".join(args))
        print(self.board)

    def validmoves(self) -> None:
        if self.is_active:
            print(self.board.valid_moves)
        else:
            raise Error("No game in progress. Try 'newgame' to start a new game.")

    def bestmove(self, restriction: str, value: str) -> None:
        if self.is_active:
            if restriction == "depth" and value.isdigit():
                print(self.brain.calculate_best_move(deepcopy(self.board), restriction, int(value)))
            
            elif restriction == "time" and re.fullmatch(r"\d+:\d+:\d+", value):
            #    seconds: int = sum(int(x) * 60 ** i for i, x in enumerate(reversed(value.split(":"))))
            #    print(self.brain.calculate_best_move(deepcopy(self.board), restriction, seconds))
                raise Error("Time restriction is not yet implemented.")
            else:
                raise Error("Input string was not in a correct format.")
        else:
            raise Error("No game in progress. Try 'newgame' to start a new game.")

    def play(self, move: str) -> None:
        if self.is_active:
            self.board.play(move)
            if self.brain:
                self.brain.empty_cache()
            print(self.board)
            
            # pi = self.brain.get_moves_probs(self.board)
            # print(pi)
            # d = self.board._pos_to_bug
            # Training.to_matrix(d, self.board.current_player_color)

            # wQ_pos = Training.get_wQ_pos(self.board)
            
            # d1 = Training.center_pos(wQ_pos, d)
            # Training.to_matrix(d1, self.board.current_player_color)
            # d1_r = Training.rotate_pos(d1)
            # Training.to_matrix(d1_r, self.board.current_player_color)
            # Training.get_matrices(self.board, pi)
        else:
            raise Error("No game in progress. Try 'newgame' to start a new game.")

    def undo(self, args: list[str]) -> None:
        if self.is_active:
            if len(args) <= 1:
                if args:
                    amount = args[0]
                    if amount.isdigit():
                        self.board.undo(int(amount))
                    else:
                        raise Error(f"Expected a positive integer but got '{amount}'")
                else:
                    self.board.undo()
                self.brain.empty_cache()
                print(self.board)
            else:
                raise Error(f"Too many arguments for '{Command.UNDO}'")
        else:
            raise Error("No game in progress. Try 'newgame' to start a new game.")


if __name__ == "__main__":
    Engine().start()
