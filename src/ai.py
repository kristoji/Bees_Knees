from typing import Optional
from random import choice
from time import sleep
from abc import ABC, abstractmethod
from board import Board

class Brain(ABC):
    def __init__(self) -> None:
        self._cache: Optional[str] = None

    @abstractmethod
    def calculate_best_move(self, board: Board) -> str:
        pass

    def empty_cache(self) -> None:
        self._cache = None

class Random(Brain):
    def calculate_best_move(self, board: Board) -> str:
        if self._cache is None:
            moves = board.valid_moves.split(";")
            self._cache = choice(moves)
        sleep(0.5)
        return self._cache

class AlphaBetaPruner(Brain):
    def calculate_best_move(self, board: Board) -> str:
        if self._cache is None:
            moves = board.valid_moves.split(";")
            self._cache = choice(moves)
        return self._cache
