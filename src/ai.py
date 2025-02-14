from typing import Final, Optional, Set, Tuple
from random import choice
from time import sleep
from abc import ABC, abstractmethod
from board import Board
from enums import GameState, PlayerColor, Error
from copy import deepcopy
from game import Move
from time import time

INF = 10000

class Node:
    def __init__(self, move_str: str, board: Board, depth: int = 0, alpha: int = -INF, beta: int = INF) -> None:
        self.move_str = move_str
        self.board = board
        self.alpha = alpha
        self.beta = beta
        self.depth = depth

        self.best_move: str = ""
        self.value: int = -INF if self.board.current_player_color == PlayerColor.WHITE else INF
        self.moves: list[Move] = []
        self.move_index: int = 0

    def initialize_moves(self) -> None: 
        self.moves = list(self.board.get_valid_moves())

    def __str__(self) -> str:
        return "-" * self.depth + f"> Node(move={self.move_str}, value={self.value}, alpha={self.alpha}, beta={self.beta}, move_i={self.move_index}/{len(self.moves)})"

# class TranspositionTable:


class Brain(ABC):
    def __init__(self) -> None:
        self._cache: Optional[str] = None
        self._my_color: Optional[str] = None

    @abstractmethod
    def calculate_best_move(self, board: Board, restriction: str, value: int) -> str:
        pass

    def empty_cache(self) -> None:
        self._cache = None

class Random(Brain):
    def calculate_best_move(self, board: Board, restriction: str, value: int) -> str:
        if self._cache is None:
            moves = board.valid_moves.split(";")
            self._cache = choice(moves)
        sleep(0.5)
        return self._cache

class AlphaBetaPruner(Brain):

    def __init__(self) -> None:
        super().__init__()
        self._eval_cost: Final[int] = 20

    def calculate_best_move(self, board: Board, restriction: str, value: int) -> str:
        if not self._cache:
            if restriction == "depth":
                self._depth = value
                # print(f"[DBG] Calculating best move for {board.current_player_color} at depth {self._depth}")
                root = Node(None, board)
                self._ab_pruning(root, self._depth)
                self._cache = root.best_move
            
            elif restriction == "time":
                start = time()
                depth = 1
                while time() - start < value:
                    root = Node(None, board)
                    self._ab_pruning(root, depth)
                    depth += 1
                    self._cache = root.best_move
            else:
                raise Error("Invalid restriction")
        return self._cache
    
    def _ab_pruning(self, root: Node, max_depth: int) -> int:
        root.initialize_moves()
        stack = [root]

        while stack:
            current = stack[-1]
            # print(f"[DBG] {current}")

            if (current.board.state != GameState.IN_PROGRESS and current.board.state != GameState.NOT_STARTED) or current.depth == max_depth:
                # print(f"[DBG] Terminal node reached: {current.board.state}, depth={current.depth}/{max_depth}")
                current.value = self.board_evaluation(current.board)
                stack.pop()
                if stack:
                    parent = stack[-1]
                    if parent.board.current_player_color == PlayerColor.WHITE:
                        if current.value > parent.value:
                            # print(f"[DBG] Parent value updated: {parent.value} <- {current.value}, move={current.move_str}")
                            parent.value = current.value
                            parent.best_move = current.move_str  # Record the move that led to this result.
                        parent.alpha = max(parent.alpha, parent.value)
                    else:
                        if current.value < parent.value:
                            # print(f"[DBG] Parent value updated: {parent.value} <- {current.value}, move={current.move_str}")
                            parent.value = current.value
                            parent.best_move = current.move_str
                        parent.beta = min(parent.beta, parent.value)
                    if parent.alpha >= parent.beta:
                        parent.move_index = len(parent.moves)
                continue

            if current.move_index < len(current.moves):
                move: Move = current.moves[current.move_index]
                current.move_index += 1

                child_board = deepcopy(current.board)
                move_str: str = child_board.stringify_move(move)
                child_board.play(move_str)
                    
                child = Node(move_str, child_board, current.depth + 1, current.alpha, current.beta)
                if child.depth < max_depth:
                    child.initialize_moves()
                stack.append(child)
                # print("[DBG] Child node created:", child)
            else:
                stack.pop()
                if stack:
                    parent = stack[-1]
                    if parent.board.current_player_color == PlayerColor.WHITE:
                        if current.value > parent.value:
                            parent.value = current.value
                            parent.best_move = current.move_str
                        parent.alpha = max(parent.alpha, parent.value)
                    else:
                        if current.value < parent.value:
                            parent.value = current.value
                            parent.best_move = current.move_str
                        parent.beta = min(parent.beta, parent.value)
                    if parent.alpha >= parent.beta:
                        parent.move_index = len(parent.moves)
        return root.value

    def board_evaluation(self, board: Board) -> int:
        match board.state:
            case GameState.WHITE_WINS:
                return INF
            case GameState.BLACK_WINS:
                return -INF
            case GameState.DRAW:
                return 0
            case GameState.IN_PROGRESS:        
                score: int = len(board.get_valid_moves(PlayerColor.WHITE)) - len(board.get_valid_moves(PlayerColor.BLACK))
                score += self._eval_cost * (- board.count_queen_neighbors(PlayerColor.WHITE) + board.count_queen_neighbors(PlayerColor.BLACK))
                return score
