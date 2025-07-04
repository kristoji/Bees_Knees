from collections import defaultdict
import math
from typing import Final, List, Optional, Set, Tuple
from random import choice, uniform
from time import sleep
from abc import ABC, abstractmethod
from engine.board import Board
from engine.enums import GameState, PlayerColor, Error
from copy import deepcopy
from engine.game import Move
from time import time
from ai.oracle import Oracle
from ai.mcts import Node_mcts
# from ai.network import NeuralNetwork

from tqdm import tqdm

INF = float("inf")

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

def print_log(msg: str) -> None:
    return 
    with open("test/log.txt", "a") as f:
        f.write(msg + "\n")
        f.flush()

def print_log2(msg: str) -> None:
    with open("test/log.txt", "a") as f:
        f.write(msg + "\n")
        f.flush()

class MCTS(Brain):
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, oracle: Oracle, exploration_weight: int = 1, num_rollouts: int = 1000, debug: bool = False) -> None:
        super().__init__()
        self.init_node = None
        self.init_board = None  # the board to be used for the next rollout
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.debug = debug
        self.oracle = oracle

    def choose(self, training:bool) -> Node_mcts:
        "Choose the best successor of node. (Choose a move in the game)"
        if training:
            u = uniform(0, 1)
            for child in self.init_node.children:
                if u < child.Q:
                    return child
                u -= child.Q
            raise Error("No child selected in MCTS.choose()")
        else:
            def score(n: Node_mcts) -> float:
                "Score function for the node. Higher is better."
                return n.N 
            return max(self.init_node.children, key=score)
        
        
    def do_rollout(self) -> None:
        "Make the tree one layer better. (Train for one iteration.)"
        leaf = self._select_and_expand()
        print_log("Selection done")

        
        reward = 1 - leaf.reward()      # perché ci interessa vedere i valori di Q e W dal padre (che ha colore opposto)
        print_log("Simulation done")

        
        self._backpropagate(leaf, reward)
        print_log("Backpropagation done")
        

    def _select_and_expand(self) -> Node_mcts:
        "Find an unexplored descendent of `node`"
        curr_node = self.init_node
    
        curr_board = self.init_board
        
        number_of_moves = 0
    
        while True:

            print_log(f"Current node: {curr_node}")

            if curr_node.is_unexplored or curr_node.is_terminal:
                break
            
            curr_node = self._uct_select(curr_node)  # descend a layer deeper
            
            # Aggiorno la curr_board giocando la mossa scelta
            curr_board.safe_play(curr_node.move)
            number_of_moves += 1
        
        print_log(f"Leaf node: {curr_node}")

        # expand di curr_node
        if curr_node.is_unexplored:
            print_log("Nodo unexplored -> expand")
            v, pi = self.oracle.predict(curr_board)
            curr_node.expand(curr_board, v, pi)

        if number_of_moves:
            curr_board.undo(number_of_moves)

        return curr_node

    def _simulate(self, node: Node_mcts) -> float:
        "Returns the reward for a random simulation (to completion) of `node`"
        # non possiamo scendere in fondo come nella repo: la simulazione si ferma
        # return 1 - node.reward()
        return node.reward()

    def _backpropagate(self, leaf: Node_mcts, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        while leaf is not None:
            leaf.N += 1
            leaf.W += reward
            leaf.Q = leaf.W / leaf.N
            reward = 1 - reward
            print_log(f"Backpropagation: {leaf} -> N = {leaf.N}, Q = {leaf.Q}")
            leaf = leaf.parent
            
    def _uct_select(self, node: Node_mcts) -> Node_mcts:
        "Select a child of node, balancing exploration & exploitation"

        sqrt_N_vertex = math.sqrt(node.N)
        def uct_Norels(n:Node_mcts) -> float:
            # return 10*n.Q + n.P * sqrt_N_vertex / (1 + n.N)
            return n.Q + 100*n.P * sqrt_N_vertex / (1 + n.N)

        return max(node.children, key=uct_Norels)

    def run_simulation_from(self, board: Board) -> None:
        self.init_board = board
        last_move = board.moves[-1] if board.moves else None
        self.init_node = Node_mcts(last_move)

        self.init_node.set_state(board.state, board.current_player_color, board.zobrist_key, 0)
        # self.init_node.N = 1    # TODO: check


        if self.debug:
            for i in tqdm(range(self.num_rollouts), desc="Rollouts", unit="rollout"):
                print_log("----------------------------------------------")
                print_log(f"Rollout {i+1} / 50")

                self.do_rollout()

                print_log(f"N = {self.init_node.N}\n Q = {self.init_node.Q}")
                print_log(f"Rollout {i+1} / 50 done")
        else:

            for _ in range(self.num_rollouts):
                self.do_rollout()

    def action_selection(self, training=False) -> str:
        # , board: Board
        # assert board == self.init_board
        node = self.choose(training)
        return self.init_board.stringify_move(node.move)
        

    def calculate_best_move(self, board: Board, restriction: str, value: int) -> str:
        self.run_simulation_from(board)
        return self.action_selection(board)

    def get_moves_probs(self) -> dict[Move, float]:
        #, board:Board
        # assert board == self.init_board
        
        moves_probabilities = {}
        for child in self.init_node.children:
            moves_probabilities[child.move] = child.N / self.num_rollouts

        return moves_probabilities
    
def visualize_mcts(root, max_depth=3):
    """
    Print the MCTS tree from `root` as ASCII art via print_log().
    """
    def _recurse(node, prefix="", is_last=True, depth=0):
        if depth > max_depth or node is None:
            return

        # Connector & label
        connector = "└── " if is_last else "├── "
        move_str = node.move if node.move is not None else "ROOT"
        label = f"{move_str} [N={node.N}, W={node.W:.2f}]"
        print_log2(f"{prefix}{connector}{label}")

        # Prepare prefix for children
        if depth < max_depth:
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = getattr(node, "children", [])
            for i, child in enumerate(children):
                _recurse(child,
                         prefix=child_prefix,
                         is_last=(i == len(children) - 1),
                         depth=depth + 1)

    # Kick off printing
    print_log2("MCTS Tree:")
    _recurse(root, prefix="", is_last=True, depth=0)