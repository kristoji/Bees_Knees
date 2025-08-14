import math
from typing import Final, Optional
from random import choice, uniform
from abc import ABC, abstractmethod
from engine.board import Board
from engine.enums import GameState, PlayerColor, Error
from copy import deepcopy
from engine.game import Move
from time import time
from ai.oracle import Oracle
from ai.node_mcts import Node_mcts

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
        moves = board.valid_moves.split(";")
        return choice(moves)

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
    # print(msg)
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

    def __init__(self, oracle: Oracle, exploration_weight: int = 10, num_rollouts: int = 1500, time_limit: float = float("inf"), debug: bool = False) -> None:
        super().__init__()
        self.init_node = None
        self.init_board = None  # the board to be used for the next rollout
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.oracle = oracle
        self.time_limit = time_limit
        self.epsilon = 0.05  # small value to avoid time limit issues
        self.start_time = time()
        self.debug = debug
        self.counter = 0  # used to check if the number of visits to the node is equal to the sum of visits to its children

    def calculate_best_move(self, board: Board, restriction: str, value: int, debug:bool = False) -> str:
        if restriction == "depth":
            self.time_limit = float("inf")  # ignore time limit
            if debug:
                start = time()
            self.num_rollouts = value # set max rollouts
            self.run_simulation_from(board, debug=False)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Time taken: {time() - start:.2f} seconds", flush=True)
            return a 
        elif restriction == "time":
            self.time_limit = value # set time limit
            self.start_time = time() # set the start time
            self.run_simulation_from(board, debug=False)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Rollouts done: {self.num_rollouts}")
            return a
        else:
            raise Error("Invalid restriction for MCTS")

    def choose(self, training:bool, debug:bool = False) -> Node_mcts:
        "Choose the best successor of node. (Choose a move in the game)"
        if debug:
            print("\n\nChildren of root node (sorted by visits):\n", flush=True)
            for child in sorted(self.init_node.children, key=lambda x: x.N, reverse=True):
                print(f"Move: {self.init_board.stringify_move(child.move)} -> N = {child.N}, W = {child.W}, Q = {child.Q}, P = {child.P}, V = {child.V}", flush=True)
        if training:
            # assert self.init_node.N == self.num_rollouts, "The number of rollouts must be equal to the number of visits to the root node."
            # print( sum([child.N for child in self.init_node.children]))
            # print(self.init_node.N)
            assert sum([child.N for child in self.init_node.children]) == self.num_rollouts-1
            rnd = uniform(0, 1)
            for child in self.init_node.children:
                if rnd <= (portion := child.N / (self.num_rollouts-1)):
                    return child
                rnd -= portion
            raise Error(str(rnd) + " - No child selected in MCTS.choose()")
        else:
            def score(n: Node_mcts) -> float:
                "Score function for the node. Higher is better."
                return n.N 
            # TODO: shuffle children with same best score
            return max(self.init_node.children, key=score)
        
        
    def do_rollout(self) -> None:
        "Make the tree one layer better. (Train for one iteration.)"
        leaf = self._select_and_expand()
        print_log("Selection done")

        
        reward = 1 - leaf.reward()      # perché ci interessa vedere i valori di Q e W dal padre (che ha colore opposto)
        print_log("Simulation done")

        
        self._backpropagate(leaf, reward)
        print_log("Backpropagation done")

        return leaf.is_terminal
        

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

        if curr_node.is_terminal and curr_node.V == -1:
            # compute the heuristic (only once) IF DRAW because, when calling the reward function, we want to avoid the DRAW if winning
            v = self.oracle.compute_heuristic(curr_board)
            curr_node.V = v

        # expand di curr_node (non può essere terminale)
        elif curr_node.is_unexplored:
            print_log("Nodo unexplored -> expand")
            v, pi = self.oracle.predict(curr_board)
            curr_node.expand(curr_board, v, pi)

        # TODO?
        # mettere v terminale -+inf. attento al draw
        # togliere 1- reward e mettere -reward

        if number_of_moves:
            curr_board.undo(number_of_moves)

        return curr_node

    def _backpropagate(self, leaf: Node_mcts, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        leaf.is_unexplored = False
        while leaf is not None:
            leaf.N += 1
            leaf.W += reward
            leaf.Q = leaf.W / leaf.N
            reward = 1 - reward # TODO: invece di tenere V in [0,1], tenerlo in [-inf, inf] e fare reward = -reward
            print_log(f"Backpropagation: {leaf} -> N = {leaf.N}, Q = {leaf.Q}")
            leaf = leaf.parent
            
    def _uct_select(self, node: Node_mcts, verbose=False) -> Node_mcts:
        "Select a child of node, balancing exploration & exploitation"

        if verbose:
            print(f"Father node N: {node.N}, sum children N: {sum(child.N for child in node.children)}", flush=True)
        
        s = sum(child.N for child in node.children)
        assert (node.N -1 == s), "The number of visits to the node must be equal to the sum of visits to its children."
        sqrt_N_vertex = math.sqrt(node.N-1)
        def uct_Norels(n:Node_mcts) -> float:
            return n.Q + self.exploration_weight * n.P * sqrt_N_vertex / (1 + n.N) #----> FIXED EXPL WEIGHT
            #return n.Q + (1 + (time() - self.start_time) / self.time_limit *(self.exploration_weight - 1)) * n.P * sqrt_N_vertex / (1 + n.N) # -----> LINEAR EXPL WEIGHT DURING TURN (NO SENSE)

        return max(node.children, key=uct_Norels)

    def run_simulation_from(self, board: Board, debug: bool=False) -> None:
        self.init_board = board
        last_move = board.moves[-1] if board.moves else None
        self.init_node = Node_mcts(last_move, board.state, board.current_player_color, board.zobrist_key)

        # self._select_and_expand() # expand the root node without updating N (in order to get root.N = sum(children.N))

        # if you can only do one move, return it
        if len(self.init_node.children) == 1:
            return

        terminal_states = 0

        if self.time_limit < float("inf"):
            self.num_rollouts = 0
            start_time = time()
            while time() - start_time < self.time_limit - self.epsilon:
                if self.do_rollout() and debug: # return true if new leaf is terminal
                    terminal_states += 1
                self.num_rollouts += 1
        else:
            if debug:
                for _ in tqdm(range(self.num_rollouts), desc="Rollouts", unit="rollout"):
                    if self.do_rollout() and debug: # return true if new leaf is terminal
                        terminal_states += 1
            else:
                for _ in range(self.num_rollouts):
                    if self.do_rollout() and debug: # return true if new leaf is terminal
                        terminal_states += 1
        
        if debug:
            print(f"\nTerminal states {terminal_states}/{self.num_rollouts} rollouts", flush=True)

    def action_selection(self, training=False, debug:bool = False) -> str:
        # , board: Board
        # assert board == self.init_board
        node = self.choose(training=training, debug=debug)
        return self.init_board.stringify_move(node.move)

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