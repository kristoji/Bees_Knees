from collections import defaultdict
import math
from typing import Final, List, Optional, Set, Tuple
from random import choice
from time import sleep
from abc import ABC, abstractmethod
from engine.board import Board
from engine.enums import GameState, PlayerColor, Error
from copy import deepcopy
from engine.game import Move
from time import time

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

    def __init__(self, exploration_weight: int = 1, num_rollouts: int = 50, debug: bool = False) -> None:
        super().__init__()
        self.init_node = None
        self.init_board = None  # the board to be used for the next rollout
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.debug = debug

    def choose(self) -> Node_mcts:
        "Choose the best successor of node. (Choose a move in the game)"

        def score(n: Node_mcts) -> float:
            if n.N == 0:
                return float("-inf")  # avoid unseen moves
            return n.Q / n.N # average reward
                
        return max(self.init_node.children, key=score)
        # if node.is_terminal():
        #     raise RuntimeError(f"choose called on terminal node {node}")

        # if node.is_unexplored:
        #     return node.expand(self.init_board)

        #return max(self.children[node], key=score)

    def do_rollout(self) -> None:
        "Make the tree one layer better. (Train for one iteration.)"
        leaf = self._select_and_expand()
        print_log("Selection done")

        
        reward = self._simulate(leaf)
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
            
            unexplored: List[Node_mcts] = [node for node in curr_node.children if node.is_unexplored]
            
            if unexplored:
                # TODO: alcuni figli sono esplorati, altri no: che facciamo???
                curr_node = unexplored[0]
                curr_board.safe_play(curr_node.move)
                number_of_moves += 1
                break
            
            curr_node = self._uct_select(curr_node)  # descend a layer deeper
            
            # Aggiorno la curr_board giocando la mossa scelta
            try:
                curr_board.safe_play(curr_node.move)
            except IndexError:
                print()
                
                print(curr_node.move.origin)
                print(curr_board.stringify_move(curr_node.move))
                print(curr_board.valid_moves)
                print(curr_board)
                print("SBRUGNA")
                exit()
            number_of_moves += 1
        
        print_log(f"Leaf node: {curr_node}")

        # expand di curr_node
        if curr_node.is_unexplored:
            print_log("Nodo unexplored -> expand")

            # TODO

            # -------------------------------- FROM CURRENT BOARD TO CENTERED BOARD ---------------------------------------
            # origin = Training.get_wQ_pos()
            # if not origin:
            #   origin = (0,0)
            # pos_to_bug_centered = Training.center_pos(curr_board.pos_to_bug, origin)
            # matrice_in = Training.to_in_mat(pos_to_bug_centered , curr_board.current_player_color)

            # -------------------------- USING NN TO GET MOVE PROBABILITIES AND V VALUE ----------------------------------
            # output_rete = chiamo_la_rete()
            # matrice_out, v = output_rete

            # ------------------------- TRANSFORMING THE OUT MATRIX IN A DICT[MOVE, FLOAT] -------------------------------
            # dict_mov_prob = Training.get_dict_from_matrix(matrice_out, origin, curr_board)
            
            # --------------------------- SETTING THE V VALUE IN THE CURRENT NODE ----------------------------------------
            # curr_node.V = v

            # ------------ EXANDING CURRENT NODE PASSING THE PROBABILITIES TO SET P VALUE IN THE CHILDREN ----------------
            # curr_node.expand(curr_board, dict_move_prob)

            curr_node.expand(curr_board)

        if number_of_moves:
            curr_board.undo(number_of_moves)

        return curr_node

    def _simulate(self, node: Node_mcts) -> float:
        "Returns the reward for a random simulation (to completion) of `node`"
        # non possiamo scendere in fondo come nella repo: la simulazione si ferma
        # chiamando la rete neurale facendo solo una espansione
        
        # Perchè è invertito il reward? 
        # in board.py passiamo il turno prima di giocare la mossa
        # quindi se il W fa la mossa e vince, il turno è del nero

        return 1 - node.reward()

    def _backpropagate(self, leaf: Node_mcts, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        while leaf is not None:
            leaf.N += 1
            leaf.Q += reward
            reward = 1 - reward
            print_log(f"Backpropagation: {leaf} -> N = {leaf.N}, Q = {leaf.Q}")
            leaf = leaf.parent
            
    def _uct_select(self, node: Node_mcts) -> Node_mcts:
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        # [!] Evitabile
        assert all(not child.is_unexplored for child in node.children)

        log_N_vertex = math.log(node.N)

        # TODO: ogni tanto capita N=0 e si rompe la formula
        def uct(n: Node_mcts) -> float:
            "Formula di Norelli -> "
            "U(state, action) = c * P(s,a) * sqrt(Sum_on_b N(s, b)) / (1 + N(s,a))"
            
            "Upper confidence bound for trees"
            N = n.N+1
            return n.Q / N + self.exploration_weight * math.sqrt(
                log_N_vertex / N
            )

        return max(node.children, key=uct)
    
    def run_simulation_from(self, board: Board) -> None:
        self.init_board = board
        last_move = board.moves[-1] if board.moves else None
        self.init_node = Node_mcts(last_move)

        self.init_node.set_state(board.state, board.current_player_color, board.zobrist_key)
        self.init_node.N = 1    # TODO: check


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

    def action_selection(self) -> str:
        # , board: Board
        # assert board == self.init_board
        node = self.choose()
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
        label = f"{move_str} [N={node.N}, Q={node.Q:.2f}]"
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