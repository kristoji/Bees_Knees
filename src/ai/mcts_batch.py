from __future__ import annotations
import math
from time import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from ai.brains import Brain
from ai.node_mcts import Node_mcts
from engine.board import Board
from engine.enums import GameState, PlayerColor, Error
from engine.game import Move
from ai.oracleGNN import OracleGNN

from ai.log_utils import countit


# --------------------------------------------------------------
# MCTS with batched rollouts
# --------------------------------------------------------------
class MCTS_BATCH(Brain):
    """Monte Carlo tree searcher that performs batched neural calls.

    Differences vs your original class:
      - Collects multiple leaf nodes per outer loop, evaluates them in batch,
        then expands and backpropagates.
      - Supports both rollout-limited and time-limited modes.
    """

    def __init__(self, oracle: OracleGNN, exploration_weight: int = 10, num_rollouts: int = 1024,
                 time_limit: float = float("inf"), batch_size: int = 32, debug: bool = False) -> None:
        super().__init__()
        self.init_node: Optional[Node_mcts] = None
        self.init_board: Optional[Board] = None
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.oracle = oracle
        self.time_limit = time_limit
        self.batch_size = max(1, int(batch_size))
        self.epsilon = 0.05
        self.start_time = time()
        self.debug = debug
        self.hashmap: Dict[int, float] = {}  # For storing V values of states

    # -------------------------
    #  Public API
    # -------------------------
    def calculate_best_move(self, board: Board, restriction: str, value: int, debug:bool = False) -> str:
        if restriction == "depth":
            self.time_limit = float("inf")  # ignore time limit
            if debug:
                start = time()
            self.num_rollouts = value # set max rollouts
            self.run_simulation_from(board, debug=debug)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Time taken: {time() - start:.2f} seconds")
            return a 
        elif restriction == "time":
            self.time_limit = value # set time limit
            self.start_time = time() # set the start time
            self.run_simulation_from(board, debug=debug)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Rollouts done: {self.num_rollouts}")
            return a
        else:
            raise Error("Invalid restriction for MCTS")

    def action_selection(self, training: bool = False, debug: bool = False) -> str:
        node = self.choose(training=training, debug=debug)
        # set the init node as the children of the init node relative to the move selected
        self.init_node = node
        return self.init_board.stringify_move(node.move)

    def get_moves_probs(self) -> Dict[Move, float]:
        moves_probabilities: Dict[Move, float] = {}
        total = max(1, sum(child.N for child in self.init_node.children))
        for child in self.init_node.children:
            moves_probabilities[child.move] = child.N / total
        return moves_probabilities

    # -------------------------
    #  Core MCTS (batched)
    # -------------------------
    def run_simulation_from(self, board: Board, debug: bool = False) -> None:

        # self.hashmap.clear()  # Clear the hashmap for new simulation

        self.init_board = board
        last_move = board.moves[-1] if board.moves else None

        if not self.init_node:
            print("TREE DA CREARE")
            self.init_node = Node_mcts(last_move, board.state, board.current_player_color, board.zobrist_key)
        else: #we use the already buildt tree
            print("TREE GIA PRESENTE")
            for child in self.init_node.children:
                if child.hash == board.zobrist_key:
                    self.init_node = child
                    print(f"INITIAL NODE has already {len(self.init_node.children)} children, and N is {self.init_node.N} [run_simulation_from]")

                    self.init_node.reset()  # reset N/W/Q for the new simulation
                    break
            else:
                raise Error("No matching child found, create a new node (probably the move played is not the one suggested by action_selection/calculate_best_move)")
                # No matching child found, create a new node
                self.init_node = Node_mcts(last_move, board.state, board.current_player_color, board.zobrist_key, parent=self.init_node)
        

        # Don't expand root here; it will be handled by the first batch flush if needed
        terminal_states = 0
        completed_rollouts = 0
        flush_counter = 0

        pending: List[dict] = []               # collected leaves info
        leaf_datas: List[Data] = []            # Data for leaf states
        next_datas: List[Data] = []            # Data for next states across all leaves (for π)

        def flush_batch():
            nonlocal pending, leaf_datas, next_datas, completed_rollouts, flush_counter
            #print(f"Compl rollout {completed_rollouts} - FLUSHHHH  (batch of size {len(pending)})", flush=True)
            if not pending:
                return

            # 1) Value for leaves
            leaf_vals = self.oracle.predict_values_batch_from_data(leaf_datas, use_sigmoid=True)

            # 2) Values for next states used in π
            if next_datas:
                next_vals = self.oracle.predict_values_batch_from_data(next_datas, use_sigmoid=True)
            else:
                next_vals = []

            # 3) Expand and backpropagate each leaf
            for info, v in zip(pending, leaf_vals):
                path_moves: List[Move] = info['path_moves']
                node: Node_mcts = info['node']
                move_specs: List[Tuple[str, Optional[int], Optional[float], Move]] = info['move_specs'] # ('spec_type', index, terminal_value, move)

                # Build π from move_specs
                per_move_scores: List[float] = []
                per_moves: List[Move] = []
                for spec_type, idx, termV, mv in move_specs:
                    if spec_type == 'terminal':
                        V = float(termV)
                        score = 1 - V
                    else:  # 'predict'
                        V = float(next_vals[idx])
                        score = 1 - V
                    per_move_scores.append(score)
                    per_moves.append(mv)                    

                if per_move_scores:
                    arr = np.array(per_move_scores, dtype=np.float32)
                    arr = np.exp(arr - np.max(arr))
                    arr /= np.sum(arr)
                    pi = {m: float(p) for m, p in zip(per_moves, arr.tolist())}
                else:
                    pi = {}

                # Reconstruct board at the leaf, expand, compute reward, backprop
                for m in path_moves:
                    self.init_board.safe_play(m)

                # For draw terminal that we flagged earlier, node.V was set; but here node is unexplored
                node.expand(self.init_board, v, pi)
                reward = 1 - node.reward()
                # self._backpropagate(node, reward)
                self._backpropagate_non_N(node, reward)  # Update N but not W/Q

                if len(path_moves) > 0:
                    self.init_board.undo(len(path_moves))
                completed_rollouts += 1

            flush_counter += 1

            # reset buffers
            pending = []
            leaf_datas = []
            next_datas = []

        # Main loop (time-limited or rollout-limited)
        if self.time_limit < float("inf"):
            while time() - self.start_time < self.time_limit - self.epsilon:
                # Collect a leaf
                collected = self._collect_leaf_for_batch()
                if collected is None:
                    break
                if collected.get('terminal_immediate', False):
                    node: Node_mcts = collected['node']
                    reward = 1 - node.reward()
                    self._backpropagate(node, reward)
                    completed_rollouts += 1
                    terminal_states += 1
                else:
                    # update N
                    self._backpropagate_N(collected['node'])  # N is updated but not W/Q
                    
                    pending.append(collected)
                    leaf_datas.append(collected['leaf_data'])
                    rebuilt_specs = []
                    for spec in collected['move_specs']: # per ogni tupla
                        if spec[0] == 'predict':
                            # sostituisci Data con l'indice nella lista next_datas
                            new_idx = len(next_datas)
                            next_datas.append(spec[1])  # Data
                            rebuilt_specs.append(('predict', new_idx, None, spec[3]))
                        else:
                            rebuilt_specs.append(spec)
                    collected['move_specs'] = rebuilt_specs

                    if len(pending) >= (1 if flush_counter==0 else self.batch_size//4 if flush_counter==1 else self.batch_size):
                        flush_batch()

            # Flush leftovers
            flush_batch()
        else:
            # rollout-limited
            target = int(self.num_rollouts)
            while completed_rollouts < target:
                collected = self._collect_leaf_for_batch()
                if collected is None:
                    break
                if collected.get('terminal_immediate', False):
                    node: Node_mcts = collected['node']
                    reward = 1 - node.reward()
                    self._backpropagate(node, reward)
                    completed_rollouts += 1
                    terminal_states += 1
                else:
                    # update N
                    self._backpropagate_N(collected['node'])  # N is updated but not W/Q
                    
                    pending.append(collected)
                    leaf_datas.append(collected['leaf_data'])
                    rebuilt_specs = []
                    for spec in collected['move_specs']: # per ogni tupla
                        if spec[0] == 'predict':
                            # sostituisci Data con l'indice nella lista next_datas
                            new_idx = len(next_datas)
                            next_datas.append(spec[1])  # Data
                            rebuilt_specs.append(('predict', new_idx, None, spec[3]))
                        else:
                            rebuilt_specs.append(spec)
                    collected['move_specs'] = rebuilt_specs

                    if len(pending) >= (1 if flush_counter==0 else self.batch_size//4 if flush_counter==1 else self.batch_size):
                        flush_batch()

            flush_batch()

        if debug:
            print(f"\nTerminal states {terminal_states}/{completed_rollouts} rollouts")

        # reflect actual count for downstream assertions
        self.num_rollouts = completed_rollouts

    # -------------------------
    #  Selection helpers
    # -------------------------
    @countit
    def _collect_leaf_for_batch(self) -> Optional[dict]:
        """Walk down the tree using UCT until an unexplored or terminal node.
        Return a dict describing what to evaluate/expand, without expanding.
        Structure:
            {
              'node': Node_mcts,
              'path_moves': List[Move],
              'leaf_data': Data,                  # ONLY for unexplored non-terminal
              'move_specs': List[Tuple],          # per move: ('terminal', None, V_term, move) or ('predict', Data, None, move)
              'terminal_immediate': bool          # True when node is terminal (no expand, just backprop)
            }
        """
        curr_node = self.init_node
        curr_board = self.init_board
        path_moves: List[Move] = []

        # Descend
        while True:
            if curr_node.is_unexplored or curr_node.is_terminal:
                break
            curr_node = self._uct_select(curr_node)
            curr_board.safe_play(curr_node.move)
            path_moves.append(curr_node.move)
        
        # Handle terminal
        if curr_node.is_terminal:
            if curr_node.V == -1:  # draw needs heuristic once
                if curr_board.state == GameState.DRAW:
                    v = 0.5
                else:
                    v = 1.0 if (
                        (curr_board.state == GameState.WHITE_WINS and curr_board.current_player_color == PlayerColor.WHITE) or
                        (curr_board.state == GameState.BLACK_WINS and curr_board.current_player_color == PlayerColor.BLACK)
                    ) else 0.0
                curr_node.V = v
            # Undo path before returning
            if path_moves:
                curr_board.undo(len(path_moves))
            return {
                'node': curr_node,
                'path_moves': path_moves,
                'terminal_immediate': True,
            }

        elif curr_node.is_unexplored and curr_node.is_expanded:  # (but not explored)
            print("SI GODE, L'AVEVAMO GIA")
            # Explore the curr_node
            curr_node.reset_children()
            # Undo path before returning
            if path_moves:
                curr_board.undo(len(path_moves))
            return {
                'node': curr_node,
                'path_moves': path_moves,
                'terminal_immediate': True,
            }
        
        else:
            # Unexplored case – collect Data for leaf and for each child next state
            # Data for leaf
            d_leaf = self.oracle._data_from_board(curr_board)
            if d_leaf is None:
                d_leaf = Data(x=torch.zeros((1, 13), dtype=torch.float32),
                            edge_index=torch.zeros((2, 0), dtype=torch.long),
                            batch=torch.zeros(1, dtype=torch.long))

            move_specs: List[Tuple[str, Optional[Data], Optional[float], Move]] = []
            valid_moves = list(curr_board.get_valid_moves())
            for m in valid_moves:
                curr_board.safe_play(m)
                if curr_board.state != GameState.IN_PROGRESS:
                    if curr_board.state == GameState.DRAW:
                        V = 0.5
                    else:
                        V = 1.0 if (
                            (curr_board.state == GameState.WHITE_WINS and curr_board.current_player_color == PlayerColor.WHITE) or
                            (curr_board.state == GameState.BLACK_WINS and curr_board.current_player_color == PlayerColor.BLACK)
                        ) else 0.0
                    move_specs.append(('terminal', None, float(V), m))
                else:
                    d_next = self.oracle._data_from_board(curr_board)
                    if d_next is None:
                        move_specs.append(('terminal', None, 0.5, m))
                    else:
                        move_specs.append(('predict', d_next, None, m))
                curr_board.undo()

            # Undo path before returning
            if path_moves:
                curr_board.undo(len(path_moves))

            return {
                'node': curr_node,
                'path_moves': path_moves,
                'leaf_data': d_leaf,
                'move_specs': move_specs,
                'terminal_immediate': False,
            }

    def _backpropagate(self, leaf: Node_mcts, reward: float) -> None:
        leaf.is_unexplored = False
        while leaf is not None:
            leaf.N += 1
            leaf.W += reward
            leaf.Q = leaf.W / leaf.N
            reward = 1 - reward
            leaf = leaf.parent

    def _backpropagate_N(self, leaf: Node_mcts) -> None:
        """Backpropagate reward to the leaf and its ancestors, updating N."""
        while leaf is not None:
            leaf.N += 1
            leaf = leaf.parent

    def _backpropagate_non_N(self, leaf: Node_mcts, reward: float) -> None:
        """Backpropagate reward to the leaf and its ancestors, but do not update N."""
        leaf.is_unexplored = False
        while leaf is not None:
            leaf.W += reward
            leaf.Q = leaf.W / leaf.N
            reward = 1 - reward
            leaf = leaf.parent

    def _uct_select(self, node: Node_mcts) -> Node_mcts:
        sqrt_N_vertex = math.sqrt(max(1, node.N))
        def uct(n: Node_mcts) -> float:
            return n.Q + self.exploration_weight * n.P * sqrt_N_vertex / (1 + n.N)
        return max(node.children, key=uct)

    def choose(self, training: bool, debug: bool = False) -> Node_mcts:
        if debug:
            print("\n\nChildren of root node (sorted by visits):\n")
            for child in sorted(self.init_node.children, key=lambda x: x.N, reverse=True):
                print(f"Move: {self.init_board.stringify_move(child.move)} -> N = {child.N}, W = {child.W}, Q = {child.Q}, P = {child.P}, V = {child.V}")
        if training:
            total = max(1, sum(child.N for child in self.init_node.children))
            rnd = np.random.rand()
            acc = 0.0
            for child in self.init_node.children:
                acc += child.N / total
                if rnd <= acc:
                    return child
            return self.init_node.children[-1]
        else:
            return max(self.init_node.children, key=lambda n: n.N)
