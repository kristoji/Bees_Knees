# Batched inference for OracleGNN + MCTS
# --------------------------------------------------------------
# This file provides a drop-in batched version of your OracleGNN
# and an updated MCTS implementation that collects multiple leaf
# states per iteration, evaluates them in a single forward pass,
# and then expands/backprops them. It keeps your existing logic:
#   - value v is predicted for the leaf state and inverted (1 - v)
#   - policy π(m) ∝ exp(1 - V(next_state_m)), with V predicted via NN
#   - terminal states are handled analytically (no NN call)
#
# Usage (example):
#   oracle = OracleGNN()
#   oracle.load("path/to/weights.pth")
#   mcts = MCTS_BATCH(oracle=oracle, exploration_weight=10, num_rollouts=1000, batch_size=64)
#   mcts.run_simulation_from(board)
#   move_str = mcts.action_selection(training=False)
#
# NOTE: This patch avoids per-state predict() calls in favor of batched
# forward passes. It does not change the public API of OracleGNN, except
# for adding new helper batch methods. MCTS_BATCH transparently uses them.

from __future__ import annotations
import math
from time import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Batch

# Your project imports (adjust paths if needed)
from ai.brains import Brain
from ai.oracle import Oracle
from ai.mcts import Node_mcts
from engine.board import Board
from engine.enums import GameState, PlayerColor, Error
from engine.game import Move

# Graph conversion helpers from your codebase
from gpt import board_to_simple_honored_graph

# Your improved GNN
from ai.improved_honored_GNN import GraphClassifierImproved
from ai.graph_honored_network import GraphClassifier
from ai.oracleGNN_batch import OracleGNN_BATCH


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

    def __init__(self, oracle: OracleGNN_BATCH, exploration_weight: int = 10, num_rollouts: int = 1000,
                 time_limit: float = float("inf"), batch_size: int = 64, debug: bool = False) -> None:
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

    # -------------------------
    #  Public API
    # -------------------------
    def calculate_best_move(self, board: Board, restriction: str = "depth", value: int = 0) -> str:
        if restriction == "depth":
            self.time_limit = float("inf")
            if value > 0:
                self.num_rollouts = int(value)
        elif restriction == "time":
            self.time_limit = float(value)
            self.start_time = time()
        else:
            raise Error("Invalid restriction for MCTS")

        self.run_simulation_from(board, debug=False)
        return self.action_selection(training=False)

    def action_selection(self, training: bool = False, debug: bool = False) -> str:
        node = self.choose(training=training, debug=debug)
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
        self.init_board = board
        last_move = board.moves[-1] if board.moves else None
        self.init_node = Node_mcts(last_move)
        self.init_node.set_state(board.state, board.current_player_color, board.zobrist_key, 0)

        # Don't expand root here; it will be handled by the first batch flush if needed
        terminal_states = 0
        completed_rollouts = 0

        pending: List[dict] = []               # collected leaves info
        leaf_datas: List[Data] = []            # Data for leaf states
        next_datas: List[Data] = []            # Data for next states across all leaves (for π)

        def flush_batch():
            nonlocal pending, leaf_datas, next_datas, completed_rollouts
            if not pending:
                return

            # 1) Value for leaves
            leaf_vals = self.oracle.predict_values_batch_from_data(leaf_datas, use_sigmoid=True)
            leaf_vals = [1 - float(v) for v in leaf_vals]  # inverted net

            # 2) Values for next states used in π
            if next_datas:
                next_vals = self.oracle.predict_values_batch_from_data(next_datas, use_sigmoid=True)
            else:
                next_vals = []

            # 3) Expand and backpropagate each leaf
            for info, v in zip(pending, leaf_vals):
                path_moves: List[Move] = info['path_moves']
                node: Node_mcts = info['node']
                move_specs: List[Tuple[str, Optional[int], Optional[float], Move]] = info['move_specs']

                # Build π from move_specs
                per_move_scores: List[float] = []
                per_moves: List[Move] = []
                for spec_type, idx, termV, mv in move_specs:
                    if spec_type == 'terminal':
                        V = float(termV)
                        score = 1 - V
                    else:  # 'predict'
                        V = float(next_vals[idx])
                        V = 1 - V  # inverted net for next state
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
                self._backpropagate(node, reward)

                self.init_board.undo(len(path_moves))
                completed_rollouts += 1

            # reset buffers
            pending = []
            leaf_datas = []
            next_datas = []

        # Main loop (time-limited or rollout-limited)
        if self.time_limit < float("inf"):
            start = time()
            while time() - start < self.time_limit - self.epsilon:
                # Collect a leaf
                collected = self._collect_leaf_for_batch()
                if collected is None:
                    # Tree is fully terminal
                    break
                if collected.get('terminal_immediate', False):
                    # Immediately backpropagate terminal leaves (no NN call)
                    node: Node_mcts = collected['node']
                    reward = 1 - node.reward()
                    self._backpropagate(node, reward)
                    completed_rollouts += 1
                    terminal_states += 1
                else:
                    pending.append(collected)
                    leaf_datas.append(collected['leaf_data'])
                    for spec in collected['move_specs']:
                        if spec[0] == 'predict':
                            # spec: ('predict', next_index_placeholder, None, move)
                            new_idx = len(next_datas)
                            next_datas.append(spec[1])  # here spec[1] temporarily stores Data
                            # replace in place with ('predict', flattened_index, None, move)
                            spec_list = list(spec)
                            spec_list[1] = new_idx
                            spec = tuple(spec_list)
                        # Update back into move_specs
                    # we need to reassign because tuples are immutable; rebuild move_specs
                    rebuilt_specs = []
                    for spec in collected['move_specs']:
                        if spec[0] == 'predict' and isinstance(spec[1], Data):
                            new_idx = len(next_datas)
                            next_datas.append(spec[1])
                            rebuilt_specs.append(('predict', new_idx, None, spec[3]))
                        else:
                            rebuilt_specs.append(spec)
                    collected['move_specs'] = rebuilt_specs

                    if len(pending) >= self.batch_size:
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
                    pending.append(collected)
                    leaf_datas.append(collected['leaf_data'])
                    rebuilt_specs = []
                    for spec in collected['move_specs']:
                        if spec[0] == 'predict':
                            new_idx = len(next_datas)
                            next_datas.append(spec[1])  # Data
                            rebuilt_specs.append(('predict', new_idx, None, spec[3]))
                        else:
                            rebuilt_specs.append(spec)
                    collected['move_specs'] = rebuilt_specs

                    if len(pending) >= self.batch_size:
                        flush_batch()

            flush_batch()

        if debug:
            print(f"\nTerminal states {terminal_states}/{completed_rollouts} rollouts")

        # reflect actual count for downstream assertions
        self.num_rollouts = completed_rollouts

    # -------------------------
    #  Selection helpers
    # -------------------------
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
        while leaf is not None:
            leaf.N += 1
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
