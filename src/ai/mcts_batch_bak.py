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
# MCTS with batched rollouts and V-value caching
# --------------------------------------------------------------
class MCTS_BATCH(Brain):
    """Monte Carlo tree searcher that performs batched neural calls with V-value caching.

    Differences vs original:
      - Caches V values in hashmap to avoid redundant neural network calls
      - Hashmap resets at each run_simulation_from call
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
        self.hashmap: Dict[int, float] = {}  # Cache for V values indexed by zobrist hash

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
                print(f"Cache hits: {len(self.hashmap)} unique states cached")
            return a 
        elif restriction == "time":
            self.time_limit = value # set time limit
            self.start_time = time() # set the start time
            self.run_simulation_from(board, debug=debug)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Rollouts done: {self.num_rollouts}")
                print(f"Cache hits: {len(self.hashmap)} unique states cached")
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
    #  Core MCTS (batched with caching)
    # -------------------------
    def run_simulation_from(self, board: Board, debug: bool = False) -> None:

        # self.hashmap.clear()  # Clear the cache for new simulation

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
        cache_hits = 0
        cache_misses = 0

        pending: List[dict] = []               # collected leaves info
        leaf_batch_indices: List[int] = []     # indices in batch for leaves that need NN eval
        leaf_datas: List[Data] = []            # Data for leaf states that need NN eval
        next_batch_indices: List[int] = []     # indices in batch for next states that need NN eval  
        next_datas: List[Data] = []            # Data for next states that need NN eval

        def flush_batch():
            nonlocal pending, leaf_batch_indices, leaf_datas, next_batch_indices, next_datas
            nonlocal completed_rollouts, flush_counter, cache_hits, cache_misses
            
            if not pending:
                return

            # 1) Evaluate leaves that need NN (not cached)
            if leaf_datas:
                leaf_vals_from_nn = self.oracle.predict_values_batch_from_data(leaf_datas, use_sigmoid=True)
                cache_misses += len(leaf_datas)
            else:
                leaf_vals_from_nn = []

            # 2) Evaluate next states that need NN (not cached)  
            if next_datas:
                next_vals_from_nn = self.oracle.predict_values_batch_from_data(next_datas, use_sigmoid=True)
                cache_misses += len(next_datas)
            else:
                next_vals_from_nn = []

            # 3) Map NN results back to batch indices and update cache
            leaf_val_map = {}
            for batch_idx, nn_val, leaf_hash in zip(leaf_batch_indices, leaf_vals_from_nn, 
                                                     [p['leaf_hash'] for i, p in enumerate(pending) if i in leaf_batch_indices]):
                leaf_val_map[batch_idx] = nn_val
                self.hashmap[leaf_hash] = nn_val  # Cache the value

            next_val_map = {}
            for batch_idx, nn_val in zip(next_batch_indices, next_vals_from_nn):
                next_val_map[batch_idx] = nn_val

            # 4) Process each pending leaf
            for i, info in enumerate(pending):
                path_moves: List[Move] = info['path_moves']
                node: Node_mcts = info['node']
                move_specs: List[Tuple] = info['move_specs']
                
                # Get leaf value (from cache or newly computed)
                if 'leaf_v_cached' in info:
                    v = info['leaf_v_cached']
                    cache_hits += 1
                else:
                    v = leaf_val_map[i]

                # Build π from move_specs
                per_move_scores: List[float] = []
                per_moves: List[Move] = []
                for spec_type, idx, value, mv, state_hash in move_specs:
                    if spec_type == 'terminal':
                        V = float(value)
                        score = 1 - V
                    elif spec_type == 'cached':
                        V = float(value)  # value from cache
                        score = 1 - V
                        cache_hits += 1
                    else:  # 'predict'
                        V = float(next_val_map[idx])
                        score = 1 - V
                        # Cache this value for future use
                        if state_hash is not None:
                            self.hashmap[state_hash] = V
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

                node.expand(self.init_board, v, pi)
                reward = 1 - node.reward()
                self._backpropagate_non_N(node, reward)  # Update N but not W/Q

                if len(path_moves) > 0:
                    self.init_board.undo(len(path_moves))
                completed_rollouts += 1

            flush_counter += 1

            # reset buffers
            pending = []
            leaf_batch_indices = []
            leaf_datas = []
            next_batch_indices = []
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
                    
                    # Check if leaf value is cached
                    leaf_hash = collected['leaf_hash']
                    if leaf_hash in self.hashmap:
                        collected['leaf_v_cached'] = self.hashmap[leaf_hash]
                    else:
                        # Need to evaluate this leaf
                        leaf_batch_indices.append(len(pending))
                        leaf_datas.append(collected['leaf_data'])
                    
                    # Process move_specs and check cache for next states
                    rebuilt_specs = []
                    for spec in collected['move_specs']:
                        if spec[0] == 'predict':
                            state_hash = spec[4]  # state hash is at index 4
                            if state_hash in self.hashmap:
                                # Use cached value
                                rebuilt_specs.append(('cached', None, self.hashmap[state_hash], spec[3], state_hash))
                            else:
                                # Need to evaluate this state
                                new_idx = len(next_datas)
                                next_datas.append(spec[1])  # Data
                                next_batch_indices.append(new_idx)
                                rebuilt_specs.append(('predict', new_idx, None, spec[3], state_hash))
                        else:
                            rebuilt_specs.append(spec)
                    collected['move_specs'] = rebuilt_specs
                    
                    pending.append(collected)

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
                    
                    # Check if leaf value is cached
                    leaf_hash = collected['leaf_hash']
                    if leaf_hash in self.hashmap:
                        collected['leaf_v_cached'] = self.hashmap[leaf_hash]
                    else:
                        # Need to evaluate this leaf
                        leaf_batch_indices.append(len(pending))
                        leaf_datas.append(collected['leaf_data'])
                    
                    # Process move_specs and check cache for next states
                    rebuilt_specs = []
                    for spec in collected['move_specs']:
                        if spec[0] == 'predict':
                            state_hash = spec[4]  # state hash is at index 4
                            if state_hash in self.hashmap:
                                # Use cached value
                                rebuilt_specs.append(('cached', None, self.hashmap[state_hash], spec[3], state_hash))
                            else:
                                # Need to evaluate this state
                                new_idx = len(next_datas)
                                next_datas.append(spec[1])  # Data
                                next_batch_indices.append(new_idx)
                                rebuilt_specs.append(('predict', new_idx, None, spec[3], state_hash))
                        else:
                            rebuilt_specs.append(spec)
                    collected['move_specs'] = rebuilt_specs
                    
                    pending.append(collected)

                    if len(pending) >= (1 if flush_counter==0 else self.batch_size//4 if flush_counter==1 else self.batch_size):
                        flush_batch()

            flush_batch()

        if debug:
            print(f"\nTerminal states {terminal_states}/{completed_rollouts} rollouts")
            print(f"Cache statistics - Hits: {cache_hits}, Misses: {cache_misses}, Hit rate: {cache_hits/(cache_hits+cache_misses)*100:.1f}%")

        # reflect actual count for downstream assertions
        self.num_rollouts = completed_rollouts

    # -------------------------
    #  Selection helpers
    # -------------------------
    @countit
    def _collect_leaf_for_batch(self) -> Optional[dict]:
        """Walk down the tree using UCT until an unexplored or terminal node.
        Return a dict describing what to evaluate/expand, without expanding.
        Structure updated to include state hashes for caching:
            {
              'node': Node_mcts,
              'path_moves': List[Move],
              'leaf_hash': int,                   # zobrist hash of leaf state
              'leaf_data': Data,                  # ONLY for unexplored non-terminal
              'move_specs': List[Tuple],          # per move: ('terminal', None, V_term, move, None) or 
                                                  #           ('predict', Data, None, move, state_hash)
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
            # Get leaf hash
            leaf_hash = curr_board.zobrist_key
            
            # Data for leaf
            d_leaf = self.oracle._data_from_board(curr_board)
            if d_leaf is None:
                d_leaf = Data(x=torch.zeros((1, 13), dtype=torch.float32),
                            edge_index=torch.zeros((2, 0), dtype=torch.long),
                            batch=torch.zeros(1, dtype=torch.long))

            move_specs: List[Tuple[str, Optional[Data], Optional[float], Move, Optional[int]]] = []
            valid_moves = list(curr_board.get_valid_moves())
            for m in valid_moves:
                curr_board.safe_play(m)
                state_hash = curr_board.zobrist_key  # Get hash after move
                
                if curr_board.state != GameState.IN_PROGRESS:
                    if curr_board.state == GameState.DRAW:
                        V = 0.5
                    else:
                        V = 1.0 if (
                            (curr_board.state == GameState.WHITE_WINS and curr_board.current_player_color == PlayerColor.WHITE) or
                            (curr_board.state == GameState.BLACK_WINS and curr_board.current_player_color == PlayerColor.BLACK)
                        ) else 0.0
                    move_specs.append(('terminal', None, float(V), m, None))
                else:
                    d_next = self.oracle._data_from_board(curr_board)
                    if d_next is None:
                        move_specs.append(('terminal', None, 0.5, m, None))
                    else:
                        move_specs.append(('predict', d_next, None, m, state_hash))
                curr_board.undo()

            # Undo path before returning
            if path_moves:
                curr_board.undo(len(path_moves))

            return {
                'node': curr_node,
                'path_moves': path_moves,
                'leaf_hash': leaf_hash,
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