from __future__ import annotations
import math
from time import time
from typing import Dict, List, Tuple, Optional
from threading import Thread, Lock, Barrier
from queue import Queue
import concurrent.futures

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
# MCTS with batched rollouts, V-value caching, and parallel tree descent
# --------------------------------------------------------------
class MCTS_BATCH(Brain):
    """Monte Carlo tree searcher with parallel descent (virtual loss) like AlphaGo.
    
    Features:
      - Parallel tree descent with virtual loss
      - Batched neural network evaluation
      - V-value caching to avoid redundant NN calls
      - Thread-safe node expansion and backpropagation
    """

    def __init__(self, oracle: OracleGNN, exploration_weight: int = 10, num_rollouts: int = 1024,
                 time_limit: float = float("inf"), batch_size: int = 32, 
                 num_threads: int = 8, virtual_loss: float = 3.0, debug: bool = False) -> None:
        super().__init__()
        self.init_node: Optional[Node_mcts] = None
        self.init_board: Optional[Board] = None
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.oracle = oracle
        self.time_limit = time_limit
        self.batch_size = max(1, int(batch_size))
        self.num_threads = max(1, int(num_threads))
        self.virtual_loss = virtual_loss
        self.epsilon = 0.05
        self.start_time = time()
        self.debug = debug
        self.hashmap: Dict[int, float] = {}  # Cache for V values indexed by zobrist hash
        
        # Thread synchronization
        self.tree_lock = Lock()  # Global lock for tree modifications
        self.cache_lock = Lock()  # Lock for hashmap access
        self.stats_lock = Lock()  # Lock for statistics

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
                print(f"Cache size: {len(self.hashmap)} unique states")
            return a 
        elif restriction == "time":
            self.time_limit = value # set time limit
            self.start_time = time() # set the start time
            self.run_simulation_from(board, debug=debug)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Rollouts done: {self.num_rollouts}")
                print(f"Cache size: {len(self.hashmap)} unique states")
            return a
        else:
            raise Error("Invalid restriction for MCTS")

    def action_selection(self, training: bool = False, debug: bool = False) -> str:
        node = self.choose(training=training, debug=debug)
        self.init_node = node
        return self.init_board.stringify_move(node.move)

    def get_moves_probs(self) -> Dict[Move, float]:
        moves_probabilities: Dict[Move, float] = {}
        total = max(1, sum(child.N for child in self.init_node.children))
        for child in self.init_node.children:
            moves_probabilities[child.move] = child.N / total
        return moves_probabilities

    # -------------------------
    #  Core MCTS with Parallel Descent
    # -------------------------
    def run_simulation_from(self, board: Board, debug: bool = False) -> None:
        with self.cache_lock:
            self.hashmap.clear()  # Clear the cache for new simulation

        self.init_board = board
        last_move = board.moves[-1] if board.moves else None

        # Initialize or reuse tree
        if not self.init_node:
            print("TREE DA CREARE")
            self.init_node = Node_mcts(last_move, board.state, board.current_player_color, board.zobrist_key)
        else:
            print("TREE GIA PRESENTE")
            for child in self.init_node.children:
                if child.hash == board.zobrist_key:
                    self.init_node = child
                    print(f"INITIAL NODE has already {len(self.init_node.children)} children, and N is {self.init_node.N}")
                    self.init_node.reset()
                    break
            else:
                raise Error("No matching child found")

        # Statistics
        terminal_states = 0
        completed_rollouts = 0
        cache_hits = 0
        cache_misses = 0

        # Parallel descent worker
        def parallel_descend_worker(worker_id: int, collect_queue: Queue, stats: dict):
            """Worker thread that descends the tree and collects leaves."""
            local_completed = 0
            local_terminals = 0
            
            while True:
                # Check stopping condition
                if self.time_limit < float("inf"):
                    if time() - self.start_time >= self.time_limit - self.epsilon:
                        break
                else:
                    with self.stats_lock:
                        if completed_rollouts >= self.num_rollouts:
                            break
                
                # Collect a leaf with virtual loss
                collected = self._collect_leaf_with_virtual_loss(worker_id)
                if collected is None:
                    break
                    
                if collected.get('terminal_immediate', False):
                    # Handle terminal nodes immediately
                    node = collected['node']
                    virtual_losses = collected['virtual_losses']
                    reward = 1 - node.reward()
                    
                    # Backpropagate and remove virtual losses
                    with self.tree_lock:
                        self._backpropagate_remove_virtual(node, reward, virtual_losses)
                    
                    local_completed += 1
                    local_terminals += 1
                    with self.stats_lock:
                        completed_rollouts += 1
                else:
                    # Add to collection queue for batch processing
                    collect_queue.put(collected)
            
            # Store worker stats
            stats[worker_id] = {'completed': local_completed, 'terminals': local_terminals}

        # Batch processor
        def batch_processor(collect_queue: Queue, num_workers: int):
            nonlocal completed_rollouts, cache_hits, cache_misses, terminal_states
            
            pending = []
            leaf_batch_indices = []
            leaf_datas = []
            next_batch_indices = []
            next_datas = []
            workers_done = 0
            flush_counter = 0
            
            while workers_done < num_workers:
                # Collect items from queue
                timeout = 0.01 if pending else 0.1
                try:
                    item = collect_queue.get(timeout=timeout)
                    if item == "DONE":
                        workers_done += 1
                        continue
                        
                    # Process item similar to original but with virtual loss tracking
                    leaf_hash = item['leaf_hash']
                    
                    with self.cache_lock:
                        if leaf_hash in self.hashmap:
                            item['leaf_v_cached'] = self.hashmap[leaf_hash]
                        else:
                            leaf_batch_indices.append(len(pending))
                            leaf_datas.append(item['leaf_data'])
                    
                    # Process move_specs with cache check
                    rebuilt_specs = []
                    for spec in item['move_specs']:
                        if spec[0] == 'predict':
                            state_hash = spec[4]
                            with self.cache_lock:
                                if state_hash in self.hashmap:
                                    rebuilt_specs.append(('cached', None, self.hashmap[state_hash], spec[3], state_hash))
                                else:
                                    new_idx = len(next_datas)
                                    next_datas.append(spec[1])
                                    next_batch_indices.append(new_idx)
                                    rebuilt_specs.append(('predict', new_idx, None, spec[3], state_hash))
                        else:
                            rebuilt_specs.append(spec)
                    item['move_specs'] = rebuilt_specs
                    
                    pending.append(item)
                    
                except:
                    pass  # Timeout
                
                # Check if we should flush
                batch_threshold = 1 if flush_counter == 0 else self.batch_size // 4 if flush_counter == 1 else self.batch_size
                if len(pending) >= batch_threshold or (workers_done == num_workers and pending):
                    # Flush batch
                    if pending:
                        # Evaluate with NN
                        if leaf_datas:
                            leaf_vals_from_nn = self.oracle.predict_values_batch_from_data(leaf_datas, use_sigmoid=True)
                            cache_misses += len(leaf_datas)
                        else:
                            leaf_vals_from_nn = []
                        
                        if next_datas:
                            next_vals_from_nn = self.oracle.predict_values_batch_from_data(next_datas, use_sigmoid=True)
                            cache_misses += len(next_datas)
                        else:
                            next_vals_from_nn = []
                        
                        # Map results
                        leaf_val_map = {}
                        for batch_idx, nn_val, item in zip(leaf_batch_indices, leaf_vals_from_nn,
                                                           [p for i, p in enumerate(pending) if i in leaf_batch_indices]):
                            leaf_val_map[batch_idx] = nn_val
                            with self.cache_lock:
                                self.hashmap[item['leaf_hash']] = nn_val
                        
                        next_val_map = {idx: val for idx, val in zip(next_batch_indices, next_vals_from_nn)}
                        
                        # Process each pending item
                        for i, info in enumerate(pending):
                            # Get leaf value
                            if 'leaf_v_cached' in info:
                                v = info['leaf_v_cached']
                                cache_hits += 1
                            else:
                                v = leaf_val_map[i]
                            
                            # Build π
                            per_move_scores = []
                            per_moves = []
                            for spec_type, idx, value, mv, state_hash in info['move_specs']:
                                if spec_type == 'terminal':
                                    V = float(value)
                                    score = 1 - V
                                elif spec_type == 'cached':
                                    V = float(value)
                                    score = 1 - V
                                    cache_hits += 1
                                else:  # 'predict'
                                    V = float(next_val_map[idx])
                                    score = 1 - V
                                    with self.cache_lock:
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
                            
                            # Expand and backpropagate with virtual loss removal
                            node = info['node']
                            virtual_losses = info['virtual_losses']
                            path_moves = info['path_moves']
                            
                            # Reconstruct board state at leaf
                            board_copy = self.init_board.copy()
                            for m in path_moves:
                                board_copy.safe_play(m)
                            
                            with self.tree_lock:
                                node.expand(board_copy, v, pi)
                                reward = 1 - node.reward()
                                self._backpropagate_remove_virtual(node, reward, virtual_losses)
                            
                            with self.stats_lock:
                                completed_rollouts += 1
                        
                        # Reset buffers
                        pending = []
                        leaf_batch_indices = []
                        leaf_datas = []
                        next_batch_indices = []
                        next_datas = []
                        flush_counter += 1

        # Launch parallel workers
        actual_threads = min(self.num_threads, max(1, self.num_rollouts // 10))
        collect_queue = Queue(maxsize=self.batch_size * 2)
        worker_stats = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_threads + 1) as executor:
            # Start batch processor
            processor_future = executor.submit(batch_processor, collect_queue, actual_threads)
            
            # Start worker threads
            worker_futures = []
            for i in range(actual_threads):
                future = executor.submit(parallel_descend_worker, i, collect_queue, worker_stats)
                worker_futures.append(future)
            
            # Wait for workers to complete
            for future in worker_futures:
                future.result()
            
            # Signal batch processor to finish
            for _ in range(actual_threads):
                collect_queue.put("DONE")
            
            # Wait for processor
            processor_future.result()
        
        # Aggregate stats
        for stats in worker_stats.values():
            terminal_states += stats.get('terminals', 0)
        
        if debug:
            print(f"\nTerminal states {terminal_states}/{completed_rollouts} rollouts")
            print(f"Cache statistics - Hits: {cache_hits}, Misses: {cache_misses}")
            if cache_hits + cache_misses > 0:
                print(f"Hit rate: {cache_hits/(cache_hits+cache_misses)*100:.1f}%")
            print(f"Parallel threads used: {actual_threads}")
        
        self.num_rollouts = completed_rollouts

    # -------------------------
    #  Parallel Descent with Virtual Loss
    # -------------------------
    def _collect_leaf_with_virtual_loss(self, worker_id: int) -> Optional[dict]:
        """Descend tree with virtual loss applied to discourage collision."""
        curr_node = self.init_node
        path_moves: List[Move] = []
        virtual_losses: List[Node_mcts] = []
        
        # Create a thread-local board copy
        board_copy = self.init_board.copy()
        
        # Apply virtual losses during descent
        while True:
            with self.tree_lock:
                if curr_node.is_unexplored or curr_node.is_terminal:
                    break
                    
                # Apply virtual loss to current node
                curr_node.N += self.virtual_loss
                curr_node.W -= self.virtual_loss
                curr_node.Q = curr_node.W / curr_node.N if curr_node.N > 0 else 0
                virtual_losses.append(curr_node)
                
                # Select next node
                curr_node = self._uct_select(curr_node)
            
            # Apply move to local board copy
            board_copy.safe_play(curr_node.move)
            path_moves.append(curr_node.move)
        
        # Handle terminal node
        if curr_node.is_terminal:
            if curr_node.V == -1:
                if board_copy.state == GameState.DRAW:
                    v = 0.5
                else:
                    v = 1.0 if (
                        (board_copy.state == GameState.WHITE_WINS and board_copy.current_player_color == PlayerColor.WHITE) or
                        (board_copy.state == GameState.BLACK_WINS and board_copy.current_player_color == PlayerColor.BLACK)
                    ) else 0.0
                curr_node.V = v
            
            return {
                'node': curr_node,
                'virtual_losses': virtual_losses,
                'terminal_immediate': True,
            }
        
        # Handle unexplored node - prepare for batch evaluation
        leaf_hash = board_copy.zobrist_key
        
        # Get leaf data
        d_leaf = self.oracle._data_from_board(board_copy)
        if d_leaf is None:
            d_leaf = Data(x=torch.zeros((1, 13), dtype=torch.float32),
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                        batch=torch.zeros(1, dtype=torch.long))
        
        # Collect move specs
        move_specs: List[Tuple[str, Optional[Data], Optional[float], Move, Optional[int]]] = []
        valid_moves = list(board_copy.get_valid_moves())
        
        for m in valid_moves:
            board_copy.safe_play(m)
            state_hash = board_copy.zobrist_key
            
            if board_copy.state != GameState.IN_PROGRESS:
                if board_copy.state == GameState.DRAW:
                    V = 0.5
                else:
                    V = 1.0 if (
                        (board_copy.state == GameState.WHITE_WINS and board_copy.current_player_color == PlayerColor.WHITE) or
                        (board_copy.state == GameState.BLACK_WINS and board_copy.current_player_color == PlayerColor.BLACK)
                    ) else 0.0
                move_specs.append(('terminal', None, float(V), m, None))
            else:
                d_next = self.oracle._data_from_board(board_copy)
                if d_next is None:
                    move_specs.append(('terminal', None, 0.5, m, None))
                else:
                    move_specs.append(('predict', d_next, None, m, state_hash))
            board_copy.undo()
        
        return {
            'node': curr_node,
            'path_moves': path_moves,
            'virtual_losses': virtual_losses,
            'leaf_hash': leaf_hash,
            'leaf_data': d_leaf,
            'move_specs': move_specs,
            'terminal_immediate': False,
        }

    def _backpropagate_remove_virtual(self, leaf: Node_mcts, reward: float, virtual_losses: List[Node_mcts]) -> None:
        """Backpropagate and remove virtual losses."""
        # Mark as explored
        leaf.is_unexplored = False
        
        # Remove virtual losses from the path
        for node in virtual_losses:
            node.N -= self.virtual_loss
            node.W += self.virtual_loss
        
        # Normal backpropagation
        curr = leaf
        while curr is not None:
            curr.N += 1
            curr.W += reward
            curr.Q = curr.W / curr.N if curr.N > 0 else 0
            reward = 1 - reward
            curr = curr.parent

    def _uct_select(self, node: Node_mcts) -> Node_mcts:
        """Thread-safe UCT selection."""
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