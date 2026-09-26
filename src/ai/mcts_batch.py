from __future__ import annotations
import math
from time import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from ai.brains import Brain
from ai.node_mcts import Node_mcts
from engine.board import Board
from engine.enums import GameState, PlayerColor, Error
from engine.game import Move

if TYPE_CHECKING:  # importing OracleGNN pulls in the whole model stack for a type hint
    from ai.oracleGNN import OracleGNN


# Spec kinds stored per legal move of an expanded leaf.
_TERMINAL = 0   # the move ends the game: V is known exactly
_CACHED = 1     # the resulting state is already in the value cache
_PREDICT = 2    # the resulting state needs a network evaluation


class MCTS_BATCH(Brain):
    """Monte Carlo tree search that batches the network calls.

    One leaf is collected at a time; the network is only asked once per batch, for
    the leaf values and the one-ply lookahead of every leaf in the batch at once.

    This is the sequential design that used to live in mcts_batch_bak.py. The
    thread-based rewrite that replaced it called Board.copy(), which did not exist,
    so it raised AttributeError on the first descent; its workers were also pure
    Python under the GIL with the tree lock held across the whole selection, so it
    could not have been faster than this even once fixed. Parallelism belongs at the
    process level, with the network behind an inference server.
    """

    # Values are keyed by zobrist and stay valid as long as the weights do, so the
    # cache is kept across moves. Bounded so a long game cannot grow it without end.
    CACHE_LIMIT = 1_000_000

    def __init__(self, oracle: "OracleGNN", exploration_weight: int = 10, num_rollouts: int = 1024,
                 time_limit: float = float("inf"), batch_size: int = 32,
                 dirichlet_eps: float = 0.0, dirichlet_alpha: float = 0.3,
                 temperature_plies: int = 0, debug: bool = False) -> None:
        super().__init__()
        self.init_node: Optional[Node_mcts] = None
        self.init_board: Optional[Board] = None
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        self.last_rollouts = 0
        self.oracle = oracle
        self.time_limit = time_limit
        self.batch_size = max(1, int(batch_size))
        self.epsilon = 0.05
        self.start_time = time()
        self.debug = debug
        self.hashmap: Dict[int, float] = {}
        # Set when the oracle carries a policy head: the prior then comes from the
        # network instead of a one-ply lookahead, so an expansion is one forward pass
        # instead of b+1 (b averages 62 on this corpus).
        self.use_policy = getattr(oracle, "policy_net", None) is not None
        self.policy_cache: Dict[int, tuple] = {}
        # Self-play needs the search to be stochastic. Without these the tree is a pure
        # argmax end to end — no sampling in selection, no noise at the root, and the
        # move played is the argmax of the visit counts — so a generator would produce
        # the same game over and over. Both default to off so match play is unchanged.
        self.dirichlet_eps = dirichlet_eps       # AlphaZero uses 0.25
        self.dirichlet_alpha = dirichlet_alpha   # roughly 10 / branching factor
        self.temperature_plies = temperature_plies
        self._noise_applied = False

    # -------------------------
    #  Public API
    # -------------------------
    def calculate_best_move(self, board: Board, restriction: str, value: int, debug: bool = False) -> str:
        if restriction == "depth":
            self.time_limit = float("inf")
            start = time()
            self.num_rollouts = value
            self.run_simulation_from(board, debug=debug)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Time taken: {time() - start:.2f} seconds")
                print(f"Cache size: {len(self.hashmap)} unique states")
            return a
        elif restriction == "time":
            self.time_limit = value
            self.start_time = time()
            self.run_simulation_from(board, debug=debug)
            a: str = self.action_selection(training=False, debug=debug)
            if debug:
                print(f"Rollouts done: {self.last_rollouts}")
                print(f"Cache size: {len(self.hashmap)} unique states")
            return a
        else:
            raise Error("Invalid restriction for MCTS")

    def action_selection(self, training: bool = False, debug: bool = False) -> str:
        # NOTE: this descends init_node into the chosen child, which is what makes the
        # tree reusable next move. Anything that wants the root's statistics, such as
        # get_moves_probs() for a self-play policy target, must read them first.
        node = self.choose(training=training, debug=debug)
        self.init_node = node
        return self.init_board.stringify_move(node.move)

    def _apply_root_noise(self) -> None:
        """Mix Dirichlet noise into the root's priors, once per search.

        Applied after the root is expanded, which only happens inside the search, so
        it cannot be done up front.
        """
        children = self.init_node.children
        if not children or self.dirichlet_eps <= 0 or self._noise_applied:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        eps = self.dirichlet_eps
        for child, n in zip(children, noise):
            child.P = (1 - eps) * child.P + eps * float(n)
        self._noise_applied = True

    def choose(self, training: bool, debug: bool = False) -> Node_mcts:
        if debug:
            print("\n\nChildren of root node (sorted by visits):\n", flush=True)
            for child in sorted(self.init_node.children, key=lambda x: x.N, reverse=True):
                print(f"Move: {self.init_board.stringify_move(child.move)} -> "
                      f"N = {child.N}, W = {child.W}, Q = {child.Q}, P = {child.P}, V = {child.V}",
                      flush=True)
        children = self.init_node.children
        # Early in a self-play game, sample proportionally to the visit counts instead
        # of taking the argmax, so openings vary. Later moves stay greedy.
        if self.temperature_plies and self.init_board.turn < self.temperature_plies:
            counts = np.array([c.N for c in children], dtype=np.float64)
            if counts.sum() > 0:
                probs = counts / counts.sum()
                return children[int(np.random.choice(len(children), p=probs))]
        return max(children, key=lambda n: n.N)

    def get_moves_probs(self) -> Dict[Move, float]:
        """Visit distribution at the root: the policy target for self-play.

        This is the improvement the search adds on top of the raw policy head, and
        training toward it is what lets the network exceed the player that produced
        the data.
        """
        total = max(1, sum(child.N for child in self.init_node.children))
        return {child.move: child.N / total for child in self.init_node.children}

    # -------------------------
    #  Core search
    # -------------------------
    def run_simulation_from(self, board: Board, debug: bool = False) -> None:
        self.init_board = board
        last_move = board.moves[-1] if board.moves else None

        if self.init_node is None:
            self.init_node = Node_mcts(last_move, board.state, board.current_player_color,
                                       board.zobrist_key)
        else:
            # Descend into the child matching the position actually reached. If the
            # opponent played something outside the tree we simply start a new one:
            # the previous code raised here, which killed the search.
            for child in self.init_node.children:
                if child.hash == board.zobrist_key:
                    self.init_node = child
                    self.init_node.reset()
                    break
            else:
                self.init_node = Node_mcts(last_move, board.state, board.current_player_color,
                                           board.zobrist_key)

        if len(self.hashmap) > self.CACHE_LIMIT:
            self.hashmap.clear()
        self._noise_applied = False

        terminal_states = 0
        completed = 0
        flush_counter = 0
        cache_hits = 0
        cache_misses = 0

        pending: List[dict] = []
        leaf_datas: List[Data] = []
        leaf_owners: List[int] = []   # pending index for each entry of leaf_datas
        next_datas: List[Data] = []

        def flush_batch():
            nonlocal pending, leaf_datas, leaf_owners, next_datas
            nonlocal completed, flush_counter, cache_hits, cache_misses

            if not pending:
                return

            hashmap = self.hashmap
            leaf_values = self.oracle.predict_values_batch_from_data(leaf_datas, use_sigmoid=True) \
                if leaf_datas else []
            next_values = self.oracle.predict_values_batch_from_data(next_datas, use_sigmoid=True) \
                if next_datas else []
            cache_misses += len(leaf_datas) + len(next_datas)

            for owner, value in zip(leaf_owners, leaf_values):
                info = pending[owner]
                info['leaf_v'] = value
                hashmap[info['leaf_hash']] = value

            board = self.init_board
            for info in pending:
                v = info['leaf_v']
                specs = info['move_specs']

                scores: List[float] = []
                moves: List[Move] = []
                for kind, slot, value, move, state_hash in specs:
                    if kind is _PREDICT:
                        V = float(next_values[slot])
                        hashmap[state_hash] = V
                    else:
                        V = float(value)
                        if kind is _CACHED:
                            cache_hits += 1
                    scores.append(1 - V)
                    moves.append(move)

                if scores:
                    arr = np.array(scores, dtype=np.float32)
                    arr = np.exp(arr - np.max(arr))
                    arr /= np.sum(arr)
                    pi = {m: float(p) for m, p in zip(moves, arr.tolist())}
                else:
                    pi = {}

                path_moves = info['path_moves']
                for m in path_moves:
                    board.safe_play(m)
                node = info['node']
                node.expand(board, v, pi)
                self._backpropagate_non_N(node, 1 - node.reward())
                if path_moves:
                    board.undo(len(path_moves))
                completed += 1

            flush_counter += 1
            pending = []
            leaf_datas = []
            leaf_owners = []
            next_datas = []

        def collect_and_maybe_flush() -> bool:
            """Collect one leaf. Returns False when the tree cannot produce one."""
            nonlocal terminal_states, completed, cache_hits
            collected = self._collect_leaf_for_batch()
            if collected is None:
                return False
            if collected['terminal_immediate']:
                node = collected['node']
                self._backpropagate(node, 1 - node.reward())
                completed += 1
                terminal_states += 1
                return True

            # Counted as visited straight away so UCT steers the rest of the batch
            # away from this leaf; W and Q follow when the batch is evaluated.
            self._backpropagate_N(collected['node'])

            if 'leaf_v' in collected:
                cache_hits += 1
            else:
                leaf_owners.append(len(pending))
                leaf_datas.append(collected['leaf_data'])

            specs = collected['move_specs']
            for i, spec in enumerate(specs):
                if spec[0] is _PREDICT:
                    specs[i] = (_PREDICT, len(next_datas), None, spec[3], spec[4])
                    next_datas.append(spec[1])
            pending.append(collected)

            # The first batches are deliberately small so the root gets expanded and
            # the tree has some shape before full-size batches are worth collecting.
            threshold = 1 if flush_counter == 0 else \
                self.batch_size // 4 if flush_counter == 1 else self.batch_size
            if len(pending) >= threshold:
                flush_batch()
            return True

        if self.use_policy:
            completed, terminal_states = self._run_policy(debug)
        elif self.time_limit < float("inf"):
            deadline = self.start_time + self.time_limit - self.epsilon
            while time() < deadline:
                if not collect_and_maybe_flush():
                    break
            flush_batch()
        else:
            target = int(self.num_rollouts)
            while completed < target:
                if not collect_and_maybe_flush():
                    break
            flush_batch()

        if debug:
            print(f"\nTerminal states {terminal_states}/{completed} rollouts")
            total = cache_hits + cache_misses
            if total:
                print(f"Cache hits: {cache_hits}, misses: {cache_misses}, "
                      f"hit rate: {cache_hits / total * 100:.1f}%")
        self.last_rollouts = completed

    # -------------------------
    #  Policy-head search
    # -------------------------
    def _run_policy(self, debug=False):
        """Batched search that asks the network once per leaf, values and prior together."""
        completed = terminal_states = 0
        pending = []
        board = self.init_board
        deadline = (self.start_time + self.time_limit - self.epsilon
                    if self.time_limit < float("inf") else None)
        target = int(self.num_rollouts)
        flush_counter = 0

        def flush():
            nonlocal pending, completed, flush_counter
            if not pending:
                return
            fresh = [info for info in pending if "value" not in info]
            if fresh:
                results = self.oracle.evaluate_positions([i["board"] for i in fresh])
                for info, (value, prior) in zip(fresh, results):
                    info["value"], info["prior"] = value, prior
                    if len(self.policy_cache) < self.CACHE_LIMIT:
                        self.policy_cache[info["leaf_hash"]] = (value, prior)
            for info in pending:
                path_moves = info["path_moves"]
                for m in path_moves:
                    board.safe_play(m)
                node = info["node"]
                node.expand(board, info["value"], info["prior"])
                self._backpropagate_non_N(node, 1 - node.reward())
                if path_moves:
                    board.undo(len(path_moves))
                completed += 1
            self._apply_root_noise()
            pending = []
            flush_counter += 1

        while True:
            if deadline is not None:
                if time() >= deadline:
                    break
            elif completed >= target:
                break

            collected = self._collect_leaf_policy()
            if collected is None:
                break
            if collected["terminal_immediate"]:
                node = collected["node"]
                self._backpropagate(node, 1 - node.reward())
                completed += 1
                terminal_states += 1
                continue

            self._backpropagate_N(collected["node"])
            cached = self.policy_cache.get(collected["leaf_hash"])
            if cached is not None:
                collected["value"], collected["prior"] = cached
            pending.append(collected)

            threshold = 1 if flush_counter == 0 else \
                self.batch_size // 4 if flush_counter == 1 else self.batch_size
            if len(pending) >= threshold:
                flush()
        flush()
        return completed, terminal_states

    def _collect_leaf_policy(self) -> Optional[dict]:
        """Descend to a leaf and snapshot it; the network is asked later, in a batch."""
        curr_node = self.init_node
        board = self.init_board
        path_moves: List[Move] = []

        while not (curr_node.is_unexplored or curr_node.is_terminal):
            curr_node = self._uct_select(curr_node)
            board.safe_play(curr_node.move)
            path_moves.append(curr_node.move)

        if curr_node.is_terminal:
            if curr_node.V == -1:
                if board.state == GameState.DRAW:
                    v = 0.5
                else:
                    v = 1.0 if (
                        (board.state == GameState.WHITE_WINS and board.current_player_color == PlayerColor.WHITE) or
                        (board.state == GameState.BLACK_WINS and board.current_player_color == PlayerColor.BLACK)
                    ) else 0.0
                curr_node.V = v
            if path_moves:
                board.undo(len(path_moves))
            return {"node": curr_node, "path_moves": path_moves, "terminal_immediate": True}

        if curr_node.is_expanded:
            curr_node.reset_children()
            if path_moves:
                board.undo(len(path_moves))
            return {"node": curr_node, "path_moves": path_moves, "terminal_immediate": True}

        # Board.copy() is cheap now that the Zobrist tables are shared, so the leaf can
        # be snapshotted and evaluated later without replaying the path twice.
        result = {"node": curr_node, "path_moves": path_moves,
                  "leaf_hash": board.zobrist_key, "board": board.copy(),
                  "terminal_immediate": False}
        if path_moves:
            board.undo(len(path_moves))
        return result

    # -------------------------
    #  Leaf collection
    # -------------------------
    def _collect_leaf_for_batch(self) -> Optional[dict]:
        """Walk down with UCT to an unexplored or terminal node and describe the work.

        The value cache is consulted here rather than after the fact: a cached state
        needs no Data at all, and building the graph is the expensive part. The
        previous version built a Data for the leaf and for every legal move first and
        only then looked in the cache, so a hit saved the network call but not the
        graph construction.
        """
        curr_node = self.init_node
        board = self.init_board
        path_moves: List[Move] = []

        while not (curr_node.is_unexplored or curr_node.is_terminal):
            curr_node = self._uct_select(curr_node)
            board.safe_play(curr_node.move)
            path_moves.append(curr_node.move)

        if curr_node.is_terminal:
            if curr_node.V == -1:   # a draw needs the heuristic once
                if board.state == GameState.DRAW:
                    v = 0.5
                else:
                    v = 1.0 if (
                        (board.state == GameState.WHITE_WINS and board.current_player_color == PlayerColor.WHITE) or
                        (board.state == GameState.BLACK_WINS and board.current_player_color == PlayerColor.BLACK)
                    ) else 0.0
                curr_node.V = v
            if path_moves:
                board.undo(len(path_moves))
            return {'node': curr_node, 'path_moves': path_moves, 'terminal_immediate': True}

        if curr_node.is_expanded:
            # Already expanded on an earlier move and then reset: its children keep
            # their priors, so re-running the network over all of them would be thrown
            # away by expand(). Just re-open it.
            curr_node.reset_children()
            if path_moves:
                board.undo(len(path_moves))
            return {'node': curr_node, 'path_moves': path_moves, 'terminal_immediate': True}

        hashmap = self.hashmap
        oracle = self.oracle
        result = {
            'node': curr_node,
            'path_moves': path_moves,
            'leaf_hash': board.zobrist_key,
            'terminal_immediate': False,
        }

        leaf_v = hashmap.get(result['leaf_hash'])
        if leaf_v is None:
            d_leaf = oracle._data_from_board(board)
            if d_leaf is None:
                d_leaf = Data(x=torch.zeros((1, 13), dtype=torch.float32),
                              edge_index=torch.zeros((2, 0), dtype=torch.long),
                              batch=torch.zeros(1, dtype=torch.long))
            result['leaf_data'] = d_leaf
        else:
            result['leaf_v'] = leaf_v

        move_specs: List[Tuple] = []
        for m in board.get_valid_moves():
            board.safe_play(m)
            state_hash = board.zobrist_key
            if board.state != GameState.IN_PROGRESS:
                if board.state == GameState.DRAW:
                    V = 0.5
                else:
                    V = 1.0 if (
                        (board.state == GameState.WHITE_WINS and board.current_player_color == PlayerColor.WHITE) or
                        (board.state == GameState.BLACK_WINS and board.current_player_color == PlayerColor.BLACK)
                    ) else 0.0
                move_specs.append((_TERMINAL, None, float(V), m, None))
            else:
                cached = hashmap.get(state_hash)
                if cached is not None:
                    move_specs.append((_CACHED, None, cached, m, state_hash))
                else:
                    d_next = oracle._data_from_board(board)
                    if d_next is None:
                        move_specs.append((_TERMINAL, None, 0.5, m, None))
                    else:
                        move_specs.append((_PREDICT, d_next, None, m, state_hash))
            board.undo()

        result['move_specs'] = move_specs
        if path_moves:
            board.undo(len(path_moves))
        return result

    # -------------------------
    #  Backpropagation
    # -------------------------
    def _backpropagate(self, leaf: Node_mcts, reward: float) -> None:
        leaf.is_unexplored = False
        while leaf is not None:
            leaf.N += 1
            leaf.W += reward
            leaf.Q = leaf.W / leaf.N
            reward = 1 - reward
            leaf = leaf.parent

    def _backpropagate_N(self, leaf: Node_mcts) -> None:
        """Count the visit now; the value arrives when the batch is evaluated."""
        while leaf is not None:
            leaf.N += 1
            leaf = leaf.parent

    def _backpropagate_non_N(self, leaf: Node_mcts, reward: float) -> None:
        """Apply the value of a visit already counted by _backpropagate_N."""
        leaf.is_unexplored = False
        while leaf is not None:
            leaf.W += reward
            leaf.Q = leaf.W / leaf.N
            reward = 1 - reward
            leaf = leaf.parent

    def _uct_select(self, node: Node_mcts) -> Node_mcts:
        c = self.exploration_weight * math.sqrt(max(1, node.N))
        best = None
        best_score = -float("inf")
        for n in node.children:
            score = n.Q + c * n.P / (1 + n.N)
            if score > best_score:
                best_score = score
                best = n
        return best
