"""Index legal moves against the graph's nodes, so a policy head can score them all at once.

The value-only setup builds its prior by evaluating the network on every child, which
costs b+1 forward passes per expansion and is where a search spends nearly all its time
(a typical mid-game position has 40-75 legal moves). A policy head scores every move in
one pass, but it needs each move expressed in terms of node embeddings.

The graph has a node per *piece*, not per cell, so a move's destination — an empty cell
in almost every case — has no node of its own. Rather than add nodes for empty cells,
which would change the input representation and invalidate the trained value network and
the shards, a destination is described by the pieces it touches:

    src   the node of the bug being moved, or -1 when it is still in hand
    bug   the bug's index in 0..27, which lets the head embed a piece in hand and
          tells it what is moving in every case
    dst   up to six node indices: the top piece of the destination cell when it is
          occupied (a beetle or mosquito climbing), otherwise the top pieces of the
          occupied cells around it, padded with -1

Every legal destination touches the hive, so `dst` is empty only on the very first
placement of a game, where the board has no pieces at all.
"""
import numpy as np

from engine.enums import BugType
from engine.game import NEIGHBOR_INDICES

MAX_DST = 6


def index_moves(board, moves=None):
    """Return (moves, src, bug, dst, dst_len) aligned with the graph's node order.

    The node order is the one board_graph.board_to_graph produces: cells in
    _pos_to_bug insertion order, and within a cell the stack from the bottom up.
    """
    if moves is None:
        moves = list(board.get_valid_moves())

    # Same walk as board_to_graph, so the indices line up.
    node_of_bug = {}
    top_node_of_cell = {}
    node_idx = 0
    for pos, bugs in board._pos_to_bug.items():
        if not bugs:
            continue
        for bug in bugs:
            node_of_bug[bug] = node_idx
            node_idx += 1
        top_node_of_cell[pos.index] = node_idx - 1

    n = len(moves)
    src = np.full(n, -1, dtype=np.int32)
    bug = np.zeros(n, dtype=np.int16)
    dst = np.full((n, MAX_DST), -1, dtype=np.int32)
    dst_len = np.zeros(n, dtype=np.int8)

    for i, move in enumerate(moves):
        if move is None:          # a pass has nothing to point at
            continue
        bug[i] = move.bug.index
        node = node_of_bug.get(move.bug)
        if node is not None:
            src[i] = node

        target = move.destination.index
        top = top_node_of_cell.get(target)
        if top is not None:
            # Climbing onto an occupied cell: the cell itself is a node.
            dst[i, 0] = top
            dst_len[i] = 1
            continue
        k = 0
        for neighbor in NEIGHBOR_INDICES[target]:
            top = top_node_of_cell.get(neighbor)
            if top is not None:
                dst[i, k] = top
                k += 1
                if k == MAX_DST:
                    break
        dst_len[i] = k

    return moves, src, bug, dst, dst_len
