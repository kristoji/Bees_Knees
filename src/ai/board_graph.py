"""Board -> graph conversion, the single source of truth for the network's input.

Kept free of torch so the dataset rebuild (tools/rebuild_dataset.py) can use exactly
the same code as inference without pulling in the model stack. OracleGNN._data_from_board
is a thin wrapper that turns the arrays into a PyG Data.

Node feature layout (13 columns):

    0       1.0 if the bug belongs to the side to move
    1..9    9-wide one-hot over the 8 bug types; slot 0 means "no bug" and is never
            set here, so column 1 is always zero. That is not an off-by-one: the
            training set uses the same encoding, so the layouts match. Do not change
            it without rebuilding the dataset.
    10      pinned   - another bug sits on top of this one
    11      pinning  - this bug sits on top of another
    12      articulation - this bug is alone on its cell and that cell is a cut
            vertex of the hive, i.e. moving it would break the one-hive rule
"""
import numpy as np

from engine.enums import BugType
from engine.game import NEIGHBOR_INDICES

NUM_FEATURES = 1 + len(list(BugType)) + 1 + 3
TYPE_COLUMN = {bug_type: 1 + i + 1 for i, bug_type in enumerate(BugType)}


def board_to_graph(board):
    """Return (x, edge_index) as numpy arrays, or None if the hive is empty.

    x is (num_nodes, 13) float32 and edge_index is (2, num_edges) int64, both in the
    node order produced by iterating the board's occupied cells and then the stack at
    each cell from the bottom up.
    """
    pos_to_bug = board._pos_to_bug
    if not pos_to_bug:
        return None

    current_player = board.current_player_color
    art_pos_set = board._art_pos

    # Nodes, grouped by stack height as we go.
    x_rows = []
    height_maps = []
    vertical = []
    node_idx = 0
    for pos, bugs in pos_to_bug.items():
        if not bugs:
            continue
        is_art = pos.index in art_pos_set
        num_bugs = len(bugs)
        pos_index = pos.index
        first_idx = node_idx
        for h, bug in enumerate(bugs):
            while len(height_maps) <= h:
                height_maps.append({})
            height_maps[h][pos_index] = node_idx
            x_rows.append((
                1.0 if bug.color == current_player else 0.0,
                TYPE_COLUMN[bug.type],
                1.0 if h < num_bugs - 1 else 0.0,   # pinned
                1.0 if h > 0 else 0.0,              # pinning
                1.0 if h == 0 and is_art else 0.0,  # articulation
            ))
            node_idx += 1
        for h in range(num_bugs - 1):
            i = first_idx + h
            vertical.append((i, i + 1))
            vertical.append((i + 1, i))

    total_nodes = node_idx
    if total_nodes == 0:
        return None

    x = np.zeros((total_nodes, NUM_FEATURES), dtype=np.float32)
    for i, (color, type_col, pinned, pinning, art) in enumerate(x_rows):
        x[i, 0] = color
        x[i, type_col] = 1.0
        x[i, -3] = pinned
        x[i, -2] = pinning
        x[i, -1] = art

    # Flat edges: same height, adjacent cells. Both directions appear because both
    # endpoints are visited.
    edges = vertical
    for pos_map in height_maps:
        for pos_index, i in pos_map.items():
            for neighbor_index in NEIGHBOR_INDICES[pos_index]:
                j = pos_map.get(neighbor_index)
                if j is not None:
                    edges.append((i, j))

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T.copy()
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return x, edge_index
