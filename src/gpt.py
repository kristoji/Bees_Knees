import os
import json
import zipfile
from datetime import datetime
from glob import glob

import numpy as np
import networkx as nx
from ai.training import Training
from engine.enums import GameState
from engine.board import Board
from engineer import Engine

# PARAMS
VERBOSE = True
PRO_MATCHES_FOLDER = "pro_matches/games-Apr-3-2024/pgn"
GAME_TO_PARSE = 1000


def log_header(title: str, width: int = 60, char: str = '='):
    bar = char * width
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{bar}\n{ts} | {title.center(width - len(ts) - 3)}\n{bar}\n", flush=True)


def log_subheader(title: str, width: int = 50, char: str = '-'):
    bar = char * width
    print(f"{bar}\n{title.center(width)}\n{bar}", flush=True)


def unzip_new_archives(directory: str) -> None:
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    for filename in os.listdir(directory):
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(directory, filename)
            extract_to = os.path.splitext(zip_path)[0]
            if os.path.isdir(extract_to) and os.listdir(extract_to):
                print(f"Skipping '{filename}': already extracted.")
                continue
            os.makedirs(extract_to, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_to)
                print(f"Extracted '{filename}' → '{extract_to}/'")
            except zipfile.BadZipFile:
                print(f"Warning: '{filename}' is not a valid zip archive.")


def parse_pgn(file_path: str) -> list[str]:
    moves = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['):
                continue
            parts = line.split('.')
            if len(parts) > 1:
                moves.append(parts[1].strip())
    return moves


def board_to_graph(board: Board):
    """
    Convert board object into node features x and adjacency list edge_index.
    Features per node: [player_color(2), insect_type(8), pinned(1), is_articulation(1)].
    """
    # Map each occupied position to an index
    pos_to_bug = board._pos_to_bug  # dict: Position -> list of Bug
    positions = list(pos_to_bug.keys())
    pos_to_index = {pos: i for i, pos in enumerate(positions)}
    
    # insect_type one-hot
    types = ['Q','S','B','G','A','M','L','P']

    # Build node features
    x = []
    for pos in positions:
        bugs = pos_to_bug[pos]
        if bugs:
            top = bugs[-1]
            # player_color: [0,1] white, [0,0] black
            color_feat = [0,1] if top.player == "White" else [0,0]
            # Assume bug.abbrev gives e.g. 'Q', 'S', etc.
            t = top.abbrev
            insect_feat = [1 if t == tp else 0 for tp in types]
            # pinned: beetle on top
            pinned = 1 if len(bugs) > 1 and bugs[-1].abbrev == 'B' else 0
        else:
            # empty node (should not happen for occupied graph)
            color_feat = [1,0]
            insect_feat = [0]*8
            pinned = 0
        # placeholder articulation; will compute after adjacency
        x.append(color_feat + insect_feat + [pinned, 0])

    # Build adjacency
    G = nx.Graph()
    G.add_nodes_from(range(len(positions)))
    for pos, i in pos_to_index.items():
        # assume board._neighbors(pos) gives adjacent positions
        for npos in board._neighbors(pos):
            j = pos_to_index.get(npos)
            if j is not None:
                G.add_edge(i, j)
    # Compute articulation points
    arts = set(nx.articulation_points(G))
    # Update feature
    for idx in arts:
        # last feature index is is_articulation
        x[idx][-1] = 1

    # Convert G to edge_index list
    row, col = zip(*G.edges()) if G.number_of_edges() > 0 else ([], [])
    # undirected: add both directions
    edge_index = [list(row) + list(col), list(col) + list(row)]

    return x, edge_index, pos_to_index


def board_move_to_indices(move_str, pos_to_index):
    """
    Map move string (e.g. 'wB1 wG3/') to (src_idx, dst_idx).
    """
    parts = move_str.split()
    src_str = parts[0]
    dst_str = parts[1].rstrip('/\\-')
    # Find source index by matching stringify_move
    src_idx = dst_idx = None
    # pos_to_index maps Position; need reverse mapping: bug string to pos
    for pos, idx in pos_to_index.items():
        bugs = board._pos_to_bug[pos]
        for bug in bugs:
            if board.stringify_move(bug) == src_str:
                src_idx = idx
    for pos, idx in pos_to_index.items():
        bugs = board._pos_to_bug[pos]
        for bug in bugs:
            if board.stringify_move(bug) == dst_str:
                dst_idx = idx
    if src_idx is None or dst_idx is None:
        raise ValueError(f"Cannot map move {move_str} to indices")
    return src_idx, dst_idx


def save_graph(move_idx, pi_entry, v, board, save_dir, game_id):
    """
    Salva un JSON con grafo e target per la mossa corrente.
    """
    x, edge_index, pos_to_index = board_to_graph(board)
    N = len(x)
    # Costruisci move_adj e pi_target
    move_adj = [[0] * N for _ in range(N)]
    pi_target = []
    for move_str, prob in pi_entry:
        i, j = board_move_to_indices(move_str, pos_to_index)
        move_adj[i][j] = 1
        pi_target.append(prob)

    graph_dict = {
        'x': x,
        'edge_index': edge_index,
        'move_adj': move_adj,
        'pi': pi_target,
        'v': v
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"game_{game_id}_move_{move_idx}.json")
    with open(path, 'w') as f:
        json.dump(graph_dict, f)


def save_matrices(T_game, T_values, game, save_dir):
    game_shape = (0, *Training.INPUT_SHAPE)
    in_mats = np.empty(game_shape, dtype=np.float32)
    out_mats = np.empty(game_shape, dtype=np.float32)
    values = np.array(T_values, dtype=np.float32)
    for in_mat, out_mat in T_game:
        in_mats = np.append(in_mats, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
        out_mats = np.append(out_mats, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
    assert in_mats.shape[0] == out_mats.shape[0] == values.shape[0]
    os.makedirs(save_dir, exist_ok=True)
    log_subheader(f"Saving game {game} matrices")
    np.savez_compressed(
        os.path.join(save_dir, f"game_{game}.npz"),
        in_mats=in_mats,
        out_mats=out_mats,
        values=values
    )


def generate_matches(source_folder: str, verbose: bool = False, want_matrices: bool = True, want_graphs: bool = True) -> None:
    engine = Engine()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = f"data/{ts}/pro_matches"
    os.makedirs(base_dir, exist_ok=True)

    game = 1
    for fname in os.listdir(source_folder):
        path_pgn = os.path.join(source_folder, fname)
        moves = parse_pgn(path_pgn)
        engine.newgame(["Base+MLP"])
        T_game, T_values, pi_list, values = [], [], [], []
        graph_dir = os.path.join(base_dir, 'graphs')

        for move_idx, san in enumerate(moves):
            # compute current policy and value
            val_moves = engine.board.get_valid_moves()
            pi = {m: 1.0 if m == engine.board._parse_move(san) else 0.0 for m in val_moves}
            pi_entry = [(engine.board.stringify_move(m), p) for m, p in pi.items()]
            # add value placeholder (will adjust after final outcome)
            values.append(1.0)

            # save graph for this move
            if want_graphs:
                save_graph(move_idx, pi_entry, values[-1], engine.board, graph_dir, game)

            # collect matrices
            if want_matrices:
                mats = Training.get_matrices_from_board(engine.board, pi)
                T_game += mats
                T_values += [1.0 if m == engine.board._parse_move(san) else -1.0 for m in val_moves]

            # play
            engine.play(san, verbose=verbose)

        # final outcome
        log_subheader(f"Game {game} finished")
        outcome = engine.board.state
        final_mult = 1.0 if outcome == GameState.WHITE_WINS else -1.0 if outcome == GameState.BLACK_WINS else 0.0
        # adjust values and matrices
        values = [v * final_mult for v in values]
        if want_matrices:
            T_values = [v * final_mult for v in T_values]
            save_matrices(T_game, T_values, game, base_dir)

        # save final board state
        final_txt = os.path.join(base_dir, f"game_{game}_board.txt")
        with open(final_txt, 'w') as f_out:
            f_out.write(str(engine.board))

        if game >= GAME_TO_PARSE:
            log_header(f"Parsed {game} games; stopping.")
            break
        game += 1


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    if BASE_PATH.endswith("src"):
        BASE_PATH = BASE_PATH[:-3]
    os.chdir(BASE_PATH)
    generate_matches(source_folder=PRO_MATCHES_FOLDER, verbose=VERBOSE)
