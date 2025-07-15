import os
import json
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from ai.training import Training
from engine.enums import GameState
from engine.board import Board
from engineer import Engine
from engine.game import Bug, PlayerColor, Move, Position
from engine.enums import Command, BugType, Direction

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
    valid_moves = board.get_valid_moves()
    # DEST POSITIONS
    positions = set(move.destination for move in valid_moves if move.destination is not None)
    # Add occupied positions, avoiding duplicates
    pos_to_bug = board._pos_to_bug
    positions.update(pos_to_bug.keys())
    positions = list(positions)
    pos_bug_to_index = {}
    i = 0
    for pos in positions:
        bugs = board._bugs_from_pos(pos)
        for bug in bugs:
            pos_bug_to_index[(pos,bug)] = i
            i += 1

    all_bugs = {
        (BugType.QUEEN_BEE, PlayerColor.WHITE) : 1,
        (BugType.SPIDER, PlayerColor.WHITE) : 2,
        (BugType.BEETLE, PlayerColor.WHITE) : 2,
        (BugType.GRASSHOPPER, PlayerColor.WHITE) : 3,
        (BugType.SOLDIER_ANT, PlayerColor.WHITE) : 3,
        (BugType.MOSQUITO, PlayerColor.WHITE) : 1,
        (BugType.LADYBUG, PlayerColor.WHITE) : 1,
        (BugType.PILLBUG, PlayerColor.WHITE) : 1,

        (BugType.QUEEN_BEE, PlayerColor.BLACK) : 1,
        (BugType.SPIDER, PlayerColor.BLACK) : 2,
        (BugType.BEETLE, PlayerColor.BLACK) : 2,
        (BugType.GRASSHOPPER, PlayerColor.BLACK) : 3,
        (BugType.SOLDIER_ANT, PlayerColor.BLACK) : 3,
        (BugType.MOSQUITO, PlayerColor.BLACK) : 1,
        (BugType.LADYBUG, PlayerColor.BLACK) : 1,
        (BugType.PILLBUG, PlayerColor.BLACK) : 1,
    }

    all_bugs_final = {
        (BugType.QUEEN_BEE, PlayerColor.WHITE) : 1,
        (BugType.SPIDER, PlayerColor.WHITE) : 2,
        (BugType.BEETLE, PlayerColor.WHITE) : 2,
        (BugType.GRASSHOPPER, PlayerColor.WHITE) : 3,
        (BugType.SOLDIER_ANT, PlayerColor.WHITE) : 3,
        (BugType.MOSQUITO, PlayerColor.WHITE) : 1,
        (BugType.LADYBUG, PlayerColor.WHITE) : 1,
        (BugType.PILLBUG, PlayerColor.WHITE) : 1,

        (BugType.QUEEN_BEE, PlayerColor.BLACK) : 1,
        (BugType.SPIDER, PlayerColor.BLACK) : 2,
        (BugType.BEETLE, PlayerColor.BLACK) : 2,
        (BugType.GRASSHOPPER, PlayerColor.BLACK) : 3,
        (BugType.SOLDIER_ANT, PlayerColor.BLACK) : 3,
        (BugType.MOSQUITO, PlayerColor.BLACK) : 1,
        (BugType.LADYBUG, PlayerColor.BLACK) : 1,
        (BugType.PILLBUG, PlayerColor.BLACK) : 1,
    }


    # Build node features
    x = []
    for pos in positions:
        bugs = board._bugs_from_pos(pos)
        art_pos = 1 if pos in board._art_pos else 0

        # PLACING PLAYED BUGS IN THE NODES SET
        for i, bug in enumerate(bugs):
            color_feat = [0, 1] if bug.color == PlayerColor.WHITE else [0, 0]
            insect_feat = [1 if bug.type == tp else 0 for tp in BugType] # ------------> Maybe we can use BugType enum directly
            pinned = 1 if i < len(bugs) - 1 else 0 # Not pinned if on top of stack
            art_pos_feat = 1 if i == 0 and art_pos else 0 # Only the bug at the bottom of the stack is an articulation point

            all_bugs[(bug.type, bug.color)] -= 1
            if all_bugs[(bug.type, bug.color)] < 0:
                print(f"Bug {bug.color},{bug.type} appears more times than expected in the board state.")
                raise ValueError(f"NEGATIVE BUG COUNT!")
            
            pos_bug_to_index[(pos, bug)] = len(x)  # Map position and bug to index
            x.append(color_feat + insect_feat + [pinned, art_pos_feat])  #        
        # PLACING EMPTY NEIGHBOR CELLS IN THE NODES SET (destination of valid moves)
        else:
            # If no bugs, use empty features
            color_feat = [1, 0]
            insect_feat = [0] * len(BugType) # ------------> Maybe we can use BugType enum directly
            pinned = 0
            art_pos_feat = 0
            pos_bug_to_index[(pos, None)] = len(x)  # Map position to index
            x.append(color_feat + insect_feat + [pinned, art_pos_feat])  
    
    # PLACING NOT PLAYED BUGS IN THE NODES SET
    for (bug_type, color), count in all_bugs.items():
        max_count = all_bugs_final[(bug_type, color)]
        for i in range(max_count - count + 1, max_count + 1):
            color_feat = [0, 1] if color == PlayerColor.WHITE else [0, 0]
            insect_feat = [1 if bug_type == tp else 0 for tp in BugType]
            pinned = 0
            art_pos_feat = 0
            x.append(color_feat + insect_feat + [pinned, art_pos_feat])
            if max_count > 1: # for Bugs with multiple instances
                pos_bug_to_index[(None, Bug(color, bug_type, i))] = len(x) - 1  # Add dummy node for missing bugs
            else: # for Bugs with single instance
                pos_bug_to_index[(None, Bug(color, bug_type))] = len(x) - 1  # Add dummy node for missing bugs
            # print(f"Adding dummy node for {color} {bug_type} {max_count - i - 1} at index {len(x) - 1}")
    # Build adjacency
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))  # Add nodes for each bug position
    # =====================================================================
    # TO-THINK: is it useful to assign labels to the edges of the graph?
    # The label can refer the type of neighbor direction
    # =====================================================================
    
    for (pos, bug), i in pos_bug_to_index.items():

        if pos is None: # for non placed bugs
            continue

        for d in Direction:
            npos = board._get_neighbor(pos, d)
            # =====================================================================
            # TO-THINK: should we consider only the last bug in the stack?
            # [i.e] neighbor = pos_to_bug[npos][-1] if npos in pos_to_bug else []
            # =====================================================================
            neighbors = board._bugs_from_pos(npos)
            for bug_other in neighbors:
                j = pos_bug_to_index.get((npos, bug_other))
                if j is not None:
                    G.add_edge(i, j)

    

    # Convert G to edge_index list
    row, col = zip(*G.edges()) if G.number_of_edges() > 0 else ([], [])
    # undirected: add both directions
    edge_index = [list(row) + list(col), list(col) + list(row)]

    return x, edge_index, pos_bug_to_index

def plot_and_save_graph(edge_index, save_path="graph.png", figsize=(8,8), dpi=300):
    """
    Given edge_index = [ [u1,u2,...], [v1,v2,...] ], build a NetworkX graph,
    draw it with a spring layout, and save it to `save_path`.
    """
    # 1. Build the graph
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    # 2. Choose a layout
    pos = nx.spring_layout(G)

    # 3. Plot
    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        with_labels=True,            # show node indices
        node_size=500,               # size of nodes
        node_color="skyblue",        # fill color
        edge_color="gray",           # edge color
        font_size=10,
    )
    plt.axis('off')  # turn off axes

    # 4. Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {save_path}")


def board_move_to_indices(move: Move, pos_bug_to_index: dict[tuple[Position, Bug], int]) -> tuple[int, int]:
    """
    Map move string (e.g. 'wB1 wG3/') to (src_idx, dst_idx).
    ATTENZIONE AL PILLBUG
    """

    src_idx = pos_bug_to_index.get((move.origin, move.bug))

    dst_idx = pos_bug_to_index.get((move.destination, None))

    return src_idx, dst_idx


def save_graph(move_idx, pi_entry: list[tuple[Move, float]], v, board, save_dir, game_id):
    """
    Salva un JSON con grafo e target per la mossa corrente.
    """
    x, edge_index, pos_bug_to_index = board_to_graph(board)
    plot_and_save_graph(edge_index, save_path=os.path.join(save_dir, f"game_{game_id}_move_{move_idx}.png"))
    N = len(x)
    # Costruisci move_adj e pi_target
    move_adj = [[0] * N for _ in range(N)]
    pi_target = []
    for move, prob in pi_entry:
        i, j = board_move_to_indices(move, pos_bug_to_index)
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
    print(f"Game {game} matrices: {in_mats.shape}, {out_mats.shape}, {values.shape}")
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
    graph_dir = os.path.join(base_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    

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
            pi_entry = [(m,p) for m, p in pi.items()]
            # pi_entry = [(engine.board.stringify_move(m), p) for m, p in pi.items()]
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
