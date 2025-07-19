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
PLOTS = False


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


def board_to_graph(board: Board) -> tuple[list[list[float]], list[list[int]], dict[tuple[Position, Bug], int]]:
    """
    Convert board object into node features x (using floats) and adjacency list edge_index.
    Features per node: [player_color, insect_type, pinned, is_articulation] as floats:
      - player_color: 1.0 for White, 0.5 for empty, 0.0 for Black
      - insect_type: scaled float in [0,1], e.g. type_index / (num_types)
      - pinned: float 1.0 or 0.0
      - is_articulation: float 1.0 or 0.0
    """
    valid_moves = board.get_valid_moves()

    # ===================================== DESTINATIONS POSITIONS =====================================
    dest_positions = list(set(move.destination for move in valid_moves if move.destination is not None))

    # ===================================== BUGS POSITIONS =====================================
    pos_to_bug = board._pos_to_bug
    bugs_positions = list(pos_to_bug.keys())

    # ===================================== DICT MAPPING NODE (pos,bug) TO INDEX =====================================
    pos_bug_to_index = {}

    pos_height_to_bug: dict[tuple[Position, int], Bug] = {}

    # ===================================== ALL BUGS AND THEIR COUNT =====================================
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

    # ====================================== CONST ALL BUGS COUNT =====================================
    all_bugs_final = all_bugs.copy()

    # ====================================== TYPE TO INDEX =====================================
    types = list(BugType)
    type_to_index = {bug_type: (i + 1) for i, bug_type in enumerate(types)}
    num_types = len(types) + 1  # +1 for empty node (no bug)

    # ===================================== NODES LIST =====================================
    x = []

    # ===================================== EVERY PLACED BUG IS A NODE =====================================
    for pos in bugs_positions:
        bugs = board._bugs_from_pos(pos)
        art_pos = 1 if pos in board._art_pos else 0

        # PLACING PLAYED BUGS IN THE NODES SET
        for h, bug in enumerate(bugs):
            color_feat = 1.0 if bug.color==PlayerColor.WHITE else 0.0
            insect_feat = (type_to_index[bug.type]) / num_types  # Scale type index to [0,1], 0 is empty node
            pinned = 1.0 if h < len(bugs)-1 else 0.0
            art = 1.0 if h==0 and art_pos else 0.0
            pos_bug_to_index[(pos, bug)] = len(x)  # Map position and bug to index
            pos_height_to_bug[(pos, h)] = bug  # Map position and height to bug      
            x.append([color_feat, insect_feat, pinned, art])

            all_bugs[(bug.type, bug.color)] -= 1
            if all_bugs[(bug.type, bug.color)] < 0:
                print(f"Bug {bug.color},{bug.type} appears more times than expected in the board state.")
                raise ValueError(f"NEGATIVE BUG COUNT!")
                
    # ===================================== EVERY DESTINATION IS AN EMPTY NODE =====================================
    for pos in dest_positions:
        #PLACING EMPTY NEIGHBOR CELLS IN THE NODES SET (destination of valid moves)
        # ======= !!! NOTE !!! =======
        # the following if is commented because we always want to have a free node on top of every bug for beetle movement
        #   ---->   if not bugs:  # (If no bugs in this position) !!! INCORRECT !!!
        # ============================
        # If no bugs, use empty features
        color_feat = 0.5
        insect_feat = 0.0
        pinned = 0.0
        art_pos_feat = 0.0
        pos_bug_to_index[(pos, None)] = len(x)  # Map position to index
        x.append([color_feat, insect_feat, pinned, art_pos_feat])
        height = len(board._bugs_from_pos(pos))  # Height of the stack at this position
        pos_height_to_bug[(pos, height)] = None  # Map position and height to None (empty cell)

    # ===================================== EVERY NON PLACED BUG IS A NODE =====================================
    for (bug_type, color), count in all_bugs.items():
        max_count = all_bugs_final[(bug_type, color)]
        for i in range(max_count - count + 1, max_count + 1):
            color_feat = [0, 1] if color == PlayerColor.WHITE else [0, 0]
            insect_feat = (type_to_index[bug_type]) / num_types  # Scale type index to [0,1], 0 is empty node
            pinned = 0.0
            art_pos_feat = 0.0
            if max_count > 1: # for Bugs with multiple instances
                pos_bug_to_index[(None, Bug(color, bug_type, i))] = len(x) # Add dummy node for missing bugs
            else: # for Bugs with single instance
                pos_bug_to_index[(None, Bug(color, bug_type))] = len(x) # Add dummy node for missing bugs
            x.append([color_feat, insect_feat, pinned, art_pos_feat])
            # print(f"Adding dummy node for {color} {bug_type} {max_count - i - 1} at index {len(x) - 1}")

    # ===================================== BUILDING THE GRAPH =====================================
    G = nx.Graph()
    G.add_nodes_from(range(len(x)))  # Add nodes for each bug position

            # =====================================================================
            # TO-THINK: is it useful to assign labels to the edges of the graph?
            # The label can refer the type of neighbor direction
            # =====================================================================
    
    # ===================================== ADDING EDGES FOR NEIGHBORS NODES =====================================
    
    # Map mapping to index for lookup
    node_index = {key:i for i,key in enumerate(pos_height_to_bug)}
    # Flat edges: same height across neighbors
    for (pos, h), bug in pos_height_to_bug.items():
        if pos is None: 
            raise ValueError(f"Position {pos} is None, cannot add edges for non-placed bugs.")
        i = node_index[(pos,h)]
        for d in Direction.flat():
            npos = board._get_neighbor(pos,d)
            j = node_index.get((npos,h))
            if j is not None:
                G.add_edge(i,j)
    
    # Above/below within same stack
    for pos, bugs in pos_to_bug.items():
        for below, above in zip(bugs[:-1], bugs[1:]):
            i = pos_bug_to_index[(pos, below)]
            j = pos_bug_to_index[(pos, above)]
            G.add_edge(i, j)
            G.add_edge(j, i)
        
        if pos in dest_positions and bugs and bugs[-1]:  # If this position is a destination, connect the top bug to the empty cell above (a possible destination)
            top_bug = pos_bug_to_index[(pos, bugs[-1])]
            empty_node = pos_bug_to_index[(pos, None)]  
            G.add_edge(top_bug, empty_node)
            G.add_edge(empty_node, top_bug)

    # ===================================== Convert G to edge_index list =====================================
    row, col = zip(*G.edges()) if G.number_of_edges() > 0 else ([], [])
    # undirected: add both directions
    edge_index = [list(row) + list(col), list(col) + list(row)]

    return x, edge_index, pos_bug_to_index

def plot_and_save_graph(edge_index: list[list[int]], pos_bug_to_index: dict[tuple[Position, Bug], int], save_path="graph.png", figsize=(8,8), dpi=300):
    """
    Draw the graph using node labels that reflect bug color (w/b) and type.

    Parameters:
    - edge_index: [2 x E] list of edges
    - pos_bug_to_index: dict mapping (position, Bug or None) to node index
    - save_path: path to save PNG
    """
    # Build graph
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    # Reverse mapping: node index -> (pos, bug)
    index_to_key = {idx: key for key, idx in pos_bug_to_index.items()}

    # Generate labels and colors for all nodes
    labels = {}
    node_colors = []
    for idx in G.nodes():
        entry = index_to_key.get(idx)
        if entry is None:
            labels[idx] = ''
            node_colors.append("skyblue")
        else:
            pos, bug = entry
            if bug is None:
                labels[idx] = ''
                node_colors.append("skyblue")
            else:
                color_letter = 'w' if bug.color == PlayerColor.WHITE else 'b'
                bug_letter = bug.type.value
                labels[idx] = f"{color_letter}{bug_letter}"
                if color_letter == 'w':
                    node_colors.append("#FFFDD0")
                elif color_letter == 'b':
                    node_colors.append("grey")
                else:
                    node_colors.append("skyblue")

    # Layout
    pos_layout = nx.spring_layout(G)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos_layout, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos_layout, edge_color="gray")
    nx.draw_networkx_labels(G, pos_layout, labels, font_size=10)
    plt.axis('off')

    # Save
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


def save_graph(move_idx: int, pi_entry: list[tuple[Move, float]], board: Board, save_dir: str):
    """
    Salva un JSON con grafo e target per la mossa corrente.
    """
    if move_idx == 0: 
        return
    # get the adjacency list 
    x, edge_index, pos_bug_to_index = board_to_graph(board)

    if PLOTS:
        plot_and_save_graph(edge_index, pos_bug_to_index, save_path=os.path.join(save_dir, f"move_{move_idx}.png"))
    
    # get the move adjacency matrix
    N = len(x)
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
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"move_{move_idx}.json")
    with open(path, 'w') as f:
        json.dump(graph_dict, f)


def save_matrices(T_game, T_values, game, save_dir):

    game_shape = (0, *Training.INPUT_SHAPE)
    in_mats = np.empty(game_shape, dtype=np.float32)
    out_mats = np.empty(game_shape, dtype=np.float32)
    values = np.array(T_values, dtype=np.float32)
    for i, (in_mat, out_mat) in enumerate(T_game):
        try:
            in_mats = np.append(in_mats, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
            out_mats = np.append(out_mats, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
        except Exception as e:
            print(f"Index: {i}/{len(T_game)}")
    
            print("\nout_mat")
            print(out_mat)
            exit()

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


def generate_matches(source_folder: str, verbose: bool = False, want_matrices: bool = False, want_graphs: bool = True) -> None:
    engine = Engine()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = f"data/pro_matches/{ts}/"
    os.makedirs(base_dir, exist_ok=True)
    graph_dir = os.path.join(base_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    

    game = 1
    for fname in os.listdir(source_folder):

        game_dir = os.path.join(graph_dir, f"game_{game}")
        os.makedirs(game_dir, exist_ok=True)

        path_pgn = os.path.join(source_folder, fname)
        log_subheader(f"Parsing file {fname}")
        moves = parse_pgn(path_pgn)

        # ---- TESTING GAME ----
        try:
            engine.newgame(["Base+MLP"])
            for san in moves:
                engine.play(san, verbose=False)
        except Exception as e:
            log_header(f"Skipping game {game} with file {fname} due to error: {e}")
            continue

        # ---- PLAYING GAME TO EXTRACT MATRICES ----
        engine.newgame(["Base+MLP"])
        T_game, T_values = [], []
        graph_dir = os.path.join(base_dir, 'graphs')
        value = 1.0
        v_values = []

        for move_idx, san in enumerate(moves):

            if san != 'pass':
                # compute current policy and value
                val_moves = engine.board.get_valid_moves()
                pi = {m: 1.0 if m == engine.board._parse_move(san) else 0.0 for m in val_moves}
                pi_entry = [(m,p) for m, p in pi.items()]

                # save graph for this move
                if want_graphs:
                    save_graph(move_idx, pi_entry, engine.board, game_dir)

                # collect matrices
                if want_matrices:
                    mats = Training.get_matrices_from_board(engine.board, pi)
                    T_game += mats
                    T_values += [value] * len(mats)

                v_values.append(value)

            value *= -1.0

            engine.play(san, verbose=verbose)

        # final outcome
        log_subheader(f"Game {game} finished")
        outcome = engine.board.state
        final_mult = 1.0 if outcome == GameState.WHITE_WINS else -1.0 if outcome == GameState.BLACK_WINS else 0.0
        
        # adjust values and matrices
        if want_matrices:
            T_values = [v * final_mult for v in T_values]
            save_matrices(T_game, T_values, game, base_dir)
        if want_graphs:
            v_values = [v * final_mult for v in v_values]
            # for each json in game_dir, add the value
            for json_file in os.listdir(game_dir):
                if json_file.endswith('.json'):
                    json_path = os.path.join(game_dir, json_file)
                    with open(json_path, 'r') as f:
                        graph_data = json.load(f)
            #         # graph_data['v'] = v_values[int(json_file.split('_')[2].split('.')[0])]
                    graph_data['v'] = v_values[int(json_file.split('_')[1].split('.')[0])]
                    with open(json_path, 'w') as f:
                        json.dump(graph_data, f)
            

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
