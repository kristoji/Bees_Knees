from ai.training import Training
from engine.enums import GameState
from engineer import Engine
import numpy as np
import os
from datetime import datetime
import json
import zipfile
import gen_dataset.convert_sfg_to_pgn as convert_sfg_to_pgn

# PARAMS
VERBOSE = True              # If True, prints the board state after each move
PRO_MATCHES_FOLDER = "pro_matches/games-Apr-3-2024/pgn"
GAME_TO_PARSE = 1000

def log_header(title: str, width: int = 60, char: str = '='):
    bar = char * width
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{bar}", flush=True)
    print(f"{ts} | {title.center(width - len(ts) - 3)}", flush=True)
    print(f"{bar}\n", flush=True)

def log_subheader(title: str, width: int = 50, char: str = '-'):
    bar = char * width
    print(f"{bar}", flush=True)
    print(f"{title.center(width)}", flush=True)
    print(f"{bar}", flush=True)


def unzip_new_archives(directory: str) -> None:
    """
    if you pass "pro_matches/" as directory, it will unzip all the .zip files in that folder
    that arent already unzipped.
    """
    # Make sure the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")

    for filename in os.listdir(directory):
        if not filename.lower().endswith('.zip'):
            continue

        zip_path = os.path.join(directory, filename)
        extract_to = os.path.join(directory, os.path.splitext(filename)[0])

        # Check if already extracted: folder exists and has at least one file
        if os.path.isdir(extract_to) and os.listdir(extract_to):
            print(f"Skipping '{filename}': already extracted.")
            continue

        # Create the output folder if needed
        os.makedirs(extract_to, exist_ok=True)

        # Extract
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
                continue  # Skip empty lines and metadata

            move = line.split('.')[1].strip()
            
            moves.append(move)

    return moves

def save_matrices(T_game, T_values, game, save_dir):

    game_shape = (0, *Training.INPUT_SHAPE)
    T_0 = np.empty(shape=game_shape, dtype=np.float32)
    T_1 = np.empty(shape=game_shape, dtype=np.float32)
    T_2 = np.array(T_values, dtype=np.float32)

    for in_mat, out_mat in T_game:
        T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
        T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

    assert T_0.shape[0] == T_1.shape[0] == T_2.shape[0], "Shapes of input matrices do not match"

    # Create the subdirectories for saving
    os.makedirs(save_dir, exist_ok=True)

    log_subheader(f"Saving game {game} in {save_dir}/game_{game}.npz")
        
    # Save the training data for this game
    np.savez_compressed(
        f"{save_dir}/game_{game}.npz",
        in_mats=T_0,
        out_mats=T_1,
        values=T_2,
    )

def generate_matches(source_folder:str, verbose: bool = False, want_matrices: bool = True, want_graphs: bool = True) -> None:

    engine = Engine()

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"data/{ts}/pro_matches", exist_ok=True)

    game = 1

    # loop ont he file in folder source_folder
    for f in os.listdir(source_folder):
        if not f.endswith('.pgn'):
            continue

        engine.newgame(["Base+MLP"])
        s = engine.board

        T_game = []
        T_values = []
        pi_list = []
        values = []
        value = 1.0

        moves = parse_pgn(os.path.join(source_folder, f))

        for san in moves:

            val_moves = s.get_valid_moves()
            pi = {move: 1.0 if move == s._parse_move(san)  else 0.0 for move in val_moves}
            pi_list.append([(s.stringify_move(move), prob) for move, prob in pi.items()])

            if want_matrices:
                mats = Training.get_matrices_from_board(s, pi)
                T_game += mats
                T_values += [value] * len(mats)
            
            values.append(value)
            value *= -1.0  # Alternate value for each move

            engine.play(san, verbose=verbose)
        
        log_subheader(f"Game {game} finished")

        # value: float = 0.0 #dummy
        value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0
        T_values = [v * value for v in T_values]  # Adjust T_values based on the final game state


        v_pi_list = list(zip(pi_list, T_values))
        save_dir = f"data/{ts}/pro_matches"
        path = os.path.join(save_dir, f"game_{game}.json")
        log_subheader(f"Saving pi and values list in {path}")
        with open(path, 'w') as f:
            json.dump(v_pi_list, f)

        if want_matrices:
            save_matrices(T_game, T_values, game, save_dir)

        if want_graphs:
            save_graphs()

        log_subheader(f"Saving board state in {save_dir}/game_{game}_board.txt")

        with open(f"{save_dir}/game_{game}_board.txt", "w") as f:
            f.write(str(engine.board))

        if game >= GAME_TO_PARSE:
            log_header(f"Parsed {game} games, stopping as per GAME_TO_PARSE limit.")
            break

        game += 1

if "__main__" == __name__:

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    if BASE_PATH[-3:] == "src":
        BASE_PATH = BASE_PATH[:-3]
    print(BASE_PATH, flush=True)
    os.chdir(BASE_PATH)  # Change working directory to the base path

    source_folder = PRO_MATCHES_FOLDER

    generate_matches(source_folder=source_folder, verbose=VERBOSE)
    print("Generation complete.", flush=True)