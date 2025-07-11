from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState, PlayerColor
from engineer import Engine
from engine.game import Move
from ai.oracle import Oracle
import numpy as np
import re
import os
from datetime import datetime

# PARAMS
VERBOSE = True              # If True, prints the board state after each move
# PRO_MATCHES_FOLDER = "pro_matches/pgn_2011"
PRO_MATCHES_FOLDER = "pro_matches/"
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


def parse_hive_game(file_path: str) -> list[str]:
    moves = []
    move_line_pattern = re.compile(r'^\d+\.\s+(.*)$')

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['):
                continue  # Skip empty lines and metadata

            move = line.split('.')[1].strip()
            
            moves.append(move)

    return moves


def generate_matches(source_folder:str, verbose: bool = False) -> None:

    engine = Engine()

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"data/{ts}/pro_matches", exist_ok=True)

    game = 1

    # loop ont he file in folder source_folder
    for f in os.listdir(source_folder):

        engine.newgame(["Base+MLP"])
        s = engine.board

        T_game = []

        moves = parse_hive_game(os.path.join(source_folder, f))

        # for m in moves:
        #     print(m)

        for m in moves:

            pi = {move: 1.0 if move == s._parse_move(m)  else 0.0 for move in s.get_valid_moves()}

            T_game += Training.get_matrices_from_board(s, pi)
            
            engine.play(m, verbose=verbose)
        
        log_subheader(f"Game {game} finished")

        # value: float = 0.0 #dummy
        value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0
        

        exit()  # For debugging purposes, remove this line in production

        # -------------- 1st VERSION  --------------
        game_shape = (0, *Training.INPUT_SHAPE)
        T_0 = np.empty(shape=game_shape, dtype=np.float32)
        T_1 = np.empty(shape=game_shape, dtype=np.float32)
        T_2 = np.empty(shape=(0,), dtype=np.float32)

        for in_mat, out_mat in T_game:
            T_2 = np.append(T_2, np.array(value, dtype=np.float32).reshape((1,)), axis=0)
            T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
            T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

        # Create the subdirectories for saving
        save_dir = f"data/{ts}/pro_matches"
        os.makedirs(save_dir, exist_ok=True)

        log_subheader(f"Saving game {game} in {save_dir}/game_{game}.npz")

        # Save the training data for this game
        np.savez_compressed(
            f"{save_dir}/game_{game}.npz",
            in_mats=T_0,
            out_mats=T_1,
            values=T_2,
        )

        log_subheader(f"Saving board state in {save_dir}/board_{game}.txt")

        with open(f"{save_dir}/board_{game}.txt", "w") as f:
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
    os.makedirs("data", exist_ok=True)

    source_folder = PRO_MATCHES_FOLDER

    generate_matches(source_folder=source_folder, verbose=VERBOSE)
    print("Generation complete.", flush=True)