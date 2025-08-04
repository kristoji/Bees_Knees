from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState
from engineer import Engine
from engine.game import Move
from ai.oracle import Oracle
import numpy as np
import os
from datetime import datetime

# PARAMS
N_GAMES = 500
N_ROLLOUTS = 1000
VERBOSE = False              # If True, prints the board state after each move
SHORT = 50
LONG = 100
SUPERLONG = 150
PRO_MATCHES_FOLDER = "pro_matches/games-Apr-3-2024/pgn"

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

def generate_matches_diff_theta(f_theta_1: Oracle, f_theta_2: Oracle, iteration: int = 0, n_games: int = 500, n_rollouts: int = 1500, verbose: bool = False, perc_allowed_draws: float = float('inf')) -> None:

    engine = Engine()

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"data/{ts}/iteration_{iteration}", exist_ok=True)


    game = 0
    draw = 0
    wins = 0
    discarded = 0

    while game < n_games:

        log_header(f"Game {game} of {n_games}: {draw}/{wins} [D/W] - Discarded: {discarded}", width=70)

        T_game = []

        engine.newgame(["Base+MLP"])
        s = engine.board

        mcts_game_1 = MCTS(oracle=f_theta_1, num_rollouts=n_rollouts)
        mcts_game_2 = MCTS(oracle=f_theta_2, num_rollouts=n_rollouts)

        white_player = mcts_game_1 if game % 2 == 0 else mcts_game_2
        black_player = mcts_game_2 if game % 2 == 0 else mcts_game_1

        winner = None

        num_moves = 0

        while not winner and num_moves < SUPERLONG:

            mcts_game = white_player if num_moves % 2 == 0 else black_player

            mcts_game.run_simulation_from(s, debug=False)

            pi: dict[Move, float] = mcts_game.get_moves_probs()

            #T_game += Training.get_matrices_from_board(s, pi)

            a: str = mcts_game.action_selection(training=True)

            engine.play(a, verbose=verbose)

            winner: GameState = engine.board.state != GameState.IN_PROGRESS

            num_moves += 1
            
            if num_moves%10==0 and not verbose: 
                print(f"  Move {num_moves}", flush=True)

        if num_moves >= SUPERLONG: # to avoid killing process for cache overflow
            log_subheader(f"Game {game} exceeded maximum moves ({SUPERLONG}). Ending game early.")
            discarded += 1
            continue
        elif engine.board.state == GameState.DRAW:
            draw += 1
            if draw > perc_allowed_draws * n_games:
                continue
        else:
            wins += 1
        game += 1

        log_subheader(f"Game {game} finished with state {engine.board.state.name}")

        value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0
        
        # -------------- 1st VERSION  --------------
        game_shape = (0, *Training.INPUT_SHAPE)
        T_0 = np.empty(shape=game_shape, dtype=np.float32)
        T_1 = np.empty(shape=game_shape, dtype=np.float32)
        T_2 = np.empty(shape=(0,), dtype=np.float32)

        for in_mat, out_mat in T_game:
            T_2 = np.append(T_2, np.array(value, dtype=np.float32).reshape((1,)), axis=0)
            T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
            T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

        # -------------- 2nd VERSION --------------
        # T_0_list, T_1_list, T_2_list = [], [], []

        # for in_mat, out_mat in T_game:
        #     T_0_list.append(np.array(in_mat, dtype=np.float32))
        #     T_1_list.append(np.array(out_mat, dtype=np.float32))
        #     T_2_list.append(np.float32(value))  # singolo float, non array

        # T_0 = np.stack(T_0_list)
        # T_1 = np.stack(T_1_list)
        # T_2 = np.array(T_2_list, dtype=np.float32)


        win_or_draw = "draws" if engine.board.state == GameState.DRAW else "wins"
        short_or_long = "short" if num_moves < SHORT else "long" if num_moves < LONG else "superlong"

        # Create the subdirectories for saving
        save_dir = f"data/{ts}/iteration_{iteration}/{win_or_draw}/{short_or_long}"
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

def generate_matches(f_theta: Oracle, iteration: int = 0, n_games: int = 500, n_rollouts: int = 1500, verbose: bool = False, perc_allowed_draws: float = float('inf')) -> None:

    engine = Engine()

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"data/{ts}/iteration_{iteration}", exist_ok=True)


    game = 1
    draw = 0
    wins = 0
    discarded = 0

    while game < n_games:

        log_header(f"Game {game} of {n_games}: {draw}/{wins} [D/W] - Discarded: {discarded}", width=70)

        T_game = []
        v_values = []

        engine.newgame(["Base+MLP"])
        s = engine.board
        mcts_game = MCTS(oracle=f_theta, num_rollouts=n_rollouts)
        winner = None

        num_moves = 0
        value = 1.0

        while not winner and num_moves < SUPERLONG:

            mcts_game.run_simulation_from(s, debug=False)

            pi: dict[Move, float] = mcts_game.get_moves_probs()

            mats = Training.get_matrices_from_board(s, pi)
            v_values += [value]*len(mats)
            T_game += mats

            a: str = mcts_game.action_selection(training=True)

            engine.play(a, verbose=verbose)

            winner: GameState = engine.board.state != GameState.IN_PROGRESS

            num_moves += 1

        if num_moves >= SUPERLONG: # to avoid killing process for cache overflow
            log_subheader(f"Game {game} exceeded maximum moves ({SUPERLONG}). Ending game early.")
            discarded += 1
            continue
        elif engine.board.state == GameState.DRAW:
            draw += 1
            if draw > perc_allowed_draws * n_games:
                continue
        else:
            wins += 1

        log_subheader(f"Game {game} finished with state {engine.board.state.name}")

        final_value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0

        # -------------- 1st VERSION  --------------
        game_shape = (0, *Training.INPUT_SHAPE)
        T_0 = np.empty(shape=game_shape, dtype=np.float32)
        T_1 = np.empty(shape=game_shape, dtype=np.float32)
        T_2 = np.array(v_values, dtype=np.float32)
        T_2 = T_2 * final_value # update the values using final_value


        for in_mat, out_mat in T_game:
            T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
            T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

        win_or_draw = "draws" if engine.board.state == GameState.DRAW else "wins"
        short_or_long = "short" if num_moves < SHORT else "long" if num_moves < LONG else "superlong"

        # Create the subdirectories for saving
        save_dir = f"data/{ts}/iteration_{iteration}/{win_or_draw}/{short_or_long}"
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

        game += 1

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

def collect_matches(source_folder: str, ts: str = "pro_matches", iteration: int = 0, verbose: bool = False) -> None:

    engine = Engine()

    #ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #base_dir = f"data/pro_matches/{ts}/iteration_{iteration}"
    #os.makedirs(base_dir, exist_ok=True)

    game = 1
    for fname in os.listdir(source_folder):

        try:

            log_header(f"Game {game}", width=70)

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
            T_game = []
            v_values = []

            s = engine.board
            value = 1.0

            for san in moves:

                if san != 'pass':

                    val_moves = engine.board.get_valid_moves()
                    pi = {m: 1.0 if m == engine.board._parse_move(san) else 0.0 for m in val_moves}

                    mats = Training.get_matrices_from_board(s, pi)
                    v_values += [value]*len(mats)
                    T_game += mats

                value *= -1.0

                engine.play(san, verbose=verbose)

            log_subheader(f"Game {game} finished with state {engine.board.state.name}")

            final_value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0

            # -------------- 1st VERSION  --------------
            game_shape = (0, *Training.INPUT_SHAPE)
            T_0 = np.empty(shape=game_shape, dtype=np.float32)
            T_1 = np.empty(shape=game_shape, dtype=np.float32)
            T_2 = np.array(v_values, dtype=np.float32)
            T_2 = T_2 * final_value # update the values using final_value

            for in_mat, out_mat in T_game:
                T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
                T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

            win_or_draw = "draws" if engine.board.state == GameState.DRAW else "wins"
            short_or_long = "short" if len(moves) < SHORT else "long" if len(moves) < LONG else "superlong"

            # Create the subdirectories for saving
            save_dir = f"data/{ts}/iteration_{iteration}/{win_or_draw}/{short_or_long}"
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
        
        except Exception as e: # added beacuse of some weird games with high stack of pieces
            log_header(f"Skipping game {game} with file {fname} due to error {e}")
            continue

        game += 1
        

if "__main__" == __name__:

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    if BASE_PATH[-3:] == "src":
        BASE_PATH = BASE_PATH[:-3]
    print(BASE_PATH, flush=True)
    os.chdir(BASE_PATH)  # Change working directory to the base path
    os.makedirs("data", exist_ok=True)

    # f_theta = Oracle()  # Initialize the Oracle instance
    # generate_matches(f_theta=f_theta, iteration=0, n_games=N_GAMES, n_rollouts=N_ROLLOUTS, verbose=VERBOSE)
    
    collect_matches(source_folder=PRO_MATCHES_FOLDER, verbose=VERBOSE)

    
    print("Generation complete.", flush=True)