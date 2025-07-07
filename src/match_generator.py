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
N_ROLLOUTS = 1500
VERBOSE = False              # If True, prints the board state after each move

def log_header(title: str, width: int = 60, char: str = '='):
    bar = char * width
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{bar}")
    print(f"{ts} | {title.center(width - len(ts) - 3)}")
    print(f"{bar}\n")

def log_subheader(title: str, width: int = 50, char: str = '-'):
    bar = char * width
    print(f"{bar}")
    print(f"{title.center(width)}")
    print(f"{bar}")


def generate_matches(f_theta: Oracle, iteration: int = 0, n_games: int = 500, n_rollouts: int = 1500, verbose: bool = False, perc_allowed_draws: float = float('inf')) -> None:

    engine = Engine()

    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"data/{ts}/iteration_{iteration}", exist_ok=True)


    game = 0
    draw = 0
    wins = 0
    while game < n_games:

        log_header(f"Game {game} of {n_games}: {draw}/{wins} [D/W]")

        T_game = []

        engine.newgame(["Base+MLP"])
        s = engine.board
        mcts_game = MCTS(oracle=f_theta, num_rollouts=n_rollouts)
        winner = None

        num_moves = 0

        while not winner:
            mcts_game.run_simulation_from(s, debug=False)

            pi: dict[Move, float] = mcts_game.get_moves_probs()
            T_game += Training.get_matrices_from_board(s, pi)
            
            a: str = mcts_game.action_selection(training=True)
            engine.play(a, verbose=verbose)
            winner: GameState = engine.board.state != GameState.IN_PROGRESS
            num_moves += 1

        if engine.board.state == GameState.DRAW:
            draw += 1
            if draw > perc_allowed_draws * n_games:
                continue
        else:
            wins += 1
        game += 1

        log_subheader(f"Game {game} finished with state {engine.board.state.name}")

        value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0

        game_shape = (0, *Training.INPUT_SHAPE)
        T_0 = np.empty(shape=game_shape, dtype=np.float32)
        T_1 = np.empty(shape=game_shape, dtype=np.float32)
        T_2 = np.empty(shape=(0,), dtype=np.float32)

        for in_mat, out_mat in T_game:
            T_2 = np.append(T_2, np.array(value, dtype=np.float32).reshape((1,)), axis=0)
            T_1 = np.append(T_1, np.array(out_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)
            T_0 = np.append(T_0, np.array(in_mat, dtype=np.float32).reshape((1,) + Training.INPUT_SHAPE), axis=0)

        win_or_draw = "draws" if engine.board.state == GameState.DRAW else "wins"
        short_or_long = "short" if num_moves < 50 else "long" if num_moves < 100 else "superlong"

        log_subheader(f"Saving game {game} in data/{ts}/iteration_{iteration}/{win_or_draw}/{short_or_long}/game_{game}.npz")

        # Save the training data for this game
        np.savez_compressed(
            f"data/{ts}/iteration_{iteration}/{win_or_draw}/{short_or_long}/game_{game}.npz",
            in_mats=T_0,
            out_mats=T_1,
            values=T_2,
        )

        log_subheader(f"Saving board state in data/{ts}/iteration_{iteration}/{win_or_draw}/{short_or_long}/board_{game}.txt")

        with open(f"data/{ts}/iteration_{iteration}/{win_or_draw}/{short_or_long}/board_{game}.txt", "w") as f:
            f.write(str(engine.board))

if "__main__" == __name__:

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    if BASE_PATH[-3:] == "src":
        BASE_PATH = BASE_PATH[:-3]
    print(BASE_PATH)
    os.chdir(BASE_PATH)  # Change working directory to the base path
    os.makedirs("data", exist_ok=True)

    f_theta = Oracle()  # Initialize the Oracle instance

    generate_matches(f_theta=f_theta, iteration=0, n_games=N_GAMES, n_rollouts=N_ROLLOUTS, verbose=VERBOSE)
    print("Generation complete.")