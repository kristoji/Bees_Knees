import os
import numpy as np
from datetime import datetime
from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState
from engineer import Engine
from ai.oracle import Oracle
import multiprocessing as mp


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if BASE_PATH[-3:] == "src":
    BASE_PATH = BASE_PATH[:-3]
print(BASE_PATH)
os.chdir(BASE_PATH)  # Change working directory to the base path
os.makedirs("data", exist_ok=True)

# GLOBAL CONSTANTS
N_GAMES = 500
N_ROLLOUTS = 1500
VERBOSE = False  # If True, prints the board state after each move
CORES = 2       # Number of parallel processes to use


def log_header(title: str, width: int = 60, char: str = '='):
    bar = char * width
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{bar}")
    print(f"{ts} | {title.center(width - len(ts) - 3)}")
    print(f"{bar}\n")


def log_subheader(title: str, width: int = 50, char: str = '-'):
    bar = char * width
    print(f"{bar}")
    print(f"{title.center(width)}")
    print(f"{bar}")


def simulate_game(args):
    """
    Simulate a single game and save its training data.
    args: (game_id: int, timestamp: str, iteration: int)
    """
    game_id, ts, iteration = args
    engine = Engine()
    oracle = Oracle()
    mcts = MCTS(oracle=oracle, num_rollouts=N_ROLLOUTS)

    # Start new game
    engine.newgame(["Base+MLP"])
    state = engine.board
    T_game = []
    winner = None
    num_moves = 0

    # Play until end
    while not winner:
        mcts.run_simulation_from(state, debug=False)
        pi = mcts.get_moves_probs()
        T_game += Training.get_matrices_from_board(state, pi)
        action = mcts.action_selection(training=True)
        engine.play(action, verbose=VERBOSE)
        state = engine.board
        winner = state.state != GameState.IN_PROGRESS
        num_moves += 1

    # Determine result and folder names
    if state.state == GameState.DRAW:
        result_dir = "draws"
        value = 0.0
    else:
        result_dir = "wins"
        value = 1.0 if state.state == GameState.WHITE_WINS else -1.0

    length_dir = (
        "short" if num_moves < 50 else
        "long" if num_moves < 100 else
        "superlong"
    )

    # Convert game data to arrays
    in_mats = np.array([in_mat for in_mat, _ in T_game], dtype=np.float32)
    out_mats = np.array([out_mat for _, out_mat in T_game], dtype=np.float32)
    values = np.full(shape=(len(T_game),), fill_value=value, dtype=np.float32)

    # Prepare path
    save_path = os.path.join("data", ts, f"iteration_{iteration}", result_dir, length_dir)
    os.makedirs(save_path, exist_ok=True)

    # Save numpy data and board text
    np.savez_compressed(
        os.path.join(save_path, f"game_{game_id}.npz"),
        in_mats=in_mats,
        out_mats=out_mats,
        values=values
    )
    with open(os.path.join(save_path, f"board_{game_id}.txt"), "w") as f:
        f.write(str(state))

    return (result_dir, length_dir)


def generate_matches(iteration: int = 0) -> None:
    # Timestamp for this batch
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create base data directory
    os.makedirs(os.path.join("data", ts, f"iteration_{iteration}"), exist_ok=True)

    # Prepare arguments for each game
    args = [(i, ts) for i in range(N_GAMES)]

    # Parallel execution
    with mp.Pool(processes=CORES) as pool:
        for idx, (result, length) in enumerate(pool.imap_unordered(simulate_game, args), 1):
            log_header(f"Completed {idx}/{N_GAMES}: {result} ({length})")

    print("Generation complete.")


if __name__ == "__main__":
    generate_matches()
