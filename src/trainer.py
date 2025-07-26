# from ai.network import NeuralNetwork
from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState
from engineer import Engine
from engine.game import Move
from ai.oracle import Oracle
from ai.oracleNN import OracleNN
import numpy as np
import os
from datetime import datetime
import time
from gen_dataset.match_generator import generate_matches
import re

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if BASE_PATH[-3:] == "src":
    BASE_PATH = BASE_PATH[:-3]
print(BASE_PATH)
os.chdir(BASE_PATH)  # Change working directory to the base path
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# GLOBALS
ENGINE = Engine()

# PARAMS
N_ITERATIONS = 3
N_GAMES = 50
N_DUELS = 10
N_ROLLOUTS = 1000
PERC_ALLOWED_DRAWS = 0.2    # [0, 1]
VERBOSE = True              # If True, prints the board state after each move
TIME_LIMIT = 5.0            # seconds for each MCTS simulation
# DEBUG = False


def duel(new_player: Oracle, old_player: Oracle, games: int = 10) -> tuple[float, float]:
    """
    Duel between two players using MCTS to determine which player is stronger.
    Returns the number of wins for each player: old_player and new_player.
    """
    old_wins = 0
    new_wins = 0

    for game in range(games):

        log_subheader(f"Duel Game {game + 1} of {games}: OLD {old_wins} - {new_wins} NEW")

        ENGINE.newgame(["Base+MLP"])
        s = ENGINE.board
        winner = None

        mcts_game_old = MCTS(oracle=old_player, time_limit=TIME_LIMIT)
        mcts_game_new = MCTS(oracle=new_player, time_limit=TIME_LIMIT)

        white_player = mcts_game_old if game % 2 == 0 else mcts_game_new
        black_player = mcts_game_new if game % 2 == 0 else mcts_game_old


        while not winner:

            white_player.run_simulation_from(s, debug=False)
            a: str = white_player.action_selection(training=False)
            ENGINE.play(a, verbose=VERBOSE)

            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS
            if winner:
                break

            black_player.run_simulation_from(s, debug=False)
            a: str = black_player.action_selection(training=False)
            ENGINE.play(a, verbose=VERBOSE)
    
            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS


        if ENGINE.board.state == GameState.WHITE_WINS:
            old_wins += 1 if game % 2 == 0 else 0
            new_wins += 1 if game % 2 == 1 else 0
        elif ENGINE.board.state == GameState.BLACK_WINS:
            old_wins += 1 if game % 2 == 1 else 0
            new_wins += 1 if game % 2 == 0 else 0
        else:
            old_wins += 0.5
            new_wins += 0.5

    return old_wins, new_wins

def reset_log(string: str = ""):
    return
    with open("test/log.txt", "w") as f:
        f.write(string)

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


def main(): 

    reset_log()
    
    #f_theta = Oracle()
    f_theta: Oracle = OracleNN()  # Use the neural network version of the oracle

    log_header("STARTING PRE-TRAINING")
    
    f_theta.training(ts="pro_matches", iteration=0)  # Initial training

    log_subheader("Pre-training completed")

    # Ensure 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Find all files starting with 'pretrain_' in 'models' directory
    pretrain_files = [f for f in os.listdir("models") if f.startswith("pretrain_") and f.endswith(".npz")]
    max_num = -1
    pattern = re.compile(r"pretrain_(\d+)\.pt")
    for fname in pretrain_files:
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    f_theta.save(f"models/pretrain_{next_num}.pt")

    exit()

    cons_unsuccess = 0

    for iteration in range(N_ITERATIONS):
        ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #ts = "2025-07-06_22-36-22"
        os.makedirs(f"data/{ts}/iteration_{iteration}", exist_ok=True)

        log_header(f"STARTING ITERATION {iteration}")

        generate_matches(   
                            f_theta=f_theta, 
                            iteration=iteration, 
                            n_games=N_GAMES, 
                            n_rollouts=N_ROLLOUTS, 
                            verbose=VERBOSE, 
                            perc_allowed_draws=PERC_ALLOWED_DRAWS)


        if iteration == 0:
            f_theta_new: Oracle = OracleNN()
        else:
            f_theta_new: Oracle = f_theta.copy()

        # f_theta_new.training((Ttot_0, Ttot_1, Ttot_2))
        f_theta_new.training(ts=ts, iteration=iteration)

        log_header("STARTING DUEL")

        old_wins, new_wins = duel(f_theta_new, f_theta, games=N_DUELS)
        
        if old_wins < new_wins:
            f_theta = f_theta_new.copy()
            f_theta.save(f"models/{iteration}.pt")
            cons_unsuccess = 0
        else:
            cons_unsuccess += 1

        if cons_unsuccess >= 3:
            print(f"Stopping training after {iteration} iterations due to no improvement.")
            break
        

if "__main__" == __name__:
    main()
    print("Training completed.")