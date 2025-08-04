# from ai.network import NeuralNetwork
from ai.brains import MCTS
from ai.training import Training
from engine.enums import GameState
from engineer import Engine
from engine.game import Move
from ai.oracle import Oracle
from ai.oracleNN import OracleNN
from ai.oracleGNN import OracleGNN
import numpy as np
import os
from datetime import datetime
import time
from gen_dataset.match_generator import generate_matches
import re
from ai.log_utils import log_header, log_subheader, reset_log

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
DRAW_LIMIT = 100            # trials after the match ends in a draw
PRETRAIN = True
PRETRAIN_PATH = "models/pretrain_0.pt"
CONSECUTIVE_UNSUCCESSES = 3   # Maximum number of iteration with no imrovement

TRAINER_MODE = 2            # 1 for CNN, 2 for GNN
GNN_PATH = "data/"
CNN_PATH = f"data/TIMESTAMP/iteration_NUMBEROFITERATION"
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

        i = 0
        while not winner:

            white_player.run_simulation_from(s, debug=False)
            a: str = white_player.action_selection(training=False)
            ENGINE.play(a, verbose=VERBOSE)
            i+= 1

            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS

            if winner:
                break

            if i == DRAW_LIMIT:
                winner: GameState = GameState.DRAW
                break

            black_player.run_simulation_from(s, debug=False)
            a: str = black_player.action_selection(training=False)
            ENGINE.play(a, verbose=VERBOSE)
            i+= 1
    
            winner: GameState = ENGINE.board.state != GameState.IN_PROGRESS

            if winner:
                break

            if i == DRAW_LIMIT:
                winner: GameState = GameState.DRAW
                break


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


    bar = char * width
    print(f"{bar}")
    print(f"{title.center(width)}")
    print(f"{bar}")

def main(): 

    reset_log()
    
    # ---------------------------- ORACLE CREATION ----------------------------
    if TRAINER_MODE == 1:
        f_theta = OracleNN()
    elif TRAINER_MODE == 2:
        f_theta: Oracle = OracleGNN()
    else:
        log_header(title="WRONG TRAINER MODE (1:CNN - 2:GNN)")

    # ---------------------------- PRETRAINED MODEL LOADING ----------------------------
    if PRETRAIN:
        f_theta.load(PRETRAIN_PATH)

    # ---------------------------- ITERATIONS LOOP ----------------------------
    cons_unsuccess = 0 # counter for consecutive wins

    for iteration in range(N_ITERATIONS): # at the end of every self-play iteration the training occurs

        ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # time stamp
        iteration_folder = f"data/{ts}/iteration_{iteration}" # folder where to save matches of the current iteration
        os.makedirs(iteration_folder, exist_ok=True)

        log_header(f"STARTING ITERATION {iteration}")

        # ------ MATCHES GENERATION - SELF PLAY ------
        # generate_matches(   
        #                     f_theta=f_theta, 
        #                     iteration=iteration, 
        #                     n_games=N_GAMES,
        #                     n_rollouts=N_ROLLOUTS,
        #                     verbose=VERBOSE,
        #                     perc_allowed_draws=PERC_ALLOWED_DRAWS)
        f_theta.generate_matches(iteration_folder=iteration_folder) #---------> TODO : generation from the oracle function

        # ------ COPYNG THE AGENT ------
        f_theta_new: Oracle = f_theta.copy()

        # ------ TRAIMING THE COPY AS THE NEW AGENT ------
        f_theta_new.training(train_data_path=iteration_folder, epochs= 15)

        # ------ COMPARING OLD/NEW VERSION ------
        log_header("STARTING DUEL")
        old_wins, new_wins = duel(f_theta_new, f_theta, games=N_DUELS)
        
        # ------ SAVING NEW BETTER VERSION ------
        if old_wins < new_wins:
            f_theta = f_theta_new.copy()
            f_theta.save(f"models/{iteration}.pt")
            cons_unsuccess = 0
        else:
            cons_unsuccess += 1

        if cons_unsuccess >= CONSECUTIVE_UNSUCCESSES:
            print(f"Stopping training after {iteration} iterations due to no improvement.")
            break
        

if "__main__" == __name__:
    main()
    print("Training completed.")