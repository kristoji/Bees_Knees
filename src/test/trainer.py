
import os
from engineer import Engine
from datetime import datetime
from test.duel import duel, duel_random

from ai.oracle import Oracle
from ai.oracleGNN import OracleGNN
from ai.log_utils import log_header, reset_log

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if BASE_PATH[-3:] == "src":
    BASE_PATH = BASE_PATH[:-3]
print(BASE_PATH)
os.chdir(BASE_PATH)  # Change working directory to the base path
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
print(f"Working directory set to: {os.getcwd()}")

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

def niam():
    # duel_random(player = Oracle(), games = 2, time_limit= 5, verbose = True, draw_limit=100)
    # cross_platform_duel(
    #     exe_player_path= os.path.join(BASE_PATH, "models/nokamute"),
    #     oracle_player= Oracle(),
    #     is_exe_white=True,
    #     games=2,
    #     time_limit=5,
    #     verbose=True,
    #     draw_limit=100
    # )
    player = OracleGNN()
    player.load("models/pretrain_GAT_3.pt")
    duel_random(
        player=player,
        games=2,    
        time_limit=5,#float("inf"),  # Set to infinity to let mcts do 1k rollouts
        verbose=True,
        draw_limit=100
    )



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
        old_wins, new_wins = duel(
            f_theta_new, 
            f_theta, 
            games=N_DUELS, 
            time_limit=TIME_LIMIT
            )
        
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
    # main()
    niam()
    print("Training completed.")