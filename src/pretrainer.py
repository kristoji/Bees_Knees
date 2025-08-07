from engineer import Engine
from ai.oracleGNN import OracleGNN
from ai.oracleRND import OracleRND
import os
from ai.log_utils import reset_log, log_header, log_subheader
from trainer import duel, duel_random
import re
import torch
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
N_DUELS = 2
N_ROLLOUTS = 1000
PERC_ALLOWED_DRAWS = 0.2    # [0, 1]
VERBOSE = True              # If True, prints the board state after each move
TIME_LIMIT = 5.0            # seconds for each MCTS simulation
# DEBUG = False



def main(): 

    reset_log()
    
    #f_theta = Oracle()
    #f_theta: Oracle = OracleNN()  # Use the neural network version of the oracle
    
    #check if cuda is available and set the device accordingly on pytorch
    if torch.cuda.is_available() and False:
        device = torch.device("cuda")
    
        # show the device name
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon Macs
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    
    #set device as an environment variable so that we can access it from everywhere
    os.environ["TORCH_DEVICE"] = str(device)

    f_theta = OracleGNN()

    log_header("STARTING PRE-TRAINING")
    
    #f_theta.training(train_data_path= "pro_matches/GNN_Apr-3-2024/graphs/",epochs=15)  # Initial training
    f_theta.training(train_data_path= "data/",epochs=20)  # Initial training
    log_subheader("Pre-training completed")

    # Ensure 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Find all files starting with 'pretrain_' in 'models' directory
    pretrain_files = [f for f in os.listdir("models") if f.startswith("pretrain_") and f.endswith(".pt")]
    max_num = -1
    pattern = re.compile(r"pretrain_(\d+)\.pt")
    for fname in pretrain_files:
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    log_subheader(f"Saving pre-trained model as 'pretrain_{next_num}.pt'")
    f_theta.save(f"models/pretrain_{next_num}.pt")

    f_theta_test = OracleGNN()
    # Load the file that was just saved, not the previous max
    f_theta_test.load(f"models/pretrain_{next_num}.pt")

    f_theta_random = OracleRND()

    # --- DUEL BETWEEN TWO ORACLES ---
    old_wins, new_wins = duel(
        new_player=f_theta_test, 
        old_player=f_theta_random, 
        games= N_DUELS,
        time_limit=TIME_LIMIT
        )
    log_header(f"Final result: OLD {old_wins} - {new_wins} NEW")


    # --- DUEL AGAINST RANDOM ---
    player_wins, rnd_wins = duel_random(
        player=f_theta_test, 
        games=N_DUELS,
        time_limit=TIME_LIMIT
        )
    log_header(f"Final result: PLAYER {player_wins} - {rnd_wins} RANDOM")


if "__main__" == __name__:
    main()
    print("Training completed.")