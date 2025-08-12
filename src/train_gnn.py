from engineer import Engine
from ai.oracleGNN import OracleGNN
from ai.oracleRND import OracleRND
import os
from ai.log_utils import reset_log, log_header, log_subheader
from test.trainer import duel, duel_random
import re
import torch
import argparse
import shutil

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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='Enable play mode')
    args = parser.parse_args()
    
    PLAY = args.play

    #check if cuda is available and set the device accordingly on pytorch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA version: {torch.version.cuda}")
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix multiplications

        # show cuddn available
        if torch.backends.cudnn.is_available():
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner to find the best algorithm for the hardware
            torch.backends.cudnn.deterministic = True  # Ensure reproducibility
            torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN operations
        else:
            print("Using CUDA without cuDNN")
        
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

    """
    kwargs_network = {
        # Architecture options
        'conv_type': 'GIN',  # 'GIN', 'GAT', 'GCN'
        'num_layers': 10,
        # GAT specific options
        'gat_heads': 4,
        'gat_concat': True,
        # Dropout options
        'conv_dropout': 0.0,
        'mlp_dropout': 0.0,
        'final_dropout': 0.1,
        # Normalization options
        'use_batch_norm': False,
        'use_layer_norm': False,
        # Residual connections
        'use_residual': True,
        # Pooling options
        'pooling': 'mean',  # 'mean', 'max', 'add', 'concat'
        # MLP options
        'mlp_layers': 2,
        'final_mlp_layers': 2
    }
    """

    kwargs_network = {
        # Architecture options
        'conv_type': 'GIN',  # 'GIN', 'GAT', 'GCN'
        'num_layers': 2,
        # GAT specific options
        'gat_heads': 8,
        'gat_concat': True,
        # Dropout options
        'conv_dropout': 0.1,
        'mlp_dropout': 0.1,
        'final_dropout': 0.2,
        # Normalization options
        'use_batch_norm': False,
        'use_layer_norm': True,
        # Residual connections
        'use_residual': False,
        # Pooling options
        'pooling': 'add',  # 'mean', 'max', 'add', 'concat'
        # MLP options
        'mlp_layers': 2,
        'final_mlp_layers': 2
    }

    f_theta = OracleGNN(device=str(device), hidden_dim=24, **kwargs_network)  # Initialize the OracleGNN


    #f_theta.training(train_data_path= "pro_matches/GNN_Apr-3-2024/graphs/",epochs=15)  # Initial training
    f_theta.training(train_data_path= "data/",epochs=75)  # Initial training
    log_subheader("Pre-training completed")

    # Ensure 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Find all files starting with 'pretrain_' in 'models' directory
    pretrain_files = [f for f in os.listdir("models") if f.startswith("pretrain_") and f.endswith(".pt")]
    max_num = -1
    pattern = re.compile(r"pretrain_(.+)_(\d+)\.pt")
    for fname in pretrain_files:
        match = pattern.match(fname)
        if match:
            num = int(match.group(2))
            if num > max_num:
                max_num = num
    next_num = max_num + 1

    architecture = kwargs_network['conv_type']
    namefile = f"pretrain_{architecture}_{next_num}.pt"
    log_subheader(f"Saving pre-trained model as '{namefile}'")
    f_theta.save(f"models/{namefile}")

    # Create a new folder for epoch files
    epoch_folder = f"models/pretrain_{architecture}_{next_num}_epochs"
    os.makedirs(epoch_folder, exist_ok=True)
    
    # Search for files containing "epoch" and ending with ".pt" in models/
    epoch_files = [f for f in os.listdir("models") if "epoch" in f and f.endswith(".pt")]
    
    # Move epoch files to the new folder
    for epoch_file in epoch_files:
        src_path = os.path.join("models", epoch_file)
        dst_path = os.path.join(epoch_folder, epoch_file)
        shutil.move(src_path, dst_path)

    #move log file to the new folder
    log_file_src = "models/training.log"
    log_file_dst = os.path.join(epoch_folder, "training.log")
    shutil.move(log_file_src, log_file_dst)

    #save the kwargs_network as a txt file
    with open(os.path.join(epoch_folder, "kwargs_network.txt"), "w") as f:
        for key, value in kwargs_network.items():
            f.write(f"{key}: {value}\n")

    if epoch_files:
        log_subheader(f"Moved {len(epoch_files)} epoch files to '{epoch_folder}'")
        log_subheader(f"Moved log file to '{epoch_folder}'")


    if PLAY:
        f_theta_test = OracleGNN()
        # Load the file that was just saved, not the previous max
        f_theta_test.load(f"models/{namefile}")

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