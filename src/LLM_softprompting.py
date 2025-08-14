import sys
import torch
from ai.oracleGNN import OracleGNN
import os
from ai.log_utils import reset_log, log_header, log_subheader
import datetime
from ai.autoencoder import Autoencoder

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

LATENT_DIM = 4096 #per LLama 3B, per DeepSeek Distill Qwen è 3584

if __name__ == "__main__":
    # Ask user about CUDA usage
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
    


        # Set device as an environment variable
    os.environ["TORCH_DEVICE"] = str(device)
    
    # Set pathsclea
    base_path = os.path.dirname(os.path.abspath(__file__))
    if base_path.endswith("src"):
        base_path = base_path[:-3]
    
    # Get list of available models
    models_dir = os.path.join(base_path, "models")
    pretrain_files = [f for f in os.listdir(models_dir) if f.startswith("pretrain_") and f.endswith(".pt")]
    if not pretrain_files:
        raise ValueError("No pretrained models found in models/ directory")
    
    # Display available models
    print("\nAvailable models:")
    for i, model_file in enumerate(pretrain_files):
        model_path = os.path.join(models_dir, model_file)
        modified_time = os.path.getmtime(model_path)
        # Convert timestamp to readable format
        readable_time = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{i}] {model_file} (modified: {readable_time})")
    
    # Ask user which model to load
    while True:
        model_choice = input("\nEnter the number of the model to load (or 'latest' for most recent): ").strip()
        if model_choice.lower() == 'latest':
            # Sort by modification time to get the latest
            latest_model = sorted(pretrain_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))[-1]
            model_path = os.path.join(models_dir, latest_model)
            print(f"Selected latest model: {latest_model}")
            break
        else:
            try:
                idx = int(model_choice)
                if 0 <= idx < len(pretrain_files):
                    model_path = os.path.join(models_dir, pretrain_files[idx])
                    print(f"Selected model: {pretrain_files[idx]}")
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(pretrain_files) - 1}.")
            except ValueError:
                print("Invalid input. Please enter a number or 'latest'.")
    
    # Load the GNN model
    oracle = OracleGNN(hidden_dim=24, **kwargs_network)
    oracle.load(model_path)

    # load the autoencoder model for LLama
    autoencoder = Autoencoder(
        input_dim=24,
        latent_dim=LATENT_DIM,
        hidden_dims=[32, 64, 128, 256, 512, 1024, 2048], 
        dropout=0.1,
        lr=1e-4,
        weight_decay=1e-6
    )
    autoencoder.load_model(os.path.join(models_dir, f"ae_{LATENT_DIM}"), device=str(device))


    """
    TODO LIST:
    - Implementare softprompting, dove si prende l'embedding dalla GNN con oracle.network.return_embedding con input il grafo di PyG
    - l'embedding viene passato all'autoencoder, in particolare all'encoder facendo autoencoder.encode, opzionalmente possiamo calcolare l'errore di conversione con .decode e MSELoss
    - il risultato dell'encoder è un embedding adatto a LLama 3.2 3B IT che possiamo caricare con huggingface e bitsandbytes quantizzato volendo.
    - Implementare il caricamento dell'embedding nel modello LLama
    - Implementare il meccanismo di soft prompting
    - Dobbiamo dare in pasto a LLama le istruzioni di Hive, considerando che l'embedding è la board corrente, e ottenere in uscita la lista delle prossime future mosse.
    - Stampare tutto in modo che si capisca.
    """
    sys.exit(0)