from ai.oracleGNN import OracleGNN
import os
from ai.log_utils import reset_log, log_header, log_subheader
import torch
import pandas as pd
import numpy as np
from ai.loader import GraphDataset
from tqdm import tqdm
import datetime


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

def load_model(model_path, **kwargs_network):
    """Load a pre-trained GNN model"""
    log_header(f"Loading model from {model_path}")
    oracle = OracleGNN(hidden_dim=24, **kwargs_network)
    oracle.load(model_path)
    return oracle

def extract_embeddings(oracle, dataset_path, output_path, batch_size=64):
    """Extract embeddings from graphs and save to CSV using dataloader for batch processing"""
    log_header(f"Extracting embeddings from {dataset_path}")
    
    # Check if CUDA is available
    device = torch.device(os.environ.get("TORCH_DEVICE", "cpu"))
    log_subheader(f"Using device: {device}")
    
    # Set model to evaluation mode and move to device
    oracle.network.eval()
    oracle.network.to(device)
    
    # Load dataset - GraphDataset already handles JSON loading and processing
    dataset = GraphDataset(folder_path=dataset_path)
    dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    
    # Lists to store embeddings and labels
    all_embeddings = []
    all_labels = []
    
    # Process batches of graphs with nested progress bars
    log_subheader(f"Processing {len(dataset)} graphs in batches of {batch_size}")
    
    # Total number of steps for the outer progress bar
    total_steps = len(dataloader)
    
    # Use torch.no_grad() for inference to save memory
    with torch.no_grad():
        # Outer progress bar for overall dataset progress
        with tqdm(total=total_steps, desc="Overall progress", position=0) as pbar_outer:
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to the device
                batch = batch.to(device)
                
                # Get embeddings from the model for the entire batch
                embeddings = oracle.network.return_embedding(batch)
                
                # Convert embeddings to numpy array
                embeddings_np = embeddings.detach().cpu().numpy()
                
                # Get labels for the batch
                labels = batch.y.detach().cpu().numpy() if hasattr(batch, 'y') else [None] * len(batch.batch.unique())
                
                # Add embeddings and labels to lists
                all_embeddings.extend(embeddings_np)
                all_labels.extend(labels)
                
                # Update the outer progress bar
                pbar_outer.update(1)
                pbar_outer.set_description(f"Processing batch {batch_idx+1}/{total_steps}")
    
    # Create DataFrame
    log_subheader("Creating DataFrame with embeddings")
    embeddings_array = np.vstack(all_embeddings)
    
    # Create column names for embeddings
    embedding_cols = [f"emb_{i}" for i in range(embeddings_array.shape[1])]
    
    # Create DataFrame with embeddings and label
    df = pd.DataFrame(embeddings_array, columns=embedding_cols)
    df['label'] = all_labels
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log_header(f"Saved {len(df)} embeddings to {output_path}")
    
    return df

def main():
    reset_log()
    
    # Ask user about CUDA usage
    use_cuda = input("Use CUDA for computation? (y/n): ").lower().strip() == 'y'
    if use_cuda and torch.cuda.is_available():
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
    
    # Load the model
    oracle = load_model(model_path, **kwargs_network)

    # Set paths for data and output
    data_path = os.path.join(base_path, "data")
    output_path = os.path.join(data_path, f"embeddings_{model_choice}.csv")
    
    # Ask user for batch size
    while True:
        try:
            batch_size = int(input("Enter batch size for processing (default 64): ") or "64")
            if batch_size > 0:
                break
            else:
                print("Batch size must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    # We can use GraphDataset for all cases since it handles JSON files
    extract_embeddings(oracle, data_path, output_path, batch_size=batch_size)
    
    log_header("Embedding extraction completed successfully!")

if __name__ == "__main__":
    main()