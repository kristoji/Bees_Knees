import torch
from ai.loader import EmbeddingDataset
from ai.autoencoder import Autoencoder

def main():
    # Configuration
    CSV_PATH = "data/embeddings_2.csv"  # Adjust path as needed
    LATENT_DIM = 2
    MODEL_DIR = f"models\\ae_{LATENT_DIM}"
    EPOCHS = 1
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-4

    VALIDATION_SPLIT = 0.2
    
    # Device configuration
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
    
    
    # Load dataset
    print("Loading dataset...")
    dataset = EmbeddingDataset(CSV_PATH, use_labels=False)
    dataset._preload_to_gpu()

    input_dim = dataset.input_dim
    print(f"Dataset loaded: {len(dataset)} samples, {input_dim} features")

    train_loader, val_loader = dataset.get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Train samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Initialize model
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dims=None,  # Use default architecture
        dropout=0.1,
        lr=1e-4,
        weight_decay=1e-6
    )

    #move everything to the GPU
    model.to(device) 

    print(f"Model architecture:")
    print(f"Input dim: {input_dim} -> Latent dim: {LATENT_DIM}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    model.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS
    )
    print("Training completed successfully!")

    print("Saving model...")
    model.save_model(MODEL_DIR)

    print("Trying to load model...")

    new_model = Autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dims=None,  # Use default architecture
        dropout=0.1,
        lr=1e-4,
        weight_decay=1e-6
    )
    new_model.load_model(MODEL_DIR)

    print("Model loaded successfully to device: ", next(new_model.parameters()).device)

if __name__ == "__main__":
    main()