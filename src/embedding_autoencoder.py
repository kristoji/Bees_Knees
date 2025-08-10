import torch
import os
from torch.utils.data import DataLoader, random_split
from ai.embedding.dataset import EmbeddingDataset
from ai.embedding.autoencoder import Autoencoder, train_autoencoder, save_model_components, plot_training_curves

def main():
    # Configuration
    CSV_PATH = "data/embeddings.csv"  # Adjust path as needed
    MODEL_DIR = "models/ae"
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    LATENT_DIM = 64
    VALIDATION_SPLIT = 0.2
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = EmbeddingDataset(CSV_PATH, use_labels=False)
    input_dim = dataset.input_dim
    print(f"Dataset loaded: {len(dataset)} samples, {input_dim} features")
    
    # Split dataset into train/validation
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dims=None,  # Use default architecture
        dropout=0.1
    )
    
    print(f"Model architecture:")
    print(f"Input dim: {input_dim} -> Latent dim: {LATENT_DIM}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the autoencoder
    trained_model, history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        early_stopping_patience=15
    )
    
    # Create output directory and save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save encoder and decoder separately
    encoder_path = os.path.join(MODEL_DIR, "encoder.pt")
    decoder_path = os.path.join(MODEL_DIR, "decoder.pt")
    
    torch.save(trained_model.encoder.state_dict(), encoder_path)
    torch.save(trained_model.decoder.state_dict(), decoder_path)
    
    print(f"Encoder saved to: {encoder_path}")
    print(f"Decoder saved to: {decoder_path}")
    
    # Save training curves
    plot_training_curves(history, MODEL_DIR)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()