import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from ai.log_utils import log_header, log_subheader

class Encoder(nn.Module):
    """Encoder part of the autoencoder"""
    def __init__(self, input_dim, latent_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        
        if hidden_dims is None:
            # Default architecture with decreasing layer sizes
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final encoding layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder part of the autoencoder"""
    def __init__(self, latent_dim, output_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        
        if hidden_dims is None:
            # Default architecture with increasing layer sizes
            hidden_dims = [output_dim // 4, output_dim // 2]
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final decoding layer to original dimensions
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    """Complete autoencoder model combining encoder and decoder"""
    def __init__(self, input_dim, latent_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        
        if hidden_dims is None:
            # Generate symmetric hidden dimensions
            encoder_dims = [input_dim // 2, input_dim // 4]
            decoder_dims = [input_dim // 4, input_dim // 2]
        else:
            # Use provided hidden dimensions
            mid_point = len(hidden_dims) // 2
            encoder_dims = hidden_dims[:mid_point]
            decoder_dims = hidden_dims[mid_point:][::-1]  # Reverse for decoder
        
        self.encoder = Encoder(input_dim, latent_dim, encoder_dims, dropout)
        self.decoder = Decoder(latent_dim, input_dim, decoder_dims, dropout)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


def train_autoencoder(model, train_loader, val_loader=None, epochs=100, 
                     lr=1e-3, weight_decay=1e-5, device="cpu", 
                     early_stopping_patience=10):
    """Train the autoencoder with detailed progress tracking"""
    log_header("Starting Autoencoder Training")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = len(train_loader)
        
        # Create outer progress bar for epochs
        log_subheader(f"Epoch {epoch+1}/{epochs}")
        
        # Inner progress bar for batches within the epoch
        with tqdm(total=train_batches, desc="Training", unit="batch") as pbar:
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed, encoded = model(data)
                loss = criterion(reconstructed, data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track loss
                batch_loss = loss.item()
                train_loss += batch_loss
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"batch_loss": f"{batch_loss:.6f}"})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase if validation data is provided
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = len(val_loader)
            
            with torch.no_grad():
                with tqdm(total=val_batches, desc="Validation", unit="batch") as pbar:
                    for batch_idx, data in enumerate(val_loader):
                        data = data.to(device)
                        reconstructed, encoded = model(data)
                        loss = criterion(reconstructed, data)
                        val_loss += loss.item()
                        pbar.update(1)
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling based on validation loss
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                log_subheader(f"New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    log_subheader(f"Early stopping triggered after {epoch+1} epochs")
                    model.load_state_dict(best_model_state)
                    break
            
            log_subheader(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            log_subheader(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
    
    # Load best model if validation was used
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {"train_losses": train_losses, "val_losses": val_losses}


def save_model_components(model, output_dir):
    """Save encoder and decoder separately"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save encoder
    encoder_path = os.path.join(output_dir, "encoder.pt")
    torch.save(model.encoder.state_dict(), encoder_path)
    log_subheader(f"Encoder saved to {encoder_path}")
    
    # Save decoder
    decoder_path = os.path.join(output_dir, "decoder.pt")
    torch.save(model.decoder.state_dict(), decoder_path)
    log_subheader(f"Decoder saved to {decoder_path}")
    
    # Save full model for reference
    full_model_path = os.path.join(output_dir, "full_autoencoder.pt")
    torch.save(model.state_dict(), full_model_path)
    log_subheader(f"Full model saved to {full_model_path}")


def plot_training_curves(history, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_losses"], label="Training Loss")
    
    if history["val_losses"]:
        plt.plot(history["val_losses"], label="Validation Loss")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Autoencoder Training Progress")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path)
    log_subheader(f"Training curves saved to {plot_path}")
