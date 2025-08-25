import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import os
from ai.log_utils import log_header, log_subheader
import tqdm

class Encoder(nn.Module):
    """Encoder part of the autoencoder"""
    def __init__(self, input_dim, latent_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        
        # Progressive expansion: 256 -> 512 -> 1024 -> ... -> (latent)
        if hidden_dims is None:
            hidden_dims = []
            dim = input_dim * 2
            while dim < latent_dim:
                hidden_dims.append(dim)
                dim *= 2
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder part of the autoencoder"""
    def __init__(self, latent_dim, output_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        
        # Progressive contraction: (latent) -> ... -> 2048 -> 1024 -> 512 -> (output)
        if hidden_dims is None:
            hidden_dims = []
            dim = latent_dim // 2
            while dim > output_dim:
                hidden_dims.append(dim)
                dim //= 2
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


class Autoencoder(pl.LightningModule):
    """Complete autoencoder model using PyTorch Lightning"""
    def __init__(self, input_dim, latent_dim, hidden_dims=None, dropout=0.1, 
                 lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        # Set up dimensions
        if hidden_dims is None:
            # Defer to Encoder/Decoder defaults (progressive doubling/halving)
            encoder_dims = None
            decoder_dims = None
        else:
            encoder_dims = hidden_dims
            decoder_dims = hidden_dims[::-1]
        
        # Initialize networks
        self.encoder = Encoder(input_dim, latent_dim, encoder_dims, dropout)
        self.decoder = Decoder(latent_dim, input_dim, decoder_dims, dropout)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch
            
        reconstructed, _ = self(data)
        loss = self.criterion(reconstructed, data)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch
            
        reconstructed, _ = self(data)
        loss = self.criterion(reconstructed, data)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def train_step(self, batch, batch_idx):
        """Training step for a single batch"""
        self.optimizer.zero_grad()
        reconstructed, _ = self(batch)
        loss = self.criterion(reconstructed, batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_autoencoder(self, train_loader, val_loader=None, epochs=100) -> None:
        """Train the autoencoder using PyTorch Lightning"""
        log_header("Starting Autoencoder Training with PyTorch Lightning")
        
        self.train()

        # Training loop with tqdm for epochs
        for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
            epoch_losses = []
            val_losses = []
            
            # Training phase
            self.train()
            batch_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_idx, batch in enumerate(batch_pbar):
                loss = self.train_step(batch, batch_idx)
                epoch_losses.append(loss.item())
                
                # Update progress bar with current loss and running average
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{avg_loss:.6f}'
                })
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, (list, tuple)):
                            data = batch[0]
                        else:
                            data = batch
                        reconstructed, _ = self(data)
                        val_loss = self.criterion(reconstructed, data)
                        val_losses.append(val_loss.item())
            
            # Log epoch summary
            final_avg_loss = sum(epoch_losses) / len(epoch_losses)
            if val_losses:
                final_avg_val_loss = sum(val_losses) / len(val_losses)
                log_subheader(f"Epoch {epoch+1} completed - Train Loss: {final_avg_loss:.6f}, Val Loss: {final_avg_val_loss:.6f}")
            else:
                log_subheader(f"Epoch {epoch+1} completed - Average Loss: {final_avg_loss:.6f}")

        return

    def save_model(self, save_path: str) -> None:
        """Save the complete model and its components"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save complete model
        model_path = os.path.join(save_path, "autoencoder.pt")
        torch.save(self.state_dict(), model_path)
        log_subheader(f"Complete model saved to {model_path}")
        
        # Save encoder
        encoder_path = os.path.join(save_path, "encoder.pt")
        torch.save(self.encoder.state_dict(), encoder_path)
        log_subheader(f"Encoder saved to {encoder_path}")
        
        # Save decoder
        decoder_path = os.path.join(save_path, "decoder.pt")
        torch.save(self.decoder.state_dict(), decoder_path)
        log_subheader(f"Decoder saved to {decoder_path}")
        
        # Save model hyperparameters
        hparams_path = os.path.join(save_path, "ae_hparams.pt")
        torch.save(self.hparams, hparams_path)
        log_subheader(f"Hyperparameters saved to {hparams_path}")

    def load_model(self, load_path: str, device='auto') -> None:
        """Load the complete model and its components"""
        log_header("Loading Autoencoder Model")
        
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load complete model
        model_path = os.path.join(load_path, "autoencoder.pt")
        self.load_state_dict(torch.load(model_path, map_location=device))
        log_subheader(f"Complete model loaded from {model_path}")

        # Load encoder
        encoder_path = os.path.join(load_path, "encoder.pt")
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        log_subheader(f"Encoder loaded from {encoder_path}")

        # Load decoder
        decoder_path = os.path.join(load_path, "decoder.pt")
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        log_subheader(f"Decoder loaded from {decoder_path}")

        # Hyperparameters are already loaded with the model state dict
        log_subheader("Hyperparameters loaded with model state")
        
        self.to(device)
        self.eval()
        self.to(device)
        self.eval()
