import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, GATConv, GCNConv, GINConv
from torch.utils.data import DataLoader
from tqdm import tqdm

# MR. OPUS FUTURE LAYER SUGGESTION.
# from torch_geometric.nn import SAGEConv, ChebConv, GraphConv, TransformerConv
# Removed custom GIN_Conv and GraphAttentionLayer implementations since we'll use PyG's versions

class MLP(torch.nn.Module):
    """Enhanced MLP with dropout, batch norm, and residual connections"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0, 
                 use_batch_norm=False, use_layer_norm=False, use_residual=False):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        self.use_residual = use_residual and (in_dim == out_dim)
        self.dropout = dropout
        
        layers = []
        
        # First layer
        layers.append(Linear(in_dim, hidden_dim))
        if use_batch_norm:
            layers.append(BatchNorm1d(hidden_dim))
        elif use_layer_norm:
            layers.append(LayerNorm(hidden_dim))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(LayerNorm(hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(Dropout(dropout))
        
        # Output layer
        layers.append(Linear(hidden_dim, out_dim))
        
        self.network = torch.nn.Sequential(*layers)
        
        # Residual projection if needed
        if self.use_residual and in_dim != out_dim:
            self.residual_projection = Linear(in_dim, out_dim)
        else:
            self.residual_projection = None

    def forward(self, x):
        identity = x
        out = self.network(x)
        
        if self.use_residual:
            if self.residual_projection is not None:
                identity = self.residual_projection(identity)
            out = out + identity
            
        return out


class ResidualGNNBlock(torch.nn.Module):
    """Residual block for GNN layers"""
    def __init__(self, conv_layer, hidden_dim, dropout=0.0, use_batch_norm=False):
        super(ResidualGNNBlock, self).__init__()
        self.conv = conv_layer
        self.dropout = Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.norm = BatchNorm1d(hidden_dim) if use_batch_norm else torch.nn.Identity()
        
    def forward(self, x, edge_index):
        #TODO: problema! only positive inputs, vedi slide 86 pacchetto 6 Deep Learning Prof. Silvestri, si potrebbe sistemare con un altro linear layer.
        identity = x
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection (only if dimensions match)
        if identity.size(-1) == out.size(-1):
            out = out + identity
            
        return out


class Graph_Net(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, 
                 # Architecture options
                 conv_type='GIN',  # 'GIN', 'GAT', 'GCN'
                 num_layers=10,
                 # GAT specific options
                 gat_heads=4,
                 gat_concat=True,
                 # GIN specific options
                 gin_eps=0.0,
                 gin_train_eps=False,
                 # Dropout options
                 conv_dropout=0.0,
                 mlp_dropout=0.0,
                 final_dropout=0.1,
                 # Normalization options
                 use_batch_norm=False,
                 use_layer_norm=False,
                 # Residual connections
                 use_residual=True,
                 # Pooling options
                 pooling='mean',  # 'mean', 'max', 'add', 'concat'
                 # MLP options
                 mlp_layers=2,
                 final_mlp_layers=2):
        
        super(Graph_Net, self).__init__()
        print("Using architecture:", conv_type)
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.pooling = pooling
        
        # Build convolutional layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            
            if conv_type == 'GIN':
                # Using PyG's GINConv with MLP as the neural network
                mlp = MLP(input_dim, hidden_dim, hidden_dim, 
                         num_layers=mlp_layers, dropout=mlp_dropout,
                         use_batch_norm=use_batch_norm, 
                         use_layer_norm=use_layer_norm,
                         use_residual=use_residual and i > 0)
                # GINConv from PyG takes a neural network as input
                conv = GINConv(mlp, eps=gin_eps, train_eps=gin_train_eps)
                
            elif conv_type == 'GAT':
                # Using PyG's GATConv
                # Note: GATConv returns (N, heads * out_channels) if concat=True
                #       or (N, out_channels) if concat=False
                out_channels = hidden_dim // gat_heads if gat_concat else hidden_dim
                conv = GATConv(input_dim, out_channels, 
                              heads=gat_heads, 
                              concat=gat_concat,
                              dropout=conv_dropout,
                              add_self_loops=True,
                              bias=True)
                
            elif conv_type == 'GCN':
                # Using PyG's GCNConv
                conv = GCNConv(input_dim, hidden_dim, 
                              add_self_loops=True, 
                              normalize=True,
                              bias=True)
            
            if use_residual:
                conv = ResidualGNNBlock(conv, hidden_dim, conv_dropout, use_batch_norm)
                self.convs.append(conv)
            else:
                self.convs.append(conv)
                if use_batch_norm:
                    self.norms.append(BatchNorm1d(hidden_dim))
                elif use_layer_norm:
                    self.norms.append(LayerNorm(hidden_dim))
                else:
                    self.norms.append(torch.nn.Identity())
                self.dropouts.append(Dropout(conv_dropout) if conv_dropout > 0 else torch.nn.Identity())
        
        # Pooling layer
        if pooling == 'concat':
            pooled_dim = hidden_dim * 3  # mean + max + add
        else:
            pooled_dim = hidden_dim
            
        # Final classification layers
        if final_mlp_layers > 1:
            self.classifier = MLP(pooled_dim, hidden_dim, num_classes,
                                num_layers=final_mlp_layers, 
                                dropout=final_dropout,
                                use_batch_norm=use_batch_norm,
                                use_layer_norm=use_layer_norm)
        else:
            layers = []
            if final_dropout > 0:
                layers.append(Dropout(final_dropout))
            layers.append(Linear(pooled_dim, num_classes))
            self.classifier = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index, batch):
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            if self.use_residual:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropouts[i](x)

        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'concat':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_add = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_add], dim=1)

        # Classification
        x = self.classifier(x)
        return x
    
    def return_embedding(self, x, edge_index, batch):
        """
        Return the embedding of the graph without classification.
        Useful for downstream tasks or visualization.
        """
        for i, conv in enumerate(self.convs):
            if self.use_residual:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropouts[i](x)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'concat':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_add = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_add], dim=1)

        return x


class GraphClassifier(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 # Learning rate and optimization
                 lr=1e-4,
                 weight_decay=1e-6,
                 lr_scheduler=None,  # 'cosine', 'step', 'plateau'
                 # Architecture parameters (pass to Graph_Net)
                 **model_kwargs):
        super(GraphClassifier, self).__init__()
        
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.conv_type = model_kwargs.get('conv_type', 'UNKNOWN')
        self.model = Graph_Net(in_dim, hidden_dim, num_classes, **model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        logits = self.model(x, edge_index, batch_idx)
        logits = logits.squeeze(dim=-1)
        
        if mode=='train' or (hasattr(data, 'y') and data.y is not None):
            data.y = data.y.float()
            loss = self.loss_module(logits, data.y)
            preds = (logits > 0).float()
            acc = (preds == data.y).sum().float() / preds.shape[0]
            return logits, loss, acc
        else:
            return logits
    
    def predict(self, data, use_sigmoid=False):
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(data, mode="predict")
            if isinstance(logits, tuple):
                logits = logits[0]  # Extract logits if tuple returned

            # rimuoviamo sigmoide così che i valori unbounded vanno diretti alla softmax.
            results = logits if not use_sigmoid else torch.sigmoid(logits)
            return results

    def return_embedding(self, data):
        self.model.eval()
        with torch.no_grad():
            x, edge_index, batch_idx = data.x, data.edge_index, data.batch
            embeddings = self.model.return_embedding(x, edge_index, batch_idx)
            
            # If logits is a tuple, extract the first element
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            return embeddings

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.lr_scheduler is None:
            return optimizer
        
        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}
        else:
            return optimizer
            
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        logits, loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, acc = self.forward(batch, mode="val")
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        logits, _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)

    def train_epoch(self, train_loader: DataLoader):
        """Custom training epoch for manual training loop"""
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Batches", leave=False):
            # Move batch to the correct device
            batch = batch.to(self.device)
            
            _, loss, _ = self.forward(batch, mode="train")
            
            # Manual optimization
            optimizer = self.configure_optimizers()
            if isinstance(optimizer, dict):
                optimizer = optimizer['optimizer']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def train_network(self, train_loader: DataLoader, epochs: int = 10):
        """Custom training loop"""
        optimizer = self.configure_optimizers()
        if isinstance(optimizer, dict):
            optimizer = optimizer['optimizer']
            
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            ep_loss = self.train_epoch(train_loader)

            if (epoch+1) % 5 == 0 and epoch != 0:
                self.save(f"models/{self.conv_type}_epoch_{epoch+1}.pt")

            text_tqdm = f'Epoch {epoch+1}/{epochs} completed. Loss: {ep_loss:.4f}'

            #append this to a log file called "training.log" in models/
            with open("models/training.log", "a") as log_file:
                log_file.write(text_tqdm + "\n")

            tqdm.write(text_tqdm)
