import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, MessagePassing, GATConv, GCNConv
from torch.utils.data import DataLoader
from torch_geometric.utils import remove_self_loops, add_self_loops
from tqdm import tqdm
import math


class GIN_Conv(MessagePassing):
    def __init__(self, MLP, eps=0.0):
        super().__init__(aggr='add')
        self.mlp = MLP
        self.epsilon = torch.nn.Parameter(torch.tensor([eps]))

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        x = (1 + self.epsilon) * x + aggr_out
        return self.mlp(x)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        return self.propagate(edge_index, x=x)


class GraphAttentionLayer(MessagePassing):
    """Custom GAT layer with residual connections and dropout"""
    def __init__(self, in_channels, out_channels, heads=1, concat=True, 
                 dropout=0.0, add_self_loops=True, bias=True):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        if bias and concat:
            self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)
        
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        out = self.propagate(edge_index, x=x)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        if self.bias is not None:
            out += self.bias
            
        return out
    
    def message(self, x_i, x_j, edge_index_i):
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = (x * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


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
                mlp = MLP(input_dim, hidden_dim, hidden_dim, 
                                num_layers=mlp_layers, dropout=mlp_dropout,
                                use_batch_norm=use_batch_norm, 
                                use_layer_norm=use_layer_norm,
                                use_residual=use_residual and i > 0)
                conv = GIN_Conv(mlp)
                
            elif conv_type == 'GAT':
                out_dim = hidden_dim // gat_heads if gat_concat else hidden_dim
                conv = GraphAttentionLayer(input_dim, out_dim, 
                                         heads=gat_heads, concat=gat_concat,
                                         dropout=conv_dropout)
                
            elif conv_type == 'GCN':
                conv = GCNConv(input_dim, hidden_dim)
            
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


class GraphClassifierImproved(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 # Learning rate and optimization
                 lr=1e-4,
                 weight_decay=1e-6,
                 lr_scheduler=None,  # 'cosine', 'step', 'plateau'
                 # Architecture parameters (pass to EnhancedGraph_Net)
                 **model_kwargs):
        super(GraphClassifierImproved, self).__init__()
        
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        
        self.model = Graph_Net(in_dim, hidden_dim, num_classes, **model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        preds = (x > 0).float()
        data.y = data.y.float()

        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            x, edge_index, batch_idx = data.x, data.edge_index, data.batch
            x = self.model(x, edge_index, batch_idx)
            x = x.squeeze(dim=-1)
            return torch.sigmoid(x)
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))

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
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)

    def train_epoch(self, train_loader: DataLoader):
        """Custom training epoch for manual training loop"""
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Batches", leave=False):
            loss, _ = self.forward(batch, mode="train")
            
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
            tqdm.write(f'Epoch {epoch+1}/{epochs} completed. Loss: {ep_loss:.4f}')


# Example usage configurations:
"""
# Basic GIN with dropout and batch norm
model = GraphClassifier(
    in_dim=10, hidden_dim=64, num_classes=1,
    conv_type='GIN',
    conv_dropout=0.2,
    final_dropout=0.5,
    use_batch_norm=True
)

# Graph Attention Network with multiple heads
model = GraphClassifier(
    in_dim=10, hidden_dim=64, num_classes=1,
    conv_type='GAT',
    gat_heads=8,
    gat_concat=True,
    conv_dropout=0.3,
    final_dropout=0.5
)

# Residual GIN with advanced features
model = GraphClassifier(
    in_dim=10, hidden_dim=128, num_classes=1,
    conv_type='GIN',
    num_layers=5,
    use_residual=True,
    use_batch_norm=True,
    conv_dropout=0.2,
    mlp_dropout=0.1,
    final_dropout=0.5,
    pooling='concat',
    mlp_layers=3,
    final_mlp_layers=3,
    lr=5e-4,
    weight_decay=1e-4,
    lr_scheduler='cosine'
)
"""