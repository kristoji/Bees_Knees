import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch.utils.data import DataLoader
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm


class GIN_Conv(MessagePassing):
    def __init__(self, MLP, eps = 0.0):
        super().__init__(aggr='add')  # Aggregation function over the messages.
        self.mlp = MLP
        self.epsilon = torch.nn.Parameter(torch.tensor([eps]))

    def message(self, x_j):
      return x_j

    def update(self,aggr_out,x):
      x = (1+self.epsilon) * x + aggr_out
      return self.mlp(x)

    def forward(self, x, edge_index):
      # Step 1: remove self-loops to the adjacency matrix.
      edge_index, _ = remove_self_loops(edge_index)
      # Step 2: Start propagating messages.
      return self.propagate(edge_index, x=x)



class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.first_fc = Linear(in_dim, hidden_dim)
        self.second_fc = Linear(hidden_dim, out_dim)
        self.activation = torch.nn.ReLU()

        # You could use torch.nn.Sequential

    def forward(self, x):
        x = self.activation(self.first_fc(x))
        x = self.activation(self.second_fc(x))

        return x

class Graph_Net(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(Graph_Net, self).__init__()
        self.mlp_input =  MLP(in_dim, hidden_dim, hidden_dim)
        self.mlp_hidden =  MLP(hidden_dim, hidden_dim, hidden_dim)
        self.conv1 = GIN_Conv(self.mlp_input)
        self.conv2 = GIN_Conv(self.mlp_hidden)
        self.conv3 = GIN_Conv(self.mlp_hidden)
        self.class_layer = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings throug the 3 convolutional layers
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = x.relu()

        #TODO: resnets dropout e regularization non fanno mai male. vedere la archideddura

        # 2. Global average pooling layer
        x = global_mean_pool(x, batch)
        # 3. Classification layer

        # TODO: mettergli il turno da qualche parte (nella MLP finale credo?) MA IN TEORIA NON CI SERVIRA PERCHE ABBIAMO MESSO 1 CURRENT PLAYER 0 OTHER

        x = self.class_layer(x)

        return x

class GraphClassifier(pl.LightningModule):

    def __init__(self,in_dim, hidden_dim, num_classes):
        super().__init__()
        #self.save_hyperparameters()
        self.model = Graph_Net(in_dim, hidden_dim, num_classes)
        self.loss_module = nn.BCEWithLogitsLoss()
        #TODO: se si caca addosso è perché bisogna adattarlo alla CE con quando le classi so 0/1


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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.0)
        return optimizer

    def training_step(self, batch):
        loss, acc = self.forward(batch, mode="train")
        
        # Zero gradients
        self.configure_optimizers().zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.configure_optimizers().step()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    #def validation_step(self, batch, batch_idx):
    #    _, acc = self.forward(batch, mode="val")
    #    self.log('val_acc', acc, prog_bar=False)

    def test_step(self, batch):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)

    def train_epoch(self, train_loader: DataLoader):
        loss = 0.0
        for batch in tqdm(train_loader, desc="Batches", leave=False):
            loss += self.training_step(batch)
        return loss / len(train_loader)

    
    def train_network(self, train_loader: DataLoader, 
                      #val_loader: DataLoader, 
                      epochs: int = 10):
        # Use tqdm for better progress tracking
        optimizer = self.configure_optimizers()
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            ep_loss = self.train_epoch(train_loader)
            # Optionally validate after each epoch
            # self.validate(val_loader)
            tqdm.write(f'Epoch {epoch+1}/{epochs} completed. Loss : {ep_loss:.4f}')
