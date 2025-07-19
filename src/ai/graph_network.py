import os
import json
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from ai.loader import GraphDataset

class HiveGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        # first conv
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        # additional
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # policy MLP: input 2*hidden_dim -> hidden_dim -> 1
        self.pi_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # value MLP: input hidden_dim -> hidden_dim -> 1
        self.v_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, data: Data, move_adj: torch.Tensor):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # message passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # value head
        # global pool per graph in batch
        v = global_mean_pool(x, batch)           # [B, hidden_dim]
        v = self.v_mlp(v).squeeze(-1)            # [B]
        # policy head
        # iterate over graphs in batch
        pi_logits = []  # list of tensors [M_i]
        for i in range(v.shape[0]):
            # mask nodes belonging to graph i
            mask = (batch == i)
            idx = mask.nonzero(as_tuple=False).view(-1)
            x_i = x[idx]                          # [N_i, hidden_dim]
            M_i = move_adj[i]                    # assume move_adj batched as list or tensor [B, N_i, N_i]
            # find valid moves
            src, dst = M_i.nonzero(as_tuple=True)
            pairs = torch.cat([x_i[src], x_i[dst]], dim=-1)  # [num_moves, 2*hidden_dim]
            s = self.pi_mlp(pairs).squeeze(-1)    # [num_moves]
            pi_logits.append(s)
        return pi_logits, v



def hive_collate(batch):
    # batch: list of tuples (data, move_adj, pi_target, v_target)
    datas, moves, pis, vs = zip(*batch)
    batch_data = Batch.from_data_list(datas)
    return batch_data, list(moves), list(pis), list(vs)


def train_epoch(model, dataloader, optimizer, device, c_v: float = 0.5):
    model.train()
    total_loss = 0.0
    for data, moves, pis, vs in dataloader:
        data = data.to(device)
        # assume moves, pis, vs are lists per graph
        optimizer.zero_grad()
        pi_logits_list, v_pred = model(data, moves)
        loss = 0.0
        # accumulate loss over batch
        for logits, pi_t, v_t in zip(pi_logits_list, pis, vs):
            loss += F.cross_entropy(logits.unsqueeze(0), pi_t.argmax().unsqueeze(0))
            loss += c_v * F.mse_loss(v_pred, v_t.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_loop(model, dataset_paths, epochs, lr, device):
    dataset = GraphDataset(dataset_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=hive_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1, epochs+1):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
        # torch.save(model.state_dict(), f"hive_gnn_epoch{epoch}.pt")
    # TODO: save on WandB
    torch.save(model.state_dict(), "hive_gnn_final.pt")
