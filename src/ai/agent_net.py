"""Value network plus a policy head that scores every legal move in one forward pass.

Why this exists: without a policy head, MCTS builds its prior by evaluating the value
network on every child, which is b+1 forward passes per expansion. Measured on this
corpus, b averages 62.6 and reaches 196, and on the cluster one move at 1600 rollouts
costs 28 seconds almost entirely because of that. With the policy head an expansion is
one pass.

The graph has a node per *piece*, so a move's destination — an empty cell in nearly every
case — has no node. Rather than add nodes for empty cells, which would change the input
representation and throw away the trained value network and the rebuilt shards, a move is
described by the pieces it touches (see ai.move_index):

    h_src   the embedding of the bug being moved, or a learned per-bug vector when it is
            still in hand
    h_dst   the mean embedding of the pieces the destination touches: the piece under it
            when climbing, otherwise the occupied cells around it
    context the pooled graph embedding, so the score can depend on the whole position

The trunk is the existing Graph_Net untouched, and the value head is still its classifier,
so a checkpoint trained by train_value.py loads into the trunk unchanged.
"""
import torch
import torch.nn as nn

from ai.graph_network import MLP, Graph_Net

NUM_BUGS = 28


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, pooled_dim, bug_dim=16, dropout=0.1):
        super().__init__()
        # A bug that is not on the board yet has no node, so it gets a learned vector.
        self.in_hand = nn.Embedding(NUM_BUGS, hidden_dim)
        self.bug_type = nn.Embedding(NUM_BUGS, bug_dim)
        self.score = MLP(2 * hidden_dim + pooled_dim + bug_dim, hidden_dim, 1,
                         num_layers=2, dropout=dropout, use_layer_norm=True)

    def forward(self, node_h, pooled, src, bug, dst, move_graph):
        """Return one logit per move.

        src (M,)    node index of the moving bug, -1 when in hand
        bug (M,)    bug index in 0..27
        dst (M, K)  node indices the destination touches, -1 padded
        move_graph (M,)  which graph in the batch each move belongs to
        """
        on_board = src >= 0
        h_src = torch.where(on_board.unsqueeze(-1),
                            node_h[src.clamp(min=0)],
                            self.in_hand(bug))

        mask = (dst >= 0).unsqueeze(-1)
        gathered = node_h[dst.clamp(min=0)] * mask
        h_dst = gathered.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        feat = torch.cat([h_src, h_dst, pooled[move_graph], self.bug_type(bug)], dim=-1)
        return self.score(feat).squeeze(-1)


class AgentNet(nn.Module):
    """Graph_Net trunk, its value head, and the policy head on top."""

    def __init__(self, in_dim=13, hidden_dim=64, **trunk_kwargs):
        super().__init__()
        self.trunk = Graph_Net(in_dim, hidden_dim, 1, **trunk_kwargs)
        pooled_dim = hidden_dim * 3 if trunk_kwargs.get("pooling") == "concat" else hidden_dim
        self.policy = PolicyHead(hidden_dim, pooled_dim,
                                 dropout=trunk_kwargs.get("final_dropout", 0.1))

    def forward(self, x, edge_index, batch, src=None, bug=None, dst=None, move_graph=None):
        """Value logits for the batch, and move logits when the move index is given."""
        node_h = self.trunk.node_embeddings(x, edge_index)
        pooled = self.trunk.pool(node_h, batch)
        value = self.trunk.classifier(pooled).squeeze(-1)
        if src is None:
            return value, None
        return value, self.policy(node_h, pooled, src, bug, dst, move_graph)


def segment_log_softmax(logits, segment, num_segments):
    """log-softmax over variable-length groups of moves, one group per graph."""
    highest = torch.full((num_segments,), float("-inf"), device=logits.device,
                         dtype=logits.dtype)
    highest = highest.scatter_reduce(0, segment, logits, reduce="amax",
                                     include_self=False)
    shifted = (logits - highest[segment]).exp()
    total = torch.zeros(num_segments, device=logits.device, dtype=logits.dtype)
    total = total.index_add(0, segment, shifted)
    log_z = highest + total.clamp(min=1e-20).log()
    return logits - log_z[segment]
