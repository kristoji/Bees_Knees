#!/usr/bin/env python3
"""
Projection layers to map game embeddings (GIN/move) into the LLM hidden space.

This provides the HiveProjectionModule and ProjectionConfig used by trainer/inference.
Shapes expected by the trainer:
- state_projection: accepts [B, 1, gin_dim] or [B, gin_dim] and returns [B, llm_dim]
- move_projection.forward_single: accepts [B, 1, move_dim] or [B, move_dim] and returns [B, llm_dim]
- move_projection (call): accepts [B, M, 1, move_dim] (or [B, M, move_dim]) and returns [B, M, llm_dim]

The implementation is a simple MLP with configurable depth and dropout.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProjectionConfig:
    gin_embedding_dim: int = 256
    move_embedding_dim: int = 256
    llm_token_dim: int = 4096
    hidden_dim: int = 512
    projection_intermediate_dim: int = 1024
    dropout_rate: float = 0.1
    num_layers: int = 3
    activation: str = "gelu"
    use_move_filtering: bool = True


def _act(name: str) -> nn.Module:
    name = (name or "").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    # default gelu
    return nn.GELU()


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int, p: float, act: str) -> nn.Sequential:
    layers = []
    last = in_dim
    for i in range(max(0, n_layers - 1)):
        layers += [nn.Linear(last, hidden), _act(act), nn.Dropout(p)]
        last = hidden
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class _MoveProjection(nn.Module):
    def __init__(self, move_dim: int, llm_dim: int, hidden: int, n_layers: int, p: float, act: str):
        super().__init__()
        self.proj = _mlp(move_dim, llm_dim, hidden, n_layers, p, act)

    def forward(self, moves: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        moves: [B, M, 1, D] or [B, M, D]
        returns: [B, M, llm_dim]
        """
        if moves.dim() == 4:
            B, M, _, D = moves.shape
            x = moves.view(B, M, D)
        elif moves.dim() == 3:
            B, M, D = moves.shape
            x = moves
        else:
            raise ValueError(f"Unexpected move tensor shape: {moves.shape}")

        y = self.proj(x)  # [B, M, llm_dim]
        return y

    def forward_single(self, move: torch.Tensor) -> torch.Tensor:
        """
        move: [B, 1, D] or [B, D]
        returns: [B, llm_dim]
        """
        if move.dim() == 3:
            B, one, D = move.shape
            move = move.view(B, D)
        return self.proj(move)


class HiveProjectionModule(nn.Module):
    def __init__(
        self,
        gin_embedding_dim: int,
        llm_token_dim: int,
        move_embedding_mode: str = "difference",
        hidden_dim: int = 512,
        projection_intermediate_dim: int = 1024,
        dropout_rate: float = 0.1,
        num_layers: int = 3,
        activation: str = "gelu",
        use_move_filtering: bool = True,
    ):
        super().__init__()
        self.config = ProjectionConfig(
            gin_embedding_dim=gin_embedding_dim,
            move_embedding_dim=gin_embedding_dim,  # if move dim differs, adjust at callsite
            llm_token_dim=llm_token_dim,
            hidden_dim=hidden_dim,
            projection_intermediate_dim=projection_intermediate_dim,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            activation=activation,
            use_move_filtering=use_move_filtering,
        )

        # Board/state projection: [B, 1, gin_dim] -> [B, llm_dim]
        self._state_mlp = _mlp(
            self.config.gin_embedding_dim,
            self.config.llm_token_dim,
            hidden_dim,
            num_layers,
            dropout_rate,
            activation,
        )

        # Move projection: [B, M, 1, move_dim] -> [B, M, llm_dim]
        self.move_projection = _MoveProjection(
            move_dim=self.config.move_embedding_dim,
            llm_dim=self.config.llm_token_dim,
            hidden=hidden_dim,
            n_layers=num_layers,
            p=dropout_rate,
            act=activation,
        )

        # Initialize weights conservatively
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def state_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, gin_dim] or [B, gin_dim]
        returns: [B, llm_dim]
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        return self._state_mlp(x)
