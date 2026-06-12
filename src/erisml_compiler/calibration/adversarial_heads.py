"""Adversarial heads for language and period invariance.

Multi-head architecture (sqnd-probe v10.16.7+):
  - NUM_ADV_HEADS heads with varying architectures
  - Each head tries to predict the nuisance variable (language or period)
  - Encoder receives reversed gradients from these heads, so it learns
    to make the embedding uninformative about the nuisance

Heads' loss enters the total loss with a configurable lambda. The
gradient-reversal layer (in losses.py) flips the sign on the way back
to the encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from erisml_compiler.calibration.losses import grad_reverse


class AdversarialHead(nn.Module):
    """A single adversarial classifier head (architecture varies)."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiHeadAdversarial(nn.Module):
    """Bank of adversarial heads with varying widths/depths.

    All heads see the same input (after gradient reversal); encoder must
    fool all of them simultaneously, preventing adversarial-hiding.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        n_heads: int = 4,
        adversarial_lambda: float = 1.0,
    ):
        super().__init__()
        # Vary architectures across heads.
        configs = [
            {"hidden_dim": 256, "n_layers": 2, "dropout": 0.3},
            {"hidden_dim": 512, "n_layers": 3, "dropout": 0.2},
            {"hidden_dim": 128, "n_layers": 2, "dropout": 0.4},
            {"hidden_dim": 256, "n_layers": 4, "dropout": 0.2},
        ]
        self.heads = nn.ModuleList([
            AdversarialHead(in_dim=in_dim, num_classes=num_classes, **configs[i % len(configs)])
            for i in range(n_heads)
        ])
        self.adversarial_lambda = adversarial_lambda

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Returns one logits tensor per head. Gradient reversal is applied
        once at the input, so the encoder receives reversed gradients."""
        z_reversed = grad_reverse(z, self.adversarial_lambda)
        return [head(z_reversed) for head in self.heads]
