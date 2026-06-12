"""
Router — domain classifier on T_comb (image + text input tokens).
Single linear layer (MLPx1, 0.02MB) — best config per Med-MoE Fig 3 ablation (row b).
Trained with cross-entropy on modality labels (Phase 2).
Frozen in Phase 3 — its softmax output IS the expert gate G_i (paper eq. 4).
"""

import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, hidden_dim: int = 576, num_domains: int = 4):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_domains)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds: (B, T, D) — T_comb (image token prepended to text tokens)
        Returns:
            logits: (B, num_domains)
        """
        h = inputs_embeds.mean(dim=1)   # (B, D) — mean-pool over sequence
        return self.linear(h)
