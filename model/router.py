"""
Router — domain classifier on top of mean hidden states.
Single linear layer: 768 → num_domains.
Trained with cross-entropy independently from LM loss.
"""

import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_domains: int = 4):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_domains)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            hidden_states: tuple of (B, T, D) tensors, one per transformer layer
        Returns:
            logits: (B, num_domains)
        """
        # stack all layers → (num_layers, B, T, D)
        stacked = torch.stack(hidden_states, dim=0)
        # mean across layers and token positions → (B, D)
        h_mean = stacked.mean(dim=0).mean(dim=1)
        return self.linear(h_mean)
