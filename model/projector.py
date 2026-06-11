"""
Projector — aligns CLIP CLS token (512-dim) into GPT-2 token space (768-dim).
Trainable 2-layer MLP: 512 → 768 → 768.
"""

import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, input_dim: int = 512, output_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 512) — CLIP CLS token
        Returns:
            (B, 1, 768) — single image token ready to prepend to LLM sequence
        """
        return self.net(x).unsqueeze(1)
