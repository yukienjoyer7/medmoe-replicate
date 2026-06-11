"""
CLIP vision encoder — frozen, returns CLS token only (1x512).
"""

import torch
import torch.nn as nn
import open_clip


class CLIPEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        super().__init__()
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual
        self.output_dim = self.visual.output_dim  # 512 for ViT-B/32

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) — preprocessed with self.preprocess
        Returns:
            (B, 512) — CLS token embedding
        """
        return self.visual(images).float()
