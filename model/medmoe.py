"""
MedMoE — full model wiring CLIP + Projector + GPT-2 with MoE FFNs + Router.

Forward pass:
  image  → CLIPEncoder → Projector → image token (B, 1, 768)
  text   → GPT-2 embedding → text tokens (B, T, 768)
  concat → [image token | text tokens] → GPT-2 transformer (MoE FFNs)
         → lm_logits (B, T+1, vocab)
         → hidden_states → Router → domain_logits (B, 4)
         → aux_loss (sum across all MoE layers)

MoEWrapper is a thin shim: it looks like a standard FFN to GPT2Block
(returns only the output tensor) but collects aux_loss as a side effect.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

from model.clip_encoder import CLIPEncoder
from model.projector import Projector
from model.moe_ffn import MoEFFN
from model.router import Router


class MoEWrapper(nn.Module):
    """
    Wraps MoEFFN to match the GPT2MLP interface (single tensor in, single tensor out).
    Aux losses are written to self._aux_store, a list passed in from the parent model.
    """

    def __init__(self, moe: MoEFFN, aux_store: list):
        super().__init__()
        self.moe = moe
        self._aux_store = aux_store

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, aux = self.moe(x)
        self._aux_store.append(aux)
        return out


class MedMoE(nn.Module):
    def __init__(
        self,
        num_experts: int = 4,
        top_k: int = 2,
        num_domains: int = 4,
    ):
        super().__init__()

        self.clip    = CLIPEncoder()
        self.proj    = Projector(input_dim=512, output_dim=768)
        self.router  = Router(hidden_dim=768, num_domains=num_domains)

        # load GPT-2 and patch every FFN block with MoEFFN
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        self._aux_store: list = []
        self._patch_ffn_blocks(num_experts, top_k)

    def _patch_ffn_blocks(self, num_experts: int, top_k: int):
        for block in self.lm.transformer.h:
            moe = MoEFFN(hidden_dim=768, num_experts=num_experts, top_k=top_k)
            block.mlp = MoEWrapper(moe, self._aux_store)

    def forward(
        self,
        images: torch.Tensor,       # (B, 3, 224, 224)
        input_ids: torch.Tensor,    # (B, T)
        labels: torch.Tensor,       # (B, T) — -100 for ignored positions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            lm_loss:      scalar
            router_logits:(B, num_domains)
            aux_loss:     scalar — sum of L_aux across all MoE layers
        """
        B, T = input_ids.shape

        # --- image token ---
        image_emb   = self.clip(images)             # (B, 512)
        image_token = self.proj(image_emb)          # (B, 1, 768)

        # --- text embeddings ---
        text_embeds = self.lm.transformer.wte(input_ids)   # (B, T, 768)

        # --- prepend image token ---
        inputs_embeds = torch.cat([image_token, text_embeds], dim=1)  # (B, T+1, 768)

        # pad labels with -100 for the image token position
        img_label_pad = torch.full((B, 1), -100, dtype=torch.long, device=labels.device)
        labels_padded = torch.cat([img_label_pad, labels], dim=1)     # (B, T+1)

        # --- GPT-2 forward ---
        self._aux_store.clear()
        out = self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels_padded,
            output_hidden_states=True,
            return_dict=True,
        )

        lm_loss      = out.loss
        hidden_states = out.hidden_states   # tuple of (B, T+1, 768), one per layer

        # --- router ---
        router_logits = self.router(hidden_states)  # (B, num_domains)

        # --- aggregate aux loss across all MoE layers ---
        aux_loss = torch.stack(self._aux_store).mean()

        return lm_loss, router_logits, aux_loss
