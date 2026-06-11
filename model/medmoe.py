"""
MedMoE model.

Starts as a plain VLM (CLIP + Projector + SmolLM + Router).
Call activate_moe() after Phase 2 training to replace FFN blocks with MoEFFN
and copy trained weights into each expert (truncated to intermediate_dim).

Three-phase design:
  Phase 1: only proj trains         — plain VLM, no MoE
  Phase 2: LLM + router train       — plain VLM, no MoE
  Phase 3: activate_moe() → MoE trains — MoE active, router frozen
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from model.clip_encoder import CLIPEncoder
from model.projector import Projector
from model.moe_ffn import MoEFFN, Expert
from model.router import Router

MODEL_ID        = "HuggingFaceTB/SmolLM-135M-Instruct"
HIDDEN_DIM      = 576
INTERMEDIATE_DIM = 576   # expert inner dim — kept small for 6GB VRAM


class MoEWrapper(nn.Module):
    """Shim: looks like LlamaMLP to LlamaDecoderLayer, collects aux_loss as side effect."""

    def __init__(self, moe: MoEFFN, aux_store: list):
        super().__init__()
        self.moe = moe
        self._aux_store = aux_store

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, aux = self.moe(x)
        self._aux_store.append(aux)
        return out


class MedMoE(nn.Module):
    def __init__(self, num_domains: int = 4):
        super().__init__()
        self.clip   = CLIPEncoder()
        self.proj   = Projector(input_dim=512, output_dim=HIDDEN_DIM)
        self.router = Router(hidden_dim=HIDDEN_DIM, num_domains=num_domains)
        self.lm     = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

        self._aux_store:  list = []
        self._moe_active: bool = False

    # ── Phase 3 ──────────────────────────────────────────────────────────────

    def activate_moe(self, num_experts: int = 4, top_k: int = 2):
        """
        Replace every LlamaMLP with MoEFFN.
        Each expert is initialised by truncating the trained LlamaMLP weights
        (gate_proj, up_proj, down_proj) to INTERMEDIATE_DIM.
        Meta-expert gets the same initialisation.
        """
        assert not self._moe_active, "MoE already active"
        device = next(self.lm.parameters()).device

        for layer in self.lm.model.layers:
            orig = layer.mlp                    # trained LlamaMLP from Phase 2
            moe  = MoEFFN(
                hidden_dim=HIDDEN_DIM,
                intermediate_dim=INTERMEDIATE_DIM,
                num_experts=num_experts,
                top_k=top_k,
            )
            # copy truncated weights into every expert and meta_expert
            for expert in list(moe.experts) + [moe.meta_expert]:
                expert.gate_proj.weight.data = orig.gate_proj.weight.data[:INTERMEDIATE_DIM, :].clone()
                expert.up_proj.weight.data   = orig.up_proj.weight.data[:INTERMEDIATE_DIM, :].clone()
                expert.down_proj.weight.data = orig.down_proj.weight.data[:, :INTERMEDIATE_DIM].clone()

            layer.mlp = MoEWrapper(moe, self._aux_store).to(device)

        self._moe_active = True
        print(f"MoE activated: {len(self.lm.model.layers)} layers × "
              f"({num_experts} experts + 1 meta), intermediate_dim={INTERMEDIATE_DIM}")

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        images:    torch.Tensor,    # (B, 3, 224, 224)
        input_ids: torch.Tensor,    # (B, T)
        labels:    torch.Tensor,    # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            lm_loss:       scalar
            router_logits: (B, num_domains)
            aux_loss:      scalar (0.0 if MoE not yet active)
        """
        B = input_ids.shape[0]

        model_dtype = next(self.lm.parameters()).dtype
        image_emb   = self.clip(images)
        image_token = self.proj(image_emb).to(model_dtype)

        text_embeds   = self.lm.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([image_token, text_embeds], dim=1)

        img_pad       = torch.full((B, 1), -100, dtype=torch.long, device=labels.device)
        labels_padded = torch.cat([img_pad, labels], dim=1)

        self._aux_store.clear()
        out = self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels_padded,
            output_hidden_states=True,
            return_dict=True,
        )

        lm_loss       = out.loss
        hidden_states = out.hidden_states
        router_logits = self.router(hidden_states)

        aux_loss = (
            torch.stack(self._aux_store).mean()
            if self._moe_active
            else torch.tensor(0.0, device=lm_loss.device)
        )

        return lm_loss, router_logits, aux_loss
