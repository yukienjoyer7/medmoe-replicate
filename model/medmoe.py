"""
MedMoE model.

Starts as a plain VLM (CLIP + Projector + SmolLM + Router).
Call activate_moe() after Phase 2 training to replace FFN blocks with MoEFFN
and copy trained weights into each expert (truncated to intermediate_dim).

Three-phase design:
  Phase 1: only proj trains         — plain VLM, no MoE
  Phase 2: LLM + router train       — plain VLM, no MoE
  Phase 3: activate_moe() → MoE trains — MoE active, router frozen

Routing in Phase 3 (paper eq. 4):
  Router reads T_comb → softmax → G_i  (sequence-level, frozen)
  MoEWrapper stores G_i before each lm() call so MoEFFN can read it.
  No separate learned gate inside MoEFFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from model.clip_encoder import CLIPEncoder
from model.projector import Projector
from model.moe_ffn import MoEFFN, Expert
from model.router import Router

MODEL_ID         = "HuggingFaceTB/SmolLM-135M-Instruct"
HIDDEN_DIM       = 576
INTERMEDIATE_DIM = 576   # expert inner dim — kept small for 6GB VRAM


class MoEWrapper(nn.Module):
    """
    Shim: looks like LlamaMLP to LlamaDecoderLayer.
    MedMoE.forward() sets current_gate_probs before calling lm() so the
    frozen Router's output reaches MoEFFN without modifying transformers internals.
    """

    def __init__(self, moe: MoEFFN):
        super().__init__()
        self.moe = moe
        self.current_gate_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x, self.current_gate_probs)


class MedMoE(nn.Module):
    def __init__(self, num_domains: int = 4):
        super().__init__()
        self.clip   = CLIPEncoder()
        self.proj   = Projector(input_dim=512, output_dim=HIDDEN_DIM)
        self.router = Router(hidden_dim=HIDDEN_DIM, num_domains=num_domains)
        self.lm     = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

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

            layer.mlp = MoEWrapper(moe).to(device)

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
            aux_loss:      scalar 0.0 (kept for API compat with train scripts)
        """
        B = input_ids.shape[0]

        model_dtype   = next(self.lm.parameters()).dtype
        image_emb     = self.clip(images)
        image_token   = self.proj(image_emb).to(model_dtype)
        text_embeds   = self.lm.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([image_token, text_embeds], dim=1)  # T_comb

        img_pad       = torch.full((B, 1), -100, dtype=torch.long, device=labels.device)
        labels_padded = torch.cat([img_pad, labels], dim=1)

        # Router reads T_comb (paper eq. 3).  In Phase 3 the router is frozen
        # so no gradient flows through it, but we still compute G_i here so
        # MoEFFN receives it as gate_probs (paper eq. 4).
        router_logits = self.router(inputs_embeds)

        if self._moe_active:
            gate_probs = F.softmax(router_logits, dim=-1)   # (B, num_domains)
            for layer in self.lm.model.layers:
                layer.mlp.current_gate_probs = gate_probs

        out = self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels_padded,
            return_dict=True,
        )

        if self._moe_active:
            for layer in self.lm.model.layers:
                layer.mlp.current_gate_probs = None

        lm_loss = out.loss
        return lm_loss, router_logits, torch.tensor(0.0, device=lm_loss.device)
