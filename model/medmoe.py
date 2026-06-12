"""
MedMoE model.

Starts as a plain VLM (CLIP + Projector + SmolLM + Router).
Call activate_moe() after Phase 2 training to replace FFN blocks with MoEFFN
and copy trained weights into each expert (truncated to intermediate_dim).

Three-phase design:
  Phase 1: only proj trains         — plain VLM, no MoE
  Phase 2: LLM + router train       — plain VLM, no MoE
  Phase 3: activate_moe() → MoE trains — MoE active, router frozen

Routing (paper Fig. 2 + eq. 4):
  Phase 2: router reads T_i (projector output, pre-LLM) → trained on domain labels
  Phase 3: router frozen inside each MoEWrapper; reads x[:, 0:1, :] (image token
           position in per-layer hidden state) → gate_probs → selects top-k experts.
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
    Holds a frozen ref to the Router; calls it on x[:, 0:1, :] (image token
    position in the current hidden state) to produce gate_probs per layer —
    matching paper Fig. 2 Phase 3 where Router sits inside each MoE block.
    """

    def __init__(self, moe: MoEFFN, router=None):
        super().__init__()
        self.moe = moe
        # non-registered reference so router params don't appear in
        # self.parameters() and are untouched by freeze_for_phase3
        object.__setattr__(self, "_router", router)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router = object.__getattribute__(self, "_router")
        if router is not None:
            gate_logits = router(x[:, 0:1, :])          # image token at this layer
            gate_probs  = F.softmax(gate_logits, dim=-1)
        else:
            gate_probs = None
        return self.moe(x, gate_probs)


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

            layer.mlp = MoEWrapper(moe, router=self.router).to(device)

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

        # Router reads T_i only (paper Fig. 2: orange arrow from projector → Router).
        # Phase 2: computes loss for router training.
        # Phase 3: router is frozen; gate_probs are computed inside each MoEWrapper
        #          on the per-layer hidden state, so no injection needed here.
        router_logits = self.router(image_token)

        out = self.lm(
            inputs_embeds=inputs_embeds,
            labels=labels_padded,
            return_dict=True,
        )

        lm_loss = out.loss
        return lm_loss, router_logits, torch.tensor(0.0, device=lm_loss.device)
