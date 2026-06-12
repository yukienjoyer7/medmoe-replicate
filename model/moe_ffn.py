"""
MoE FFN layer — drop-in replacement for a transformer FFN block.

Expert structure matches SmolLM's LlamaMLP (SwiGLU) so Phase 3 can copy
weights from the trained FFN. intermediate_dim is kept small (default=576)
to fit in 6GB VRAM.

Output = Σ(top-k gated domain experts) + meta_expert(x)   (paper eq. 4)
         ↑ G_i from frozen sequence-level Router            ↑ always activated

No learned gate inside this module — gating comes from the Router via
MoEWrapper.current_gate_probs (set by MedMoE.forward before calling lm).
"""

import torch
import torch.nn as nn


class Expert(nn.Module):
    """SwiGLU expert — matches LlamaMLP structure for weight copying in Phase 3."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN with meta-expert.

    Args:
        hidden_dim:       token embedding dimension (576 for SmolLM)
        intermediate_dim: inner FFN dimension (kept at 576 to save memory)
        num_experts:      domain expert count (default 4)
        top_k:            experts activated per token (default 2)
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        assert top_k <= num_experts

        self.num_experts = num_experts
        self.top_k       = top_k

        self.experts     = nn.ModuleList(
            [Expert(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )
        self.meta_expert = Expert(hidden_dim, intermediate_dim)

        self.last_routing: dict | None = None  # populated on every forward pass

    def forward(self, x: torch.Tensor, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, T, D)
            gate_probs: (B, num_experts) — softmax of Router logits, sequence-level
        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape
        x_flat  = x.view(-1, D)   # (B*T, D)

        # broadcast sequence-level gate to every token in the sequence
        token_probs = gate_probs.unsqueeze(1).expand(-1, T, -1).reshape(B * T, self.num_experts)

        top_k_probs, top_k_indices = torch.topk(token_probs, self.top_k, dim=-1)
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        self.last_routing = {
            "indices":    top_k_indices.detach().cpu(),   # (B*T, top_k)
            "weights":    top_k_weights.detach().cpu(),   # (B*T, top_k)
            "gate_probs": token_probs.detach().cpu(),     # (B*T, num_experts)
        }

        # --- domain experts ---
        domain_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            weight     = top_k_weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    domain_out[mask] += weight[mask] * self.experts[e](x_flat[mask])

        # --- meta expert (always activated, captures global info) ---
        meta_out = self.meta_expert(x_flat)

        return (domain_out + meta_out).view(B, T, D)
