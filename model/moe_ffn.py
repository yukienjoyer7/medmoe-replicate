"""
MoE FFN layer — drop-in replacement for a transformer FFN block.

Expert structure matches SmolLM's LlamaMLP (SwiGLU) so Phase 3 can copy
weights from the trained FFN. intermediate_dim is kept small (default=576)
to fit in 6GB VRAM.

Output = Σ(top-k gated domain experts) + meta_expert(x)
         ↑ specialised                    ↑ always activated, captures global info

Aux loss (Switch Transformer):
  L_aux = N * Σ_i(f_i * p_i)  — only over domain experts, not meta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.gate        = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            output:   (B, T, D)
            aux_loss: scalar
        """
        B, T, D = x.shape
        x_flat  = x.view(-1, D)
        N       = x_flat.shape[0]

        # --- gating ---
        gate_logits = self.gate(x_flat)
        gate_probs  = F.softmax(gate_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # --- domain experts ---
        domain_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            weight     = top_k_weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    domain_out[mask] += weight[mask] * self.experts[e](x_flat[mask])

        # --- meta expert (always activated) ---
        meta_out = self.meta_expert(x_flat)

        output = (domain_out + meta_out).view(B, T, D)

        # --- aux load-balancing loss (domain experts only) ---
        aux_loss = self._aux_loss(gate_probs, top_k_indices, N)

        return output, aux_loss

    def _aux_loss(
        self,
        gate_probs:    torch.Tensor,
        top_k_indices: torch.Tensor,
        N:             int,
    ) -> torch.Tensor:
        dispatch = torch.zeros(N, self.num_experts, device=gate_probs.device)
        dispatch.scatter_(1, top_k_indices, 1.0)
        f = dispatch.mean(dim=0)
        p = gate_probs.mean(dim=0)
        return self.num_experts * (f * p).sum()
