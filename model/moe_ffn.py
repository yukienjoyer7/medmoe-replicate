"""
MoE FFN layer — drop-in replacement for a standard transformer FFN block.

Architecture:
  input (B, T, D)
    → gating network → softmax → top-k selection
    → each token routed to k experts
    → weighted sum of expert outputs
  output (B, T, D)

Also computes the Switch Transformer auxiliary load-balancing loss:
  L_aux = N * sum_i(f_i * p_i)
where f_i = hard dispatch fraction (no grad), p_i = soft routing prob (has grad).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert: same structure as a GPT-2 FFN (D → 4D → D)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN.

    Args:
        hidden_dim: token embedding dimension (768 for GPT-2)
        num_experts: number of parallel expert FFNs (default 4)
        top_k: how many experts each token is routed to (default 2)
    """

    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        assert top_k <= num_experts

        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
            aux_loss: scalar tensor
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)          # (B*T, D)
        N = x_flat.shape[0]             # total tokens in batch

        # --- gating ---
        gate_logits = self.gate(x_flat)                         # (N, num_experts)
        gate_probs  = F.softmax(gate_logits, dim=-1)            # (N, num_experts)

        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # re-normalise so selected weights sum to 1
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # (N, k)

        # --- expert computation ---
        output = torch.zeros_like(x_flat)                       # (N, D)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]                    # (N,)
            weight     = top_k_weights[:, k].unsqueeze(-1)      # (N, 1)

            for e in range(self.num_experts):
                mask = (expert_idx == e)                        # (N,) bool
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e](x_flat[mask])

        output = output.view(B, T, D)

        # --- auxiliary load-balancing loss ---
        aux_loss = self._aux_loss(gate_probs, top_k_indices, N)

        return output, aux_loss

    def _aux_loss(
        self,
        gate_probs: torch.Tensor,   # (N, num_experts) — soft, has grad
        top_k_indices: torch.Tensor,# (N, k) — hard dispatch
        N: int,
    ) -> torch.Tensor:
        # f_i: fraction of tokens dispatched to expert i (no gradient)
        dispatch = torch.zeros(N, self.num_experts, device=gate_probs.device)
        dispatch.scatter_(1, top_k_indices, 1.0)
        f = dispatch.mean(dim=0)                    # (num_experts,)

        # p_i: mean routing probability for expert i (has gradient)
        p = gate_probs.mean(dim=0)                  # (num_experts,)

        return self.num_experts * (f * p).sum()
