"""Latent Mixture fusion for CARE-Edit."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LatentMixture(nn.Module):
    """Fuse expert outputs with router weights and a timestep-aware base gate."""

    def __init__(self, dim: int, *, num_experts: int = 4) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.base_gate = nn.Sequential(
            nn.LayerNorm(dim + 1),
            nn.Linear(dim + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        expert_outputs: torch.Tensor | dict[str, torch.Tensor],
        router_weights: torch.Tensor,
        *,
        base_output: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(expert_outputs, dict):
            ordered = [expert_outputs[name] for name in ("text", "mask", "reference", "base")]
            stacked = torch.stack(ordered, dim=2)
            base_output = base_output if base_output is not None else expert_outputs.get("base")
        else:
            stacked = expert_outputs

        if stacked.ndim != 4:
            raise ValueError("expert_outputs must have shape [B, N, E, D]")
        if router_weights.shape[:3] != stacked.shape[:3]:
            raise ValueError("router_weights must have shape [B, N, E]")

        fused = (router_weights.unsqueeze(-1) * stacked).sum(dim=2)
        if base_output is None:
            return fused, router_weights

        batch = fused.shape[0]
        if timesteps is None:
            timesteps = torch.zeros(batch, dtype=fused.dtype, device=fused.device)
        pooled = base_output.mean(dim=1)
        gate_input = torch.cat([pooled, timesteps.view(batch, 1).to(pooled)], dim=-1)
        gamma = self.base_gate(gate_input).view(batch, 1, 1)
        mixed = (1 - gamma) * fused + gamma * base_output
        return mixed, router_weights


def total_variation_loss(weights: torch.Tensor, height: int | None = None, width: int | None = None) -> torch.Tensor:
    """Spatial smoothness loss for mixture maps."""

    if weights.ndim != 3:
        raise ValueError("weights must have shape [B, N, E]")
    batch, token_count, experts = weights.shape
    if height is None or width is None:
        side = int(token_count**0.5)
        if side * side != token_count:
            return weights.new_zeros(())
        height = width = side
    maps = weights.transpose(1, 2).reshape(batch, experts, height, width)
    loss_h = F.l1_loss(maps[:, :, 1:, :], maps[:, :, :-1, :])
    loss_w = F.l1_loss(maps[:, :, :, 1:], maps[:, :, :, :-1])
    return loss_h + loss_w
