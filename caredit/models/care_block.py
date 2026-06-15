"""Composable CARE-Edit routed expert block."""

from __future__ import annotations

import torch
from torch import nn

from .experts import CareExpertBank
from .latent_mixture import LatentMixture
from .router import ConditionAwareRouter, RouterOutput


class CareRoutedBlock(nn.Module):
    """Run all experts, route token-wise, and fuse outputs."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        top_k: int = 3,
        task_vocab_size: int = 16,
        shared_expert_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.experts = CareExpertBank(dim, num_heads=num_heads)
        self.router = ConditionAwareRouter(
            dim,
            task_vocab_size=task_vocab_size,
            top_k=top_k,
            shared_expert_weight=shared_expert_weight,
        )
        self.mixture = LatentMixture(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        text: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        base: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterOutput]:
        router_output = self.router(tokens, task_ids=task_ids, timesteps=timesteps)
        expert_outputs = self.experts(
            tokens,
            text=text,
            mask=mask,
            reference=reference,
            base=base,
        )
        mixed, _ = self.mixture(
            expert_outputs,
            router_output.weights,
            base_output=expert_outputs["base"],
            timesteps=timesteps,
        )
        return mixed, router_output
