"""Condition-aware token router for CARE-Edit."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RouterOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    weights: torch.Tensor
    topk_indices: torch.Tensor
    temperature: float


class ConditionAwareRouter(nn.Module):
    """Latent-attention router with sparse top-k expert selection."""

    def __init__(
        self,
        dim: int,
        *,
        num_experts: int = 4,
        task_vocab_size: int = 16,
        top_k: int = 3,
        temperature: float = 1.0,
        min_temperature: float = 0.3,
        shared_expert_weight: float = 0.05,
        ema_decay: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.shared_expert_weight = shared_expert_weight
        self.ema_decay = ema_decay

        self.token_key = nn.Linear(dim, dim)
        self.task_embed = nn.Embedding(task_vocab_size, dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, num_experts),
        )
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        self.register_buffer("ema_logits", torch.zeros(num_experts), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        task_ids: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        temperature: float | None = None,
    ) -> RouterOutput:
        batch, seq_len, _ = tokens.shape
        if task_ids is None:
            task_ids = torch.zeros(batch, dtype=torch.long, device=tokens.device)
        if timesteps is None:
            timesteps = torch.zeros(batch, dtype=tokens.dtype, device=tokens.device)

        token_key = self.token_key(tokens)
        task_context = self.task_embed(task_ids.to(tokens.device))
        time_context = self.time_embed(timesteps.to(tokens.device, tokens.dtype).view(batch, 1))
        query = (task_context + time_context).unsqueeze(1).expand(-1, seq_len, -1)
        logits = self.router(torch.cat([token_key, query], dim=-1)) + self.expert_bias

        if self.training and self.ema_decay > 0:
            current = logits.detach().mean(dim=(0, 1))
            self.ema_logits.mul_(self.ema_decay).add_(current, alpha=1 - self.ema_decay)
            logits = logits - self.ema_logits.view(1, 1, -1)

        temp = max(float(temperature if temperature is not None else self.temperature), self.min_temperature)
        topk_values, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
        sparse = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse.scatter_(-1, topk_indices, topk_values)
        probabilities = F.softmax(logits / temp, dim=-1)
        weights = F.softmax(sparse / temp, dim=-1)

        if self.shared_expert_weight > 0:
            shared = torch.zeros_like(weights)
            shared[..., -1] = 1.0
            weights = (1 - self.shared_expert_weight) * weights + self.shared_expert_weight * shared
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return RouterOutput(
            logits=logits,
            probabilities=probabilities,
            weights=weights,
            topk_indices=topk_indices,
            temperature=temp,
        )

    def annealed_temperature(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return self.temperature
        progress = min(max(step / total_steps, 0.0), 1.0)
        return self.temperature + progress * (self.min_temperature - self.temperature)


def load_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    """Encourage uniform average expert usage."""

    expert_usage = probabilities.mean(dim=(0, 1))
    target = torch.full_like(expert_usage, 1.0 / expert_usage.numel())
    return F.mse_loss(expert_usage, target, reduction="sum")
