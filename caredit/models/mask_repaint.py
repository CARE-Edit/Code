"""Mask Repaint module for boundary-aware mask refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskRepaint(nn.Module):
    """Predict residual updates to a soft latent mask."""

    def __init__(self, dim: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current_latents: torch.Tensor,
        reference_latents: torch.Tensor | None,
        previous_mask: torch.Tensor,
    ) -> torch.Tensor:
        if previous_mask.ndim == 2:
            previous_mask = previous_mask.unsqueeze(-1)
        if reference_latents is None:
            reference_latents = torch.zeros_like(current_latents)
        if reference_latents.shape[1] != current_latents.shape[1]:
            reference_latents = _match_tokens(reference_latents, current_latents.shape[1])
        delta = torch.sigmoid(self.net(torch.cat([current_latents, reference_latents, previous_mask], dim=-1)))
        return (previous_mask + delta - 0.5).clamp(0, 1)


def _match_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[1] == target_tokens:
        return tokens
    return F.interpolate(tokens.transpose(1, 2), size=target_tokens, mode="nearest").transpose(1, 2)


def image_gradients(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    grad_y = mask[:, :, 1:, :] - mask[:, :, :-1, :]
    grad_x = mask[:, :, :, 1:] - mask[:, :, :, :-1]
    return grad_y, grad_x


def boundary_loss(
    predicted_mask: torch.Tensor,
    target_mask: torch.Tensor | None,
    *,
    smooth_weight: float = 0.1,
) -> torch.Tensor:
    """Boundary consistency plus Laplacian smoothness."""

    if target_mask is None:
        return predicted_mask.new_zeros(())
    pred_y, pred_x = image_gradients(predicted_mask)
    tgt_y, tgt_x = image_gradients(target_mask.to(predicted_mask))
    boundary = F.l1_loss(pred_y, tgt_y) + F.l1_loss(pred_x, tgt_x)
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask.unsqueeze(1)
    lap = (
        -4 * predicted_mask[:, :, 1:-1, 1:-1]
        + predicted_mask[:, :, :-2, 1:-1]
        + predicted_mask[:, :, 2:, 1:-1]
        + predicted_mask[:, :, 1:-1, :-2]
        + predicted_mask[:, :, 1:-1, 2:]
    )
    smooth = lap.abs().mean() if lap.numel() else predicted_mask.new_zeros(())
    return boundary + smooth_weight * smooth
