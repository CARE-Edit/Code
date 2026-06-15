"""CARE-Edit training loss composition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from caredit.models.latent_mixture import total_variation_loss
from caredit.models.mask_repaint import boundary_loss
from caredit.models.router import RouterOutput, load_balance_loss


@dataclass
class CareLossWeights:
    diffusion: float = 1.0
    load: float = 0.01
    mask: float = 0.1
    mix: float = 0.05


@dataclass
class CareLossBreakdown:
    total: torch.Tensor
    diffusion: torch.Tensor
    load: torch.Tensor
    mask: torch.Tensor
    mix: torch.Tensor

    def as_log_dict(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach().cpu()),
            "loss/diffusion": float(self.diffusion.detach().cpu()),
            "loss/load": float(self.load.detach().cpu()),
            "loss/mask": float(self.mask.detach().cpu()),
            "loss/mix": float(self.mix.detach().cpu()),
        }


def compute_care_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    router_output: RouterOutput | None = None,
    mixture_weights: torch.Tensor | None = None,
    predicted_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    weights: CareLossWeights | None = None,
) -> CareLossBreakdown:
    weights = weights or CareLossWeights()
    diffusion = F.mse_loss(prediction.float(), target.float(), reduction="mean")
    load = (
        load_balance_loss(router_output.probabilities)
        if router_output is not None
        else prediction.new_zeros(())
    )
    mask = (
        boundary_loss(predicted_mask, target_mask)
        if predicted_mask is not None and target_mask is not None
        else prediction.new_zeros(())
    )
    mix = (
        total_variation_loss(mixture_weights)
        if mixture_weights is not None
        else prediction.new_zeros(())
    )
    total = (
        weights.diffusion * diffusion
        + weights.load * load
        + weights.mask * mask
        + weights.mix * mix
    )
    return CareLossBreakdown(total=total, diffusion=diffusion, load=load, mask=mask, mix=mix)
