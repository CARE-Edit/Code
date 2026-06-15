"""Heterogeneous expert adapters for CARE-Edit."""

from __future__ import annotations

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    """Small residual MLP adapter shared by CARE expert implementations."""

    def __init__(self, dim: int, bottleneck_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(dim // 4, 16)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(self.norm(tokens))


class CrossAttentionAdapter(nn.Module):
    """Residual adapter with optional condition cross-attention."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        bottleneck_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.condition_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.adapter = ResidualAdapter(dim, bottleneck_dim=bottleneck_dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            return self.adapter(tokens)
        query = self.norm(tokens)
        key_value = self.condition_norm(condition)
        attended, _ = self.attn(query, key_value, key_value, need_weights=False)
        return self.adapter(tokens + attended)


class TextExpert(CrossAttentionAdapter):
    """Semantic expert driven by prompt/text embeddings."""


class BaseExpert(CrossAttentionAdapter):
    """Global structure expert anchored to the source image."""


class ReferenceExpert(nn.Module):
    """Identity/style expert using FiLM-style modulation from reference tokens."""

    def __init__(
        self,
        dim: int,
        *,
        bottleneck_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ref_pool = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
        )
        self.adapter = ResidualAdapter(dim, bottleneck_dim=bottleneck_dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        if reference is None:
            return self.adapter(tokens)
        scale, shift = self.ref_pool(reference.mean(dim=1)).chunk(2, dim=-1)
        modulated = self.norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.adapter(tokens + modulated)


class MaskExpert(nn.Module):
    """Spatial expert gated by mask tokens."""

    def __init__(
        self,
        dim: int,
        *,
        bottleneck_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_proj = nn.Sequential(nn.Linear(1, dim), nn.Sigmoid())
        self.adapter = ResidualAdapter(dim, bottleneck_dim=bottleneck_dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return self.adapter(tokens)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        gate = self.mask_proj(mask.to(dtype=tokens.dtype, device=tokens.device))
        return self.adapter(tokens * (1 + gate))


class CareExpertBank(nn.Module):
    """Container that runs all CARE experts and returns outputs in fixed order."""

    names = ("text", "mask", "reference", "base")

    def __init__(self, dim: int, *, num_heads: int = 8, bottleneck_dim: int | None = None) -> None:
        super().__init__()
        self.text = TextExpert(dim, num_heads=num_heads, bottleneck_dim=bottleneck_dim)
        self.mask = MaskExpert(dim, bottleneck_dim=bottleneck_dim)
        self.reference = ReferenceExpert(dim, bottleneck_dim=bottleneck_dim)
        self.base = BaseExpert(dim, num_heads=num_heads, bottleneck_dim=bottleneck_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        text: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        base: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "text": self.text(tokens, text),
            "mask": self.mask(tokens, mask),
            "reference": self.reference(tokens, reference),
            "base": self.base(tokens, base),
        }
