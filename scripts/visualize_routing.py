"""Visualize CARE-Edit router weights saved as tensors."""

from __future__ import annotations

import argparse
import _bootstrap  # noqa: F401
from pathlib import Path

import torch
from PIL import Image


EXPERT_NAMES = ("text", "mask", "reference", "base")


def save_map(values: torch.Tensor, path: Path) -> None:
    values = values.float()
    values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)
    array = (values.cpu().numpy() * 255).astype("uint8")
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Tensor [B, N, E] saved with torch.save")
    parser.add_argument("--output-dir", default="outputs/routing")
    args = parser.parse_args()

    weights = torch.load(args.weights, map_location="cpu")
    if isinstance(weights, dict):
        weights = weights.get("weights", weights.get("router_weights"))
    if weights is None or weights.ndim != 3:
        raise ValueError("Expected router weights with shape [B, N, E]")
    side = int(weights.shape[1] ** 0.5)
    if side * side != weights.shape[1]:
        raise ValueError("Token count must be square for visualization")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for batch_idx in range(weights.shape[0]):
        for expert_idx in range(weights.shape[2]):
            name = EXPERT_NAMES[expert_idx] if expert_idx < len(EXPERT_NAMES) else f"expert_{expert_idx}"
            save_map(weights[batch_idx, :, expert_idx].reshape(side, side), out / f"b{batch_idx}_{name}.png")


if __name__ == "__main__":
    main()
