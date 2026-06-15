"""CARE-Edit inference CLI."""

from __future__ import annotations

import argparse
import _bootstrap  # noqa: F401
from pathlib import Path

import torch
import yaml
from PIL import Image

from caredit.pipeline import CareFluxPipeline


def load_image(path: str | None) -> Image.Image | None:
    return Image.open(path).convert("RGB") if path else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/care_full.yaml")
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--reference-image")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--checkpoint",
        help="CARE-Edit trainable checkpoint saved by scripts/train_care.py. "
        "Falls back to model.checkpoint in the config when omitted.",
    )
    parser.add_argument("--output", default="outputs/inference/result.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dtype = getattr(torch, config["model"].get("dtype", "bfloat16"))
    generator = torch.Generator(device=args.device).manual_seed(int(config["inference"].get("seed", 42)))
    pipe = CareFluxPipeline.from_pretrained(
        config["model"]["flux_path"],
        device=args.device,
        dtype=dtype,
        model_config=config.get("model", {}),
        checkpoint_path=args.checkpoint,
    )
    result = pipe(
        prompt=args.prompt,
        base_image=load_image(args.base_image),
        mask=load_image(args.mask),
        reference_image=load_image(args.reference_image),
        height=int(config["model"].get("resolution", 512)),
        width=int(config["model"].get("resolution", 512)),
        num_inference_steps=int(config["inference"].get("num_steps", 28)),
        guidance_scale=float(config["inference"].get("guidance_scale", 3.5)),
        image_guidance_scale=float(config["inference"].get("image_guidance_scale", 1.0)),
        generator=generator,
    )
    image = result.images[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
