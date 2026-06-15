"""Create a tiny CARE-format dataset for schema and dataloader smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="examples/toy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    base = Image.new("RGB", (128, 128), "white")
    target = Image.new("RGB", (128, 128), "white")
    reference = Image.new("RGB", (128, 128), "white")
    mask = Image.new("L", (128, 128), 0)

    base_draw = ImageDraw.Draw(base)
    target_draw = ImageDraw.Draw(target)
    ref_draw = ImageDraw.Draw(reference)
    mask_draw = ImageDraw.Draw(mask)

    base_draw.rectangle((40, 40, 88, 88), fill="blue")
    target_draw.rectangle((40, 40, 88, 88), fill="red")
    ref_draw.rectangle((32, 32, 96, 96), fill="red")
    mask_draw.rectangle((36, 36, 92, 92), fill=255)

    base.save(image_dir / "base.png")
    target.save(image_dir / "target.png")
    reference.save(image_dir / "reference.png")
    mask.save(image_dir / "mask.png")
    mask.save(image_dir / "fine_mask.png")

    record = {
        "id": "toy-0001",
        "task_type": "replacement",
        "prompt": "Replace the blue square with the red reference square.",
        "base_image": "base.png",
        "target_image": "target.png",
        "mask": "mask.png",
        "fine_mask": "fine_mask.png",
        "reference_image": "reference.png",
    }
    with (output_dir / "train.json").open("w", encoding="utf-8") as handle:
        json.dump([record], handle, indent=2)


if __name__ == "__main__":
    main()
