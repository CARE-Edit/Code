"""Convert common JSON records to the CARE-Edit schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pick(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key]:
            return record[key]
    return None


def convert_record(record: dict[str, Any], idx: int, default_task: str) -> dict[str, Any]:
    image_paths = pick(record, "image_paths", "images")
    reference = pick(record, "reference_image", "ref_image", "ref")
    if reference is None and isinstance(image_paths, list) and image_paths:
        reference = image_paths[0]
    base = pick(record, "base_image", "source_image", "input_image", "image")
    if base is None and isinstance(image_paths, list) and len(image_paths) > 1:
        base = image_paths[0]
    return {
        "id": str(pick(record, "id", "uid") or idx),
        "task_type": str(pick(record, "task_type", "task") or default_task),
        "prompt": str(pick(record, "prompt", "instruction", "text") or ""),
        "base_image": base,
        "target_image": pick(record, "target_image", "image_tgt_path", "edited_image", "output_image"),
        "mask": pick(record, "mask", "coarse_mask", "mask_path"),
        "fine_mask": pick(record, "fine_mask", "gt_mask"),
        "reference_image": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON list")
    parser.add_argument("--output", required=True, help="Output CARE JSON")
    parser.add_argument("--default-task", default="instruction")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list")

    converted = [convert_record(record, idx, args.default_task) for idx, record in enumerate(records)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
