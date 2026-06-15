"""Unified CARE-Edit dataset schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms import functional as TVF


TASK_TO_ID = {
    "instruction": 0,
    "removal": 1,
    "replacement": 2,
    "style": 3,
    "contextual": 4,
    "multi_subject": 5,
}


@dataclass
class CareSample:
    id: str
    prompt: str
    task_type: str
    base_image: torch.Tensor
    target_image: torch.Tensor | None
    mask: torch.Tensor | None
    fine_mask: torch.Tensor | None
    reference_image: torch.Tensor | None

    @property
    def task_id(self) -> int:
        return TASK_TO_ID.get(self.task_type, 0)


class CareEditDataset(Dataset[CareSample]):
    """Dataset reading the JSON schema documented in `docs/data_format.md`."""

    def __init__(
        self,
        json_file: str | Path,
        *,
        image_root: str | Path | None = None,
        resolution: int = 512,
        require_target: bool = True,
        task_filter: set[str] | None = None,
    ) -> None:
        self.json_file = Path(json_file)
        self.image_root = Path(image_root) if image_root else self.json_file.parent
        self.resolution = resolution
        self.require_target = require_target
        self.task_filter = task_filter
        with self.json_file.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError("CARE dataset JSON must contain a list of records")
        self.records = [
            record for record in records if task_filter is None or record.get("task_type") in task_filter
        ]
        self.image_transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
        self.mask_transform = ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> CareSample:
        record = self.records[idx]
        base_path = record.get("base_image", record.get("image"))
        if not base_path:
            raise KeyError("Each record needs `base_image` or `image`")
        target_path = record.get("target_image")
        if self.require_target and not target_path:
            raise KeyError("Training records need `target_image`")

        return CareSample(
            id=str(record.get("id", idx)),
            prompt=str(record.get("prompt", record.get("instruction", ""))),
            task_type=str(record.get("task_type", "instruction")),
            base_image=self._load_image(base_path),
            target_image=self._load_image(target_path) if target_path else None,
            mask=self._load_mask(record.get("mask", record.get("coarse_mask"))),
            fine_mask=self._load_mask(record.get("fine_mask")),
            reference_image=self._load_image(record.get("reference_image")) if record.get("reference_image") else None,
        )

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.image_root / path

    def _load_image(self, value: str | Path) -> torch.Tensor:
        image = Image.open(self._path(value)).convert("RGB")
        image = TVF.resize(image, [self.resolution, self.resolution])
        return self.image_transform(image)

    def _load_mask(self, value: str | Path | None) -> torch.Tensor | None:
        if not value:
            return None
        mask = Image.open(self._path(value)).convert("L")
        mask = TVF.resize(mask, [self.resolution, self.resolution], interpolation=TVF.InterpolationMode.NEAREST)
        return self.mask_transform(mask)


def _stack_optional(items: list[torch.Tensor | None], fallback: torch.Tensor) -> torch.Tensor:
    filled = [item if item is not None else torch.zeros_like(fallback) for item in items]
    return torch.stack(filled, dim=0)


def collate_care_batch(samples: list[CareSample]) -> dict[str, Any]:
    base = torch.stack([sample.base_image for sample in samples], dim=0)
    fallback_image = torch.zeros_like(samples[0].base_image)
    fallback_mask = torch.zeros(1, *samples[0].base_image.shape[-2:], dtype=base.dtype)
    return {
        "ids": [sample.id for sample in samples],
        "prompts": [sample.prompt for sample in samples],
        "task_types": [sample.task_type for sample in samples],
        "task_ids": torch.tensor([sample.task_id for sample in samples], dtype=torch.long),
        "base_image": base,
        "target_image": _stack_optional([sample.target_image for sample in samples], fallback_image),
        "reference_image": _stack_optional([sample.reference_image for sample in samples], fallback_image),
        "mask": _stack_optional([sample.mask for sample in samples], fallback_mask),
        "fine_mask": _stack_optional([sample.fine_mask for sample in samples], fallback_mask),
        "has_reference": torch.tensor([sample.reference_image is not None for sample in samples]),
        "has_mask": torch.tensor([sample.mask is not None for sample in samples]),
    }
