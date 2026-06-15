"""Train CARE-Edit reproduction."""

from __future__ import annotations

import argparse
import _bootstrap  # noqa: F401
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from caredit.data import CareEditDataset, collate_care_batch
from caredit.training.trainer import CareEditModel


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infinite(loader):
    while True:
        yield from loader


def build_loader(config: dict, *, stage: int) -> DataLoader:
    data_cfg = config["data"]
    training_cfg = config["training"]
    tasks = data_cfg["stage1_tasks"] if stage == 1 else data_cfg["stage2_tasks"]
    dataset = CareEditDataset(
        data_cfg["train_json"],
        image_root=data_cfg.get("image_root"),
        resolution=config["model"].get("resolution", 512),
        task_filter=set(tasks),
    )
    return DataLoader(
        dataset,
        batch_size=training_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=training_cfg.get("dataloader_workers", 0),
        collate_fn=collate_care_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/care_full.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = int(config["training"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    model = CareEditModel(config, device=args.device)
    if config["training"].get("gradient_checkpointing", False):
        model.transformer.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )

    stage1_steps = int(config["training"].get("stage1_steps", 0))
    max_steps = int(config["training"].get("max_steps", 1000))
    loaders = {
        1: infinite(build_loader(config, stage=1)),
        2: infinite(build_loader(config, stage=2)),
    }
    progress = tqdm(range(1, max_steps + 1), desc="CARE-Edit training")
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    accumulation_steps = max(accumulation_steps, 1)
    precision = str(config["training"].get("mixed_precision", "no")).lower()
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = args.device.startswith("cuda") and precision in {"bf16", "fp16"}

    for step in progress:
        stage = 1 if step <= stage1_steps else 2
        optimizer.zero_grad(set_to_none=True)
        loss = None
        for _ in range(accumulation_steps):
            batch = next(loaders[stage])
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_autocast):
                loss = model(batch)
                (loss.total / accumulation_steps).backward()
        if loss is None:
            raise RuntimeError("No training loss was computed.")
        torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(),
            float(config["training"].get("max_grad_norm", 1.0)),
        )
        optimizer.step()
        progress.set_postfix({"loss": f"{float(loss.total.detach().cpu()):.4f}", "stage": stage})

        if step % int(config["training"].get("checkpointing_steps", 1000)) == 0:
            ckpt = output_dir / "checkpoints" / f"step-{step}.pt"
            model.save_trainable(ckpt)
            with (output_dir / "last_metrics.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(loss.as_log_dict(), f)

    model.save_trainable(output_dir / "checkpoints" / "final.pt")


if __name__ == "__main__":
    main()
