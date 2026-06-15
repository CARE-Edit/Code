# CARE-Edit Runbook

## Verified Locally

These commands check Python syntax, imports, and lightweight unit tests:

```bash
python -m compileall caredit scripts app.py
python -m unittest discover -s tests
```

## Training Requirements

Full training requires:

- HuggingFace access to `black-forest-labs/FLUX.1-dev`
- the vendored OminiControl utilities under `caredit/vendor/omini_control`,
  used for FLUX token packing and transformer forwarding
- CARE-format training JSON and images
- CUDA GPU memory sufficient for FLUX.1-dev LoRA fine-tuning

Create a tiny local dataset for schema and dataloader checks:

```bash
python scripts/create_toy_dataset.py --output-dir examples/toy
```

Run:

```bash
python scripts/train_care.py --config configs/care_full.yaml
```

Use `configs/care_full.yaml` for the current scaffold training schedule. It
uses post-transformer CARE modules, gradient accumulation, and optional
transformer gradient checkpointing.

## Inference Requirements

Run:

```bash
python scripts/infer_care.py \
  --config configs/care_full.yaml \
  --checkpoint outputs/full/checkpoints/final.pt \
  --base-image path/to/base.png \
  --mask path/to/mask.png \
  --reference-image path/to/ref.png \
  --prompt "Replace the selected object." \
  --output outputs/result.png
```

The `--checkpoint` argument should point to a checkpoint produced by
`CareEditModel.save_trainable`, such as `outputs/full/checkpoints/final.pt`.
If it is omitted, the script falls back to the FLUX/OminiControl wrapper and
should be treated as a smoke test rather than CARE-Edit paper inference.

## Evaluation

For EMU-Edit and MagicBrush, use the official test splits and metric
implementations. The expected preparation and reporting protocol is documented
in `docs/evaluation_protocol.md`.
