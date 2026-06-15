# CARE-Edit Data Format

Training and evaluation files are JSON lists. Relative paths are resolved
against `data.image_root` in the config, or against the JSON file directory if
no root is configured.

```json
[
  {
    "id": "sample-0001",
    "task_type": "replacement",
    "prompt": "Replace the selected mug with a red teapot.",
    "base_image": "base/0001.png",
    "target_image": "target/0001.png",
    "mask": "mask/coarse/0001.png",
    "fine_mask": "mask/fine/0001.png",
    "reference_image": "ref/teapot.png"
  }
]
```

## Fields

- `prompt`: edit instruction.
- `base_image`: source image to edit. `image` is accepted as an alias.
- `target_image`: supervised edited image. Required for training, optional for
  inference-only JSON files.
- `mask`: coarse user mask used as spatial condition.
- `fine_mask`: optional boundary target for Mask Repaint.
- `reference_image`: optional reference subject or style image.
- `task_type`: one of `instruction`, `removal`, `replacement`, `style`,
  `contextual`, or `multi_subject`.

## Converter Scripts

The `scripts/convert_*.py` utilities write this schema from common research
datasets. They intentionally keep conversion explicit rather than guessing
missing masks or references.
