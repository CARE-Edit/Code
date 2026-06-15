# CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing

**CARE-Edit routes diffusion tokens to specialized experts for high-fidelity contextual image editing.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.08589-b31b1b.svg)](https://arxiv.org/abs/2603.08589)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://care-edit.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2603.08589)
![CVPR 2026](https://img.shields.io/badge/CVPR-2026-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FLUX](https://img.shields.io/badge/Backbone-FLUX.1--dev-purple)

> [Yucheng Wang*](https://wangandyyucheng.github.io/), [Zedong Wang*](https://jacky1128.github.io/), [Yuetong Wu](https://scholar.google.com/citations?user=Fjqr0fEAAAAJ&hl=zh-CN), [Yue Ma](https://mayuelala.github.io/), [Dan Xu†](https://www.danxurgb.net/)
>
> The Hong Kong University of Science and Technology


## News

- ☑ Our paper is now available on [arXiv](https://arxiv.org/abs/2603.08589).
- ☑ **CARE-Edit** is accepted by **CVPR 2026**. 
- ☑ Training and Inference code have been pre-released.
- ☐ All checkpoints and datasets full-released.

## Roadmap

- [x] Release Routing Select, expert adapters, Mask Repaint, and Latent Mixture modules.
- [x] Release training, inference, Gradio, and visualization entry points.
- [ ] Release curated datasets.
- [ ] Release benchmark-specific conversion wrappers.

## Overview

Unified image editors often process text, masks, reference images, and source images through a fixed shared pathway. CARE-Edit instead allocates computation dynamically: a lightweight router selects the most relevant experts for each latent token and timestep, then a latent mixture module fuses expert outputs while Mask Repaint refines spatial guidance.

![CARE-Edit motivation and tasks](assets/abstract.svg)

## Framework

CARE-Edit is built on a frozen FLUX DiT backbone with lightweight trainable modules:

- **Text Expert** for semantic reasoning and instruction following.
- **Mask Expert** for spatial localization and boundary-sensitive editing.
- **Reference Expert** for subject identity and style transfer.
- **Base Expert** for global coherence and background preservation.

![CARE-Edit framework](assets/out.svg)

## Results

### Contextual Editing

![CARE-Edit contextual editing examples](assets/comp1.svg)

### Qualitative Comparisons

![CARE-Edit qualitative comparisons](assets/comp2.svg)

## Release Contents

```text
CARE-Edit/
├── app.py                         # Gradio demo
├── configs/
│   └── care_full.yaml             # Current training schedule
├── caredit/
│   ├── data/                      # CARE-format dataset loader
│   ├── models/                    # Experts, router, Mask Repaint, Latent Mixture
│   ├── pipeline/                  # FLUX/OminiControl pipeline wrapper
│   ├── training/                  # CARE loss and training model
│   └── vendor/                    # Minimal vendored OminiControl utilities
├── docs/
│   ├── data_format.md             # Dataset schema
│   ├── evaluation_protocol.md     # EMU-Edit and MagicBrush evaluation notes
│   └── runbook.md                 # Verified commands and runtime notes
└── scripts/                       # Train, infer, convert, visualize
```

## Installation

Create an environment with Python 3.10 or newer:

```bash
conda create -n care-edit python=3.10 -y
conda activate care-edit
```

Install the package:

```bash
pip install -e .[train]
```

This release vendors the minimal OminiControl utilities needed for FLUX token packing and multi-condition transformer forwarding under `caredit/vendor/omini_control/`, so a separate sibling `OminiControl/` checkout is not required.

## Model Preparation

You need access to the FLUX base checkpoint:

| Component | Source | Used For |
| --- | --- | --- |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | frozen DiT backbone, VAE, text encoders |
| CARE-Edit trainable weights | official release checkpoint | router, experts, mask repaint, latent mixture, LoRA adapters |

Full training and inference require Hugging Face access to FLUX.1-dev.

## Data Format

Training and benchmark preparation use a simple JSON schema:

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

See [`docs/data_format.md`](docs/data_format.md) for the full schema.


## Training

Current training schedule:

```bash
python scripts/train_care.py --config configs/care_full.yaml
```

The config follows the released scaffold setting: FLUX.1-dev, LoRA rank 4, AdamW
with learning rate `1e-4`, two-stage curriculum training, gradient accumulation,
and the combined objective:

```text
LCARE = Ldiff + 0.01 Lload + 0.1 Lmask + 0.05 Lmix
```

## Inference

```bash
python scripts/infer_care.py \
  --config configs/care_full.yaml \
  --checkpoint outputs/full/checkpoints/final.pt \
  --base-image path/to/base.png \
  --mask path/to/mask.png \
  --reference-image path/to/reference.png \
  --prompt "Replace the selected object with the reference object." \
  --output outputs/result.png
```

You can also set `model.checkpoint` in `configs/care_full.yaml`. Without a CARE
checkpoint, the CLI falls back to the base FLUX/OminiControl wrapper.

See [`docs/implementation_scope.md`](docs/implementation_scope.md) for the
current implementation boundary and the paper features that are not yet claimed.

## Gradio Demo

```bash
$env:CARE_EDIT_CHECKPOINT="outputs/full/checkpoints/final.pt"
python app.py
```

The demo supports instruction editing, object removal or replacement,
reference-guided editing, and mask-guided local editing when a compatible CARE
checkpoint is provided. Without `CARE_EDIT_CHECKPOINT`, it is a wrapper smoke
test rather than a paper-result demo.

## Evaluation

We follow the instruction-based editing protocol used in the paper for
**EMU-Edit** and **MagicBrush**. Both benchmarks are converted to the CARE-Edit
input schema while preserving the official test split, source image, edit
instruction, and target image. Generated outputs are then evaluated against the
official targets with the same metric families reported in the paper, including
image preservation, text-image alignment, reconstruction distance, and DINO
similarity.

See [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) for the
benchmark preparation and reporting protocol. We do not include a placeholder
metric script here because benchmark evaluation should use the official splits
and metric implementations.

## Router Visualization

Save router weights from a training or inference run as a tensor with shape
`[B, N, E]`, then render expert maps:

```bash
python scripts/visualize_routing.py \
  --weights outputs/router_weights.pt \
  --output-dir outputs/routing
```


## License

The main repository is released under the Apache 2.0 license. See
[`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for
third-party code, model, and dataset notices.

## Citation

If CARE-Edit is helpful for your research, please cite:

```bibtex
@inproceedings{wang2026careedit,
  title={CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing},
  author={Wang, Yucheng and Wang, Zedong and Wu, Yuetong and Ma, Yue and Xu, Dan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Acknowledgements

This release builds on ideas and engineering patterns from the following projects:

- [OminiControl](https://github.com/Yuanshi9815/OminiControl) for FLUX multi-condition control and LoRA training references.
- [UNO](https://github.com/bytedance/UNO) for FLUX subject-conditioning, data flow, and evaluation references.


## Contact

For questions about the paper, please contact [ywangls@connect.ust.hk](mailto:ywangls@connect.ust.hk).
