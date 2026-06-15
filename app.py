"""Gradio demo for CARE-Edit reproduction."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover
    raise ImportError("Install gradio to run the CARE-Edit demo.") from exc

from caredit.pipeline import CareFluxPipeline


CONFIG_PATH = Path("configs/care_full.yaml")
PIPELINE: CareFluxPipeline | None = None


def get_pipeline() -> CareFluxPipeline:
    global PIPELINE
    if PIPELINE is None:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        PIPELINE = CareFluxPipeline.from_pretrained(
            config["model"]["flux_path"],
            device=device,
            dtype=getattr(torch, config["model"].get("dtype", "bfloat16")),
            model_config=config.get("model", {}),
            checkpoint_path=os.environ.get("CARE_EDIT_CHECKPOINT"),
        )
    return PIPELINE


def generate(base_image, mask, reference_image, prompt, steps, guidance):
    pipe = get_pipeline()
    result = pipe(
        prompt=prompt,
        base_image=base_image,
        mask=mask,
        reference_image=reference_image,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    )
    return result.images[0]


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="pil", label="Base Image"),
        gr.Image(type="pil", label="Mask", image_mode="L"),
        gr.Image(type="pil", label="Reference Image"),
        gr.Textbox(label="Prompt", value="Remove the selected object."),
        gr.Slider(4, 50, value=28, step=1, label="Steps"),
        gr.Slider(1.0, 8.0, value=3.5, step=0.1, label="Guidance"),
    ],
    outputs=gr.Image(type="pil", label="CARE-Edit Result"),
    title="CARE-Edit Pre-release Demo",
)


if __name__ == "__main__":
    demo.launch()
