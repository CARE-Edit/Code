"""Training module for CARE-Edit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from caredit.models import CareRoutedBlock, MaskRepaint, RouterOutput
from caredit.pipeline.flux_care import encode_images, load_flux_pipeline, resize_mask_tokens, transformer_forward
from caredit.training.losses import CareLossBreakdown, CareLossWeights, compute_care_loss

try:
    from peft import LoraConfig
except Exception:  # pragma: no cover
    LoraConfig = None


class CareEditModel(nn.Module):
    """FLUX-backed CARE-Edit training wrapper."""

    def __init__(self, config: dict[str, Any], *, device: str | torch.device = "cuda") -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
        self.pipe = load_flux_pipeline(model_cfg["flux_path"], device=device, dtype=dtype)
        self.transformer = self.pipe.transformer
        self.transformer.requires_grad_(False)
        self.transformer.train()
        self._init_lora(model_cfg)
        self.dtype = dtype
        self.device_ref = torch.device(device)
        self.loss_weights = CareLossWeights(**config.get("loss", {}))

        hidden_size = getattr(self.transformer.config, "in_channels", 64)
        self.care_block = CareRoutedBlock(
            hidden_size,
            top_k=model_cfg.get("top_k", 3),
            shared_expert_weight=model_cfg.get("shared_expert_weight", 0.05),
        ).to(device=device, dtype=dtype)
        self.mask_repaint = MaskRepaint(hidden_size).to(device=device, dtype=dtype)

    def _init_lora(self, model_cfg: dict[str, Any]) -> None:
        if LoraConfig is None:
            raise ImportError("peft is required for CARE-Edit LoRA training.")
        rank = int(model_cfg.get("lora_rank", 4))
        target_modules = model_cfg.get(
            "target_modules",
            "(.*x_embedder|.*attn\\.to_k|.*attn\\.to_q|.*attn\\.to_v|.*attn\\.to_out\\.0|.*proj_out)",
        )
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        for adapter_name in ("base", "reference", "mask"):
            self.transformer.add_adapter(lora_config, adapter_name=adapter_name)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch: dict[str, Any]) -> CareLossBreakdown:
        if encode_images is None or transformer_forward is None:
            raise ImportError("OminiControl FLUX helpers are required for CARE training.")
        target = batch["target_image"].to(self.device_ref, dtype=self.dtype)
        base = batch["base_image"].to(self.device_ref, dtype=self.dtype)
        reference = batch["reference_image"].to(self.device_ref, dtype=self.dtype)
        mask_img = batch["mask"].to(self.device_ref, dtype=self.dtype).repeat(1, 3, 1, 1)
        prompts = batch["prompts"]
        task_ids = batch["task_ids"].to(self.device_ref)

        with torch.no_grad():
            x_0, img_ids = encode_images(self.pipe, target)
            base_latents, base_ids = encode_images(self.pipe, base)
            ref_latents, ref_ids = encode_images(self.pipe, reference)
            mask_latents, mask_ids = encode_images(self.pipe, mask_img)
            prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                device=self.device_ref,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )
            t = torch.sigmoid(torch.randn((target.shape[0],), device=self.device_ref))
            x_1 = torch.randn_like(x_0)
            x_t = ((1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1).to(self.dtype)
            guidance = torch.ones_like(t) if self.transformer.config.guidance_embeds else None

        branch_count = 5
        group_mask = torch.ones([branch_count, branch_count], dtype=torch.bool, device=self.device_ref)
        group_mask[2:, 2:] = torch.eye(3, dtype=torch.bool, device=self.device_ref)
        pred = transformer_forward(
            self.transformer,
            image_features=[x_t, base_latents, ref_latents, mask_latents],
            text_features=[prompt_embeds],
            img_ids=[img_ids, base_ids, ref_ids, mask_ids],
            txt_ids=[text_ids],
            timesteps=[t, t, torch.zeros_like(t), torch.zeros_like(t), torch.zeros_like(t)],
            pooled_projections=[pooled_prompt_embeds] * branch_count,
            guidances=[guidance] * branch_count,
            adapters=[None, None, "base", "reference", "mask"],
            return_dict=False,
            group_mask=group_mask,
        )[0]

        router_output: RouterOutput | None = None
        mixture_weights = None
        predicted_mask = None
        if pred.shape[-1] == base_latents.shape[-1]:
            cared, router_output = self.care_block(
                pred,
                text=None,
                mask=resize_mask_tokens(batch["mask"].to(self.device_ref), pred.shape[1]).to(pred.dtype),
                reference=ref_latents,
                base=base_latents,
                task_ids=task_ids,
                timesteps=t,
            )
            pred = cared
            mixture_weights = router_output.weights
            predicted_mask = self.mask_repaint(
                pred,
                ref_latents,
                resize_mask_tokens(batch["mask"].to(self.device_ref), pred.shape[1]).to(pred.dtype),
            )

        return compute_care_loss(
            pred,
            x_1 - x_0,
            router_output=router_output,
            mixture_weights=mixture_weights,
            predicted_mask=_tokens_to_square_mask(
                predicted_mask,
                target_hw=batch["fine_mask"].shape[-2:],
            )
            if predicted_mask is not None
            else None,
            target_mask=batch["fine_mask"].to(self.device_ref, dtype=self.dtype),
            weights=self.loss_weights,
        )

    def save_trainable(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        state = {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names or name.startswith(("care_block.", "mask_repaint."))
        }
        torch.save(state, path)

    def load_trainable(self, path: str | Path, *, strict: bool = False) -> Any:
        """Load weights saved by `save_trainable` back into the training wrapper."""

        state = torch.load(path, map_location=self.device_ref)
        return self.load_state_dict(state, strict=strict)


def _tokens_to_square_mask(
    tokens: torch.Tensor | None,
    target_hw: tuple[int, int] | torch.Size | None = None,
) -> torch.Tensor | None:
    if tokens is None:
        return None
    side = int(tokens.shape[1] ** 0.5)
    if side * side != tokens.shape[1]:
        return None
    mask = tokens.transpose(1, 2).reshape(tokens.shape[0], 1, side, side)
    if target_hw is not None and tuple(mask.shape[-2:]) != tuple(target_hw):
        mask = torch.nn.functional.interpolate(mask, size=tuple(target_hw), mode="bilinear")
    return mask
