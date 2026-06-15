"""FLUX pipeline helpers for CARE-Edit.

This module keeps the diffusers/OminiControl integration in one place. The
lightweight CARE modules can be tested without FLUX, while these helpers are
used by training and inference when the heavy dependencies and checkpoints are
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
import torch.nn.functional as F
from PIL import Image

from caredit.models import CareRoutedBlock, MaskRepaint

try:
    from caredit.vendor.omini_control.flux_omini import (
        Condition as OminiCondition,
        encode_images,
        generate as omini_generate,
        transformer_forward,
    )
except Exception:  # pragma: no cover - exercised only without vendored optional deps.
    OminiCondition = None
    encode_images = None
    omini_generate = None
    transformer_forward = None

try:
    from diffusers.pipelines import FluxPipeline
except Exception:  # pragma: no cover - diffusers may be absent in lightweight checks.
    FluxPipeline = Any  # type: ignore

try:
    from peft import LoraConfig
except Exception:  # pragma: no cover
    LoraConfig = None


ConditionKind = Literal["base", "reference", "mask", "text", "shared"]


@dataclass
class CareCondition:
    """A typed condition used by CARE-Edit."""

    image: Image.Image | torch.Tensor
    kind: ConditionKind
    adapter: str | None = None
    position_delta: tuple[int, int] | None = None
    position_scale: float = 1.0
    latent_mask: torch.Tensor | None = None
    is_complement: bool = False

    def to_omini(self) -> Any:
        if OminiCondition is None:
            raise ImportError("Vendored OminiControl pipeline utilities are unavailable.")
        return OminiCondition(
            condition=self.image,
            adapter_setting=self.adapter,
            position_delta=self.position_delta,
            position_scale=self.position_scale,
            latent_mask=self.latent_mask,
            is_complement=self.is_complement,
        )


def load_flux_pipeline(
    flux_path: str,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> FluxPipeline:
    """Load a FLUX pipeline with the encoders frozen by default."""

    if FluxPipeline is Any:
        raise ImportError("diffusers is required to load a FLUX pipeline.")
    pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=dtype).to(device)
    for module_name in ("text_encoder", "text_encoder_2", "vae"):
        module = getattr(pipe, module_name, None)
        if module is not None:
            module.requires_grad_(False).eval()
    return pipe


def mask_to_condition_image(mask: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
    """Convert a mask to a 3-channel condition image accepted by FLUX VAE."""

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 3 and mask.shape[0] == 1:
            return mask.repeat(3, 1, 1)
        if mask.ndim == 4 and mask.shape[1] == 1:
            return mask.repeat(1, 3, 1, 1)
        return mask
    return mask.convert("L").convert("RGB")


def make_care_conditions(
    *,
    base_image: Image.Image | torch.Tensor | None = None,
    reference_image: Image.Image | torch.Tensor | None = None,
    mask: Image.Image | torch.Tensor | None = None,
    adapters: dict[str, str | None] | None = None,
) -> list[CareCondition]:
    """Create CARE conditions in the paper's B/R/M order."""

    adapters = adapters or {}
    conditions: list[CareCondition] = []
    if base_image is not None:
        conditions.append(CareCondition(base_image, "base", adapter=adapters.get("base", "base")))
    if reference_image is not None:
        conditions.append(
            CareCondition(
                reference_image,
                "reference",
                adapter=adapters.get("reference", "reference"),
                position_delta=(0, -32),
            )
        )
    if mask is not None:
        conditions.append(
            CareCondition(mask_to_condition_image(mask), "mask", adapter=adapters.get("mask", "mask"))
        )
    return conditions


class CareFluxPipeline:
    """Thin CARE-Edit wrapper around the OminiControl FLUX generation loop."""

    def __init__(
        self,
        pipe: FluxPipeline,
        *,
        adapters: dict[str, str | None] | None = None,
        model_config: dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.pipe = pipe
        self.adapters = adapters or {
            "main": None,
            "base": "base",
            "reference": "reference",
            "mask": "mask",
        }
        self.model_config = model_config or {}
        self.dtype = dtype
        self.device = next(pipe.transformer.parameters()).device
        hidden_size = getattr(pipe.transformer.config, "in_channels", 64)
        self.care_block = CareRoutedBlock(
            hidden_size,
            top_k=self.model_config.get("top_k", 3),
            shared_expert_weight=self.model_config.get("shared_expert_weight", 0.05),
        ).to(device=self.device, dtype=dtype)
        self.mask_repaint = MaskRepaint(hidden_size).to(device=self.device, dtype=dtype)
        self.care_block.eval()
        self.mask_repaint.eval()
        self._care_enabled = False
        self._condition_kinds: list[ConditionKind] = []
        self._raw_mask: Image.Image | torch.Tensor | None = None
        if checkpoint_path:
            self.load_trainable(checkpoint_path)

    @classmethod
    def from_pretrained(
        cls,
        flux_path: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        adapters: dict[str, str | None] | None = None,
        model_config: dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> "CareFluxPipeline":
        return cls(
            load_flux_pipeline(flux_path, device=device, dtype=dtype),
            adapters=adapters,
            model_config=model_config,
            checkpoint_path=checkpoint_path or (model_config or {}).get("checkpoint"),
            dtype=dtype,
        )

    def _init_lora_adapters(self) -> None:
        """Create the inference adapters expected by CARE training checkpoints."""

        if LoraConfig is None:
            raise ImportError("peft is required to load CARE-Edit LoRA checkpoints.")
        rank = int(self.model_config.get("lora_rank", 4))
        target_modules = self.model_config.get(
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
            try:
                self.pipe.transformer.add_adapter(lora_config, adapter_name=adapter_name)
            except ValueError:
                # PEFT raises when an adapter already exists; that is fine for repeated loads.
                continue

    def load_trainable(self, path: str | Path, *, strict: bool = False) -> Any:
        """Load a `CareEditModel.save_trainable()` checkpoint for inference."""

        self._init_lora_adapters()
        state = torch.load(path, map_location=self.device)
        transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        care_state = {
            key.removeprefix("care_block."): value
            for key, value in state.items()
            if key.startswith("care_block.")
        }
        repaint_state = {
            key.removeprefix("mask_repaint."): value
            for key, value in state.items()
            if key.startswith("mask_repaint.")
        }
        result = {
            "transformer": self.pipe.transformer.load_state_dict(transformer_state, strict=strict),
            "care_block": self.care_block.load_state_dict(care_state, strict=strict),
            "mask_repaint": self.mask_repaint.load_state_dict(repaint_state, strict=strict),
        }
        self._care_enabled = True
        return result

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str,
        base_image: Image.Image | torch.Tensor | None = None,
        mask: Image.Image | torch.Tensor | None = None,
        reference_image: Image.Image | torch.Tensor | None = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        image_guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        transformer_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if omini_generate is None:
            raise ImportError("Vendored OminiControl generate() is unavailable.")
        conditions = make_care_conditions(
            base_image=base_image,
            reference_image=reference_image,
            mask=mask,
            adapters=self.adapters,
        )
        self._condition_kinds = [condition.kind for condition in conditions]
        self._raw_mask = mask
        return omini_generate(
            self.pipe,
            prompt=prompt,
            conditions=[condition.to_omini() for condition in conditions],
            main_adapter=self.adapters.get("main"),
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            output_type=output_type,
            return_dict=return_dict,
            noise_pred_postprocessor=self._postprocess_noise_pred if self._care_enabled else None,
            transformer_kwargs={**self.model_config, **(transformer_kwargs or {})},
        )

    def _postprocess_noise_pred(
        self,
        *,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        condition_latents: list[torch.Tensor],
        **_: Any,
    ) -> torch.Tensor:
        """Apply the post-transformer CARE modules used by the current trainer."""

        if not self._care_enabled:
            return noise_pred
        base = self._condition_latents("base", condition_latents, like=noise_pred)
        reference = self._condition_latents("reference", condition_latents, like=noise_pred)
        mask_tokens = self._mask_tokens(noise_pred)
        cared, _ = self.care_block(
            noise_pred,
            text=None,
            mask=mask_tokens,
            reference=reference,
            base=base,
            timesteps=timestep.to(noise_pred.device, noise_pred.dtype),
        )
        if reference is not None and mask_tokens is not None:
            _ = self.mask_repaint(cared, reference, mask_tokens)
        return cared

    def _condition_latents(
        self,
        kind: ConditionKind,
        condition_latents: list[torch.Tensor],
        *,
        like: torch.Tensor,
    ) -> torch.Tensor | None:
        for condition_kind, latents in zip(self._condition_kinds, condition_latents, strict=False):
            if condition_kind == kind and latents.shape[-1] == like.shape[-1]:
                if latents.shape[1] != like.shape[1]:
                    return None
                if latents.shape[0] == like.shape[0]:
                    return latents.to(device=like.device, dtype=like.dtype)
                if latents.shape[0] == 1:
                    return latents.expand(like.shape[0], -1, -1).to(device=like.device, dtype=like.dtype)
        return None

    def _mask_tokens(self, like: torch.Tensor) -> torch.Tensor | None:
        if self._raw_mask is None:
            return None
        token_count = like.shape[1]
        side = int(token_count**0.5)
        if side * side != token_count:
            return None
        if isinstance(self._raw_mask, torch.Tensor):
            mask = self._raw_mask.to(device=like.device, dtype=like.dtype)
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            if mask.shape[1] != 1:
                mask = mask[:, :1]
            tokens = resize_mask_tokens(mask, token_count)
            if tokens.shape[0] == like.shape[0]:
                return tokens
            if tokens.shape[0] == 1:
                return tokens.expand(like.shape[0], -1)
            return None
        mask_image = self._raw_mask.convert("L").resize((side, side), Image.Resampling.NEAREST)
        values = torch.tensor(list(mask_image.getdata()), device=like.device, dtype=like.dtype) / 255.0
        return values.view(1, token_count).expand(like.shape[0], -1)


def encode_optional_image(
    pipe: FluxPipeline,
    image: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Encode an optional image tensor into FLUX latent tokens."""

    if image is None:
        return None, None
    if encode_images is None:
        raise ImportError("Vendored OminiControl encode_images() is unavailable.")
    return encode_images(pipe, image)


def resize_mask_tokens(mask: torch.Tensor, token_count: int) -> torch.Tensor:
    """Resize a BCHW mask to a flattened token mask."""

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    side = int(token_count**0.5)
    if side * side != token_count:
        return F.interpolate(mask.float(), size=(token_count, 1), mode="nearest").flatten(1)
    resized = F.interpolate(mask.float(), size=(side, side), mode="nearest")
    return resized.flatten(1)


def resolve_path(path: str | Path, roots: Iterable[str | Path]) -> Path:
    """Resolve `path` against the first root containing it."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in roots:
        resolved = Path(root) / candidate
        if resolved.exists():
            return resolved
    return Path(next(iter(roots), ".")) / candidate
