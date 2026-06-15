# Evaluation Protocol

CARE-Edit evaluates instruction-based image editing on **EMU-Edit** and
**MagicBrush**, plus subject-driven editing on **DreamBench++**, using the
official test splits. This document records the benchmark preparation and
reporting protocol; it intentionally avoids shipping a standalone metric script
until the official benchmark wrappers are released.

## Benchmarks

### EMU-Edit

- Use the official EMU-Edit test split.
- Keep each source image, instruction, and target image unchanged.
- Convert each sample to the CARE-Edit JSON schema with:
  - `base_image`: source image.
  - `target_image`: official edited target.
  - `prompt`: edit instruction.
  - `task_type`: `instruction`.
  - `mask` and `reference_image`: omitted unless the benchmark sample provides
    them.

### MagicBrush

- Use the official MagicBrush test split.
- Preserve the official source image, instruction, and target image.
- If a region annotation or mask is available, store it in `mask`; otherwise the
  sample is evaluated as an instruction-only edit.
- Convert each sample to the CARE-Edit JSON schema with `task_type` set to
  `instruction`.

### DreamBench++

- Use the official DreamBench++ subject and prompt splits.
- Store the source image in `base_image` and the subject/style reference in
  `reference_image`.
- Preserve the official prompt text and subject IDs.
- Convert each sample to the CARE-Edit JSON schema with `task_type` set to
  `multi_subject` or `style`, depending on the benchmark category.

## Generation

For all benchmarks:

- Generate one edited image per test sample.
- Use the same resolution and preprocessing described in the experiment setup.
- Keep prompts unmodified except for any normalization specified by the
  benchmark protocol.
- Save outputs with stable sample IDs so they can be matched back to official
  targets.

## Metrics

Report the same metric families used in the paper:

- **CLIPim**: image preservation / image-level consistency.
- **CLIPout**: text-image alignment for the edited output and instruction.
- **L1**: reconstruction distance to the target image.
- **DINO**: perceptual/semantic similarity to the target image.
- **CLIP-I / CLIP-T / DINO-I**: subject-driven image and text alignment metrics
  for DreamBench++.

When releasing the final benchmark scripts, the implementations should call the
official metric backends or pinned public implementations and should include the
exact model names, preprocessing, and aggregation rules in the script header.

## Reporting

For each benchmark, report:

- Number of evaluated samples.
- Mean score for each metric.
- Any skipped sample IDs and the reason for skipping.
- The model checkpoint hash and generation configuration.

This protocol is meant to make the future evaluation code auditable and avoid
mixing informal placeholder metrics with the paper's reported benchmark numbers.
