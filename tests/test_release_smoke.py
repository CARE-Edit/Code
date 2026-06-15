from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch
import yaml

from caredit.models import CareRoutedBlock, MaskRepaint
from caredit.pipeline.flux_care import resize_mask_tokens
from scripts.create_toy_dataset import main as create_toy_dataset_main


class ReleaseSmokeTest(unittest.TestCase):
    def test_routed_block_and_mask_repaint_shapes(self) -> None:
        tokens = torch.randn(2, 16, 8)
        mask = torch.rand(2, 16)
        reference = torch.randn(2, 16, 8)
        base = torch.randn(2, 16, 8)
        block = CareRoutedBlock(8, top_k=3)
        repaint = MaskRepaint(8)

        routed, router_output = block(
            tokens,
            mask=mask,
            reference=reference,
            base=base,
            task_ids=torch.tensor([0, 2]),
            timesteps=torch.tensor([0.25, 0.75]),
        )
        predicted_mask = repaint(routed, reference, mask)

        self.assertEqual(routed.shape, tokens.shape)
        self.assertEqual(router_output.weights.shape, (2, 16, 4))
        self.assertEqual(predicted_mask.shape, (2, 16, 1))

    def test_config_matches_current_implementation_scope(self) -> None:
        with Path("configs/care_full.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["model"]["care_placement"], "post_transformer")
        self.assertNotIn("routed_layer_stride", config["model"])
        self.assertIn("gradient_accumulation_steps", config["training"])

    def test_mask_token_resize(self) -> None:
        mask = torch.ones(1, 1, 8, 8)
        tokens = resize_mask_tokens(mask, token_count=16)

        self.assertEqual(tokens.shape, (1, 16))

    def test_toy_dataset_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["create_toy_dataset.py", "--output-dir", tmpdir]
                create_toy_dataset_main()
            finally:
                sys.argv = old_argv

            self.assertTrue((Path(tmpdir) / "train.json").exists())
            self.assertTrue((Path(tmpdir) / "images" / "base.png").exists())


if __name__ == "__main__":
    unittest.main()
