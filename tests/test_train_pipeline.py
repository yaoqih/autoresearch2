from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from quant_impl.data.market import load_market_bundle
from quant_impl.pipelines.train import train_pipeline

from tests.helpers import make_synthetic_market_parquet, make_test_config


class TrainPipelineTest(unittest.TestCase):
    def test_train_pipeline_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            result = train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            self.assertIn("summary", result)
            self.assertTrue(config.model_path.exists())
            self.assertIn("research_score", result["summary"])

    def test_train_pipeline_uses_latest_rolling_window_for_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)

            result = train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config, force=False)
            deployment_fit = result["deployment_fit"]
            deployment_valid_start = len(bundle["dates"]) - config.data.valid_days
            expected_train_start = max(0, deployment_valid_start - config.data.train_days)
            artifact = torch.load(config.model_path, map_location="cpu", weights_only=False)

            self.assertEqual(deployment_fit["train_start_index"], expected_train_start)
            self.assertEqual(deployment_fit["train_end_index"], deployment_valid_start)
            self.assertEqual(deployment_fit["valid_start_index"], deployment_valid_start)
            self.assertEqual(deployment_fit["valid_end_index"], len(bundle["dates"]))
            self.assertEqual(len(deployment_fit["history"]), 5)
            self.assertEqual(artifact["deployment_training_config"]["epochs"], 5)
            self.assertFalse(artifact["deployment_training_config"]["early_stopping_enabled"])

    def test_train_pipeline_deploy_only_skips_cross_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)

            result = train_pipeline(config, device="cpu", profile="full", force_prepare=True, deploy_only=True)
            bundle = load_market_bundle(config, force=False)
            artifact = torch.load(config.model_path, map_location="cpu", weights_only=False)
            expected_train_start = len(bundle["dates"]) - config.data.train_days

            self.assertEqual(result["folds"], [])
            self.assertIsNone(result["summary"])
            self.assertTrue(result["deploy_only"])
            self.assertEqual(len(result["deployment_fit"]["history"]), 5)
            self.assertEqual(result["deployment_fit"]["train_start_index"], expected_train_start)
            self.assertEqual(result["deployment_fit"]["train_end_index"], len(bundle["dates"]))
            self.assertEqual(result["deployment_fit"]["valid_days"], 0)
            self.assertEqual(result["deployment_fit"]["valid_start_index"], 0)
            self.assertEqual(result["deployment_fit"]["valid_end_index"], 0)
            self.assertEqual(result["deployment_fit"]["best_epoch"], config.training.deployment_epochs)
            self.assertIsNone(result["deployment_fit"]["best_valid_metrics"])
            self.assertTrue(config.model_path.exists())
            self.assertTrue(artifact["deploy_only"])
            self.assertEqual(artifact["summary"], None)

    def test_train_pipeline_uses_configured_deployment_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.training.deployment_epochs = 3
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)

            result = train_pipeline(config, device="cpu", profile="full", force_prepare=True, deploy_only=True)
            artifact = torch.load(config.model_path, map_location="cpu", weights_only=False)

            self.assertEqual(len(result["deployment_fit"]["history"]), 3)
            self.assertEqual(result["deployment_training_config"]["epochs"], 3)
            self.assertEqual(artifact["deployment_training_config"]["epochs"], 3)


if __name__ == "__main__":
    unittest.main()
