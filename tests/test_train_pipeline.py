from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
