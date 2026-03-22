from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_impl.data.market import load_market_bundle
from quant_impl.pipelines.predict import predict_pipeline
from quant_impl.pipelines.train import train_pipeline
from quant_impl.pipelines.validate import validate_pipeline

from tests.helpers import make_synthetic_market_parquet, make_test_config


class PredictionValidationTest(unittest.TestCase):
    def test_prediction_archive_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)
            validation = validate_pipeline(config)
            self.assertEqual(prediction["status"], "pending")
            self.assertEqual(validation["validated"], 1)
            self.assertTrue(config.validation_history_path.exists())


if __name__ == "__main__":
    unittest.main()
