from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from quant_impl.data.market import load_market_bundle, realized_day_lookup
from quant_impl.pipelines.predict import predict_pipeline
from quant_impl.pipelines.train import train_pipeline
from quant_impl.pipelines.validate import validate_pipeline
from quant_impl.utils.io import read_json

from tests.helpers import make_synthetic_market_parquet, make_test_config


class PredictionValidationTest(unittest.TestCase):
    def test_predict_pipeline_uses_training_contract_and_trading_calendar_for_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = date.fromisoformat(bundle["dates"][-5])

            class StubCalendar:
                _next = {
                    as_of_date: date(2020, 4, 8),
                    date(2020, 4, 8): date(2020, 4, 9),
                    date(2020, 4, 9): date(2020, 4, 10),
                    date(2020, 4, 10): date(2020, 4, 13),
                    date(2020, 4, 13): date(2020, 4, 14),
                    date(2020, 4, 14): date(2020, 4, 15),
                }

                def next_session(self, value: date) -> date:
                    return self._next[value]

            config.data.entry_offset_days = 4
            config.data.exit_offset_days = 6
            config.data.ret_windows = (5, 3, 2)

            with patch("quant_impl.pipelines.predict.TradingCalendar", StubCalendar, create=True):
                prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date.isoformat())

            self.assertEqual(prediction["entry_date"], "2020-04-08")
            self.assertEqual(prediction["exit_date"], "2020-04-09")

    def test_predict_pipeline_rejects_feature_contract_mismatch_in_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)

            artifact = torch.load(config.model_path, map_location="cpu", weights_only=False)
            artifact["feature_names"] = list(reversed(artifact["feature_names"]))
            torch.save(artifact, config.model_path)

            with self.assertRaisesRegex(ValueError, "feature contract"):
                predict_pipeline(config, device="cpu", as_of_date=bundle["dates"][-5])

    def test_prediction_archive_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)
            daily_path = config.paths.predictions_dir / "daily" / f"{as_of_date}.json"
            index_path = config.paths.predictions_dir / "index.json"
            latest_path = config.paths.predictions_dir / "latest.json"

            self.assertTrue(daily_path.exists())
            self.assertTrue(index_path.exists())
            self.assertTrue(latest_path.exists())

            daily_payload = read_json(daily_path)
            index_payload = read_json(index_path)
            latest_payload = read_json(latest_path)

            validation = validate_pipeline(config)

            self.assertEqual(prediction["status"], "pending")
            self.assertEqual(daily_payload["status"], "pending")
            self.assertEqual(daily_payload["selected"]["code"], prediction["selected_code"])
            self.assertEqual(daily_payload["selected"]["score"], prediction["selected_score"])
            self.assertEqual(daily_payload["top_candidates"], prediction["top_candidates"])
            self.assertEqual(index_payload[0]["as_of_date"], as_of_date)
            self.assertEqual(index_payload[0]["status"], "pending")
            self.assertEqual(latest_payload["as_of_date"], as_of_date)
            self.assertEqual(validation["validated"], 1)
            self.assertTrue(config.validation_history_path.exists())

            validated_daily = read_json(daily_path)
            validated_index = read_json(index_path)
            history = pd.read_csv(config.validation_history_path)

            self.assertEqual(validated_daily["status"], "validated")
            self.assertIsNotNone(validated_daily["validation"])
            self.assertEqual(validated_daily["summary"]["alpha"], validated_daily["validation"]["alpha"])
            self.assertEqual(validated_daily["summary"]["selected_return"], validated_daily["validation"]["selected_return"])
            self.assertIn("selected_ideal_return", validated_daily["validation"])
            self.assertIn("open_limit_day1", validated_daily["validation"])
            self.assertIn("tradeable", validated_daily["validation"])
            self.assertEqual(validated_index[0]["status"], "validated")
            self.assertEqual(validated_index[0]["alpha"], validated_daily["validation"]["alpha"])
            self.assertEqual(len(history), 1)

            second_validation = validate_pipeline(config)
            history_again = pd.read_csv(config.validation_history_path)
            self.assertEqual(second_validation["validated"], 0)
            self.assertEqual(len(history_again), 1)

    def test_validate_backfills_canonical_daily_archive_from_legacy_run_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)

            daily_path = config.paths.predictions_dir / "daily" / f"{as_of_date}.json"
            index_path = config.paths.predictions_dir / "index.json"
            latest_path = config.paths.predictions_dir / "latest.json"

            daily_path.unlink()
            index_path.unlink()
            latest_path.unlink()

            validation = validate_pipeline(config)

            self.assertEqual(validation["validated"], 1)
            self.assertTrue(daily_path.exists())
            backfilled_payload = read_json(daily_path)
            self.assertEqual(backfilled_payload["archive_id"], prediction["archive_id"])
            self.assertEqual(backfilled_payload["status"], "validated")

    def test_validate_pipeline_uses_prediction_model_contract_when_bundle_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)
            expected_return = realized_day_lookup(bundle, as_of_date)[prediction["selected_code"]]

            if config.bundle_path.exists():
                config.bundle_path.unlink()
            config.data.cache_version = "mutated_after_prediction"
            config.data.entry_offset_days = 0
            config.data.exit_offset_days = 1
            config.data.target_clip = (-0.01, 0.01)

            validation = validate_pipeline(config)
            daily_payload = read_json(config.paths.predictions_dir / "daily" / f"{as_of_date}.json")

            self.assertEqual(validation["validated"], 1)
            self.assertAlmostEqual(daily_payload["validation"]["selected_return"], float(expected_return), places=8)


if __name__ == "__main__":
    unittest.main()
