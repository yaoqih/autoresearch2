from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import pandas as pd
import torch
from unittest.mock import patch

from quant_impl.data.market import load_market_bundle, realized_day_detail_lookup, realized_day_lookup
from quant_impl.pipelines.predict_history import predict_history_pipeline
from quant_impl.pipelines.predict import predict_pipeline
from quant_impl.pipelines.train import train_pipeline
from quant_impl.pipelines.validate import validate_pipeline
from quant_impl.utils.io import read_json

from tests.helpers import make_synthetic_market_parquet, make_test_config


class PredictionValidationTest(unittest.TestCase):
    def test_predict_pipeline_uses_training_contract_for_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            as_of_index = bundle["dates"].index(as_of_date)

            config.data.entry_offset_days = 4
            config.data.exit_offset_days = 6
            config.data.ret_windows = (5, 3, 2)

            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)

            self.assertEqual(prediction["entry_date"], bundle["dates"][as_of_index + 1])
            self.assertEqual(prediction["exit_date"], bundle["dates"][as_of_index + 2])

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
            self.assertIn("executed_code", validated_daily["validation"])
            self.assertIn("executed_rank", validated_daily["validation"])
            self.assertIn("fallback_window_size", validated_daily["validation"])
            self.assertIn("all_fallback_blocked", validated_daily["validation"])
            self.assertIn("schema_version", validated_daily["validation"])
            self.assertIn("validation", validated_daily["top_candidates"][0])
            self.assertIn("strict_open_return", validated_daily["top_candidates"][0]["validation"])
            self.assertIn("tradeable", validated_daily["top_candidates"][0]["validation"])
            self.assertIn("executed", validated_daily["top_candidates"][0]["validation"])
            self.assertEqual(validated_index[0]["status"], "validated")
            self.assertEqual(validated_index[0]["alpha"], validated_daily["validation"]["alpha"])
            self.assertIn("executed_code", validated_index[0])
            self.assertIn("executed_rank", validated_index[0])
            self.assertIn("fallback_window_size", validated_index[0])
            self.assertIn("all_fallback_blocked", validated_index[0])
            self.assertIn("executed_code", history.columns)
            self.assertIn("executed_rank", history.columns)
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

    def test_predict_history_pipeline_backfills_validation_for_multiple_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            start_date = bundle["dates"][-6]
            end_date = bundle["dates"][-5]

            result = predict_history_pipeline(
                config,
                device="cpu",
                start_date=start_date,
                end_date=end_date,
                validate=True,
            )

            self.assertEqual(result["predicted"], 2)
            self.assertEqual(result["dates"], [start_date, end_date])
            self.assertEqual(result["validation"]["validated"], 2)

            first_daily = read_json(config.paths.predictions_dir / "daily" / f"{start_date}.json")
            second_daily = read_json(config.paths.predictions_dir / "daily" / f"{end_date}.json")
            history = pd.read_csv(config.validation_history_path)

            self.assertEqual(first_daily["status"], "validated")
            self.assertEqual(second_daily["status"], "validated")
            self.assertEqual(len(history), 2)

    def test_validate_pipeline_falls_back_to_next_tradeable_top_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.inference.archive_top_n = 10
            config.inference.execution_fallback_top_k = 10
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=12, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)
            realized_map = realized_day_detail_lookup(bundle, as_of_date)
            self.assertIsNotNone(realized_map)
            top_candidates = prediction["top_candidates"]
            top1 = top_candidates[0]["code"]
            top2 = top_candidates[1]["code"]
            modified_map = {code: dict(values) for code, values in realized_map.items()}
            modified_map[top1]["open_limit_day1"] = True
            modified_map[top1]["strict_open_return"] = 0.0
            modified_map[top2]["open_limit_day1"] = False
            modified_map[top2]["strict_open_return"] = 0.03125
            modified_map[top2]["ideal_return"] = 0.03125

            with patch("quant_impl.pipelines.validate.realized_day_detail_lookup", return_value=modified_map):
                validation = validate_pipeline(config)

            validated_daily = read_json(config.paths.predictions_dir / "daily" / f"{as_of_date}.json")
            self.assertEqual(validation["validated"], 1)
            self.assertEqual(validated_daily["validation"]["executed_code"], top2)
            self.assertEqual(validated_daily["validation"]["executed_rank"], 2)
            self.assertEqual(validated_daily["validation"]["fallback_applied"], 1)
            self.assertEqual(validated_daily["validation"]["fallback_window_size"], 10)
            self.assertAlmostEqual(validated_daily["validation"]["selected_return"], 0.03125, places=8)
            self.assertFalse(validated_daily["top_candidates"][0]["validation"]["executed"])
            self.assertTrue(validated_daily["top_candidates"][1]["validation"]["executed"])

    def test_validate_pipeline_sets_zero_return_when_top10_all_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.inference.archive_top_n = 10
            config.inference.execution_fallback_top_k = 10
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=12, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)
            realized_map = realized_day_detail_lookup(bundle, as_of_date)
            self.assertIsNotNone(realized_map)
            modified_map = {code: dict(values) for code, values in realized_map.items()}

            for item in prediction["top_candidates"]:
                modified_map[item["code"]]["open_limit_day1"] = True
                modified_map[item["code"]]["strict_open_return"] = 0.0

            with patch("quant_impl.pipelines.validate.realized_day_detail_lookup", return_value=modified_map):
                validation = validate_pipeline(config)

            validated_daily = read_json(config.paths.predictions_dir / "daily" / f"{as_of_date}.json")
            self.assertEqual(validation["validated"], 1)
            self.assertEqual(validated_daily["validation"]["all_top10_blocked"], 1)
            self.assertEqual(validated_daily["validation"]["all_fallback_blocked"], 1)
            self.assertEqual(validated_daily["validation"]["fallback_window_size"], 10)
            self.assertEqual(validated_daily["validation"]["selected_return"], 0.0)
            self.assertIsNone(validated_daily["validation"]["executed_code"])
            self.assertEqual(validated_daily["summary"]["selected_return"], 0.0)

    def test_validate_pipeline_honors_configured_fallback_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.inference.archive_top_n = 5
            config.inference.execution_fallback_top_k = 3
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=12, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)
            bundle = load_market_bundle(config)
            as_of_date = bundle["dates"][-5]
            prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)
            realized_map = realized_day_detail_lookup(bundle, as_of_date)
            self.assertIsNotNone(realized_map)
            modified_map = {code: dict(values) for code, values in realized_map.items()}

            for item in prediction["top_candidates"][:2]:
                modified_map[item["code"]]["open_limit_day1"] = True
                modified_map[item["code"]]["strict_open_return"] = 0.0

            rank3 = prediction["top_candidates"][2]["code"]
            modified_map[rank3]["open_limit_day1"] = False
            modified_map[rank3]["strict_open_return"] = 0.0625
            modified_map[rank3]["ideal_return"] = 0.0625

            with patch("quant_impl.pipelines.validate.realized_day_detail_lookup", return_value=modified_map):
                validate_pipeline(config)

            validated_daily = read_json(config.paths.predictions_dir / "daily" / f"{as_of_date}.json")
            self.assertEqual(validated_daily["validation"]["executed_rank"], 3)
            self.assertEqual(validated_daily["validation"]["fallback_window_size"], 3)

    def test_predict_pipeline_uses_local_market_dates_for_recent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            train_pipeline(config, device="cpu", profile="screen", force_prepare=True)

            merged = pd.read_parquet(config.paths.merged_parquet, columns=["date"])
            raw_dates = sorted(pd.to_datetime(merged["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
            as_of_date = raw_dates[-2]
            expected_entry_date = raw_dates[-1]
            expected_exit_date = (pd.Timestamp(expected_entry_date) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

            with patch("quant_impl.pipelines.predict.TradingCalendar", side_effect=RuntimeError("network calendar disabled")):
                prediction = predict_pipeline(config, device="cpu", as_of_date=as_of_date)

            self.assertEqual(prediction["entry_date"], expected_entry_date)
            self.assertEqual(prediction["exit_date"], expected_exit_date)


if __name__ == "__main__":
    unittest.main()
