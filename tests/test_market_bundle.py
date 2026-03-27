from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from quant_impl.data.market import (
    _build_single_stock_feature_frame,
    build_market_cache,
    build_walk_forward_splits,
    canonicalize_raw_frame,
    evaluate_ranker,
    load_market_bundle,
    realized_day_detail_lookup,
)

from tests.helpers import make_synthetic_market_parquet, make_test_config


class MarketBundleTest(unittest.TestCase):
    def test_evaluate_ranker_rolls_to_first_fillable_name(self) -> None:
        class DummyModel:
            def eval(self):
                return self

            def forward_components(self, features):
                size = features.shape[0]
                return {
                    "broad_score": torch.zeros(size, dtype=torch.float32),
                    "linear_score": torch.zeros(size, dtype=torch.float32),
                    "rerank_latent": torch.zeros(size, 16, dtype=torch.float32),
                }

        bundle = {
            "features": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
            "targets": torch.tensor([0.10, 0.03, -0.02], dtype=torch.float32),
            "asset_ids": torch.tensor([0, 1, 2], dtype=torch.int32),
            "day_ptr": torch.tensor([0, 3], dtype=torch.int64),
            "dates": ["2020-01-01"],
            "open_limit_day1": torch.tensor([1, 0, 0], dtype=torch.uint8),
            "open_limit_day1_current": torch.tensor([1, 0, 0], dtype=torch.uint8),
            "open_limit_day1_exact": torch.tensor([1, 0, 0], dtype=torch.uint8),
            "open_limit_day1_hybrid": torch.tensor([1, 0, 0], dtype=torch.uint8),
            "one_word_day1": torch.tensor([1, 0, 0], dtype=torch.uint8),
        }
        config = make_test_config(Path("."))
        dummy_model = DummyModel()

        with patch(
            "quant_impl.modeling.ranker.rerank_shortlist_scores",
            return_value=(torch.tensor([0.9, 0.8, 0.1], dtype=torch.float32), []),
        ):
            report = evaluate_ranker(
                dummy_model,
                bundle,
                config.data,
                [0],
                torch.device("cpu"),
                fallback_top_k=3,
                batch_days=1,
            )

        self.assertAlmostEqual(report["metrics"]["mean_return"], 0.03, places=8)
        self.assertAlmostEqual(report["trade_rate"], 1.0, places=8)
        self.assertAlmostEqual(report["block_rate_open_limit"], 1.0, places=8)
        self.assertAlmostEqual(report["all_fallback_blocked_rate"], 0.0, places=8)
        self.assertAlmostEqual(report["switch_rate"], 1.0, places=8)
        self.assertAlmostEqual(report["mean_selected_rank_when_traded"], 2.0, places=8)
        self.assertAlmostEqual(report["supplemental"]["ideal_metrics"]["mean_return"], 0.10, places=8)
        self.assertEqual(report["execution_fallback_top_k"], 3)
        self.assertEqual(report["execution_block_mode"], "hybrid")

    def test_evaluate_ranker_returns_cash_when_all_fallback_candidates_blocked(self) -> None:
        class DummyModel:
            def eval(self):
                return self

            def forward_components(self, features):
                size = features.shape[0]
                return {
                    "broad_score": torch.zeros(size, dtype=torch.float32),
                    "linear_score": torch.zeros(size, dtype=torch.float32),
                    "rerank_latent": torch.zeros(size, 16, dtype=torch.float32),
                }

        bundle = {
            "features": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
            "targets": torch.tensor([0.10, 0.03, -0.02], dtype=torch.float32),
            "asset_ids": torch.tensor([0, 1, 2], dtype=torch.int32),
            "day_ptr": torch.tensor([0, 3], dtype=torch.int64),
            "dates": ["2020-01-01"],
            "open_limit_day1": torch.tensor([1, 1, 1], dtype=torch.uint8),
            "open_limit_day1_current": torch.tensor([1, 1, 1], dtype=torch.uint8),
            "open_limit_day1_exact": torch.tensor([1, 1, 1], dtype=torch.uint8),
            "open_limit_day1_hybrid": torch.tensor([1, 1, 1], dtype=torch.uint8),
            "one_word_day1": torch.tensor([1, 1, 1], dtype=torch.uint8),
        }
        config = make_test_config(Path("."))
        dummy_model = DummyModel()

        with patch(
            "quant_impl.modeling.ranker.rerank_shortlist_scores",
            return_value=(torch.tensor([0.9, 0.8, 0.1], dtype=torch.float32), []),
        ):
            report = evaluate_ranker(
                dummy_model,
                bundle,
                config.data,
                [0],
                torch.device("cpu"),
                fallback_top_k=3,
                batch_days=1,
            )

        self.assertAlmostEqual(report["metrics"]["mean_return"], 0.0, places=8)
        self.assertAlmostEqual(report["trade_rate"], 0.0, places=8)
        self.assertAlmostEqual(report["block_rate_open_limit"], 1.0, places=8)
        self.assertAlmostEqual(report["all_fallback_blocked_rate"], 1.0, places=8)
        self.assertAlmostEqual(report["switch_rate"], 0.0, places=8)
        self.assertAlmostEqual(report["mean_selected_rank_when_traded"], 0.0, places=8)

    def test_canonicalize_raw_frame_handles_amount_and_money_without_duplicate_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2026-03-24"],
                "open": [10.0],
                "close": [10.5],
                "high": [10.8],
                "low": [9.9],
                "volume": [1000.0],
                "amount": [10500.0],
                "money": [10500.0],
                "amplitude": [0.09],
                "pct_chg": [0.05],
                "change": [0.5],
                "turnover_rate": [1.2],
                "symbol": ["000001.SZ"],
                "factor": [1.0],
            }
        )

        normalized = canonicalize_raw_frame(frame, "sz000001")

        self.assertEqual(list(normalized.columns).count("amount"), 1)
        self.assertEqual(float(normalized["amount"].iloc[0]), 10500.0)

    def test_build_bundle_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            build_market_cache(config, force=True)
            bundle = load_market_bundle(config)
            splits = build_walk_forward_splits(bundle, config.data)
            self.assertGreater(bundle["features"].shape[0], 0)
            self.assertEqual(len(bundle["feature_names"]), len(bundle["feature_names"]))
            self.assertIn("open_limit_day1", bundle)
            self.assertIn("open_limit_day1_current", bundle)
            self.assertIn("open_limit_day1_exact", bundle)
            self.assertIn("open_limit_day1_hybrid", bundle)
            self.assertIn("one_word_day1", bundle)
            self.assertEqual(bundle["open_limit_day1"].shape[0], bundle["targets"].shape[0])
            self.assertEqual(bundle["open_limit_day1_current"].shape[0], bundle["targets"].shape[0])
            self.assertEqual(bundle["open_limit_day1_exact"].shape[0], bundle["targets"].shape[0])
            self.assertEqual(bundle["open_limit_day1_hybrid"].shape[0], bundle["targets"].shape[0])
            self.assertEqual(bundle["one_word_day1"].shape[0], bundle["targets"].shape[0])
            self.assertTrue((bundle["open_limit_day1"] == bundle["open_limit_day1_hybrid"]).all().item())
            self.assertGreaterEqual(len(splits), 1)

    def test_load_market_bundle_rebuilds_legacy_contract_missing_explicit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            build_market_cache(config, force=True)
            bundle = load_market_bundle(config)

            legacy_bundle = {
                key: value
                for key, value in bundle.items()
                if key not in {"open_limit_day1_current", "open_limit_day1_exact", "open_limit_day1_hybrid"}
            }
            torch.save(legacy_bundle, config.bundle_path)

            rebuilt = load_market_bundle(config)

            self.assertIn("open_limit_day1_current", rebuilt)
            self.assertIn("open_limit_day1_exact", rebuilt)
            self.assertIn("open_limit_day1_hybrid", rebuilt)
            self.assertTrue((rebuilt["open_limit_day1"] == rebuilt["open_limit_day1_hybrid"]).all().item())

    def test_limit_up_detection_handles_adjusted_price_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            config.data.min_listed_days = 5
            dates = pd.bdate_range("2025-11-26", periods=32)
            closes = [20.0 + i * 0.2 for i in range(len(dates))]
            index_0105 = dates.get_loc(pd.Timestamp("2026-01-05"))
            index_0106 = dates.get_loc(pd.Timestamp("2026-01-06"))
            index_0107 = dates.get_loc(pd.Timestamp("2026-01-07"))
            index_0108 = dates.get_loc(pd.Timestamp("2026-01-08"))
            closes[index_0105] = 79.26
            closes[index_0106] = 87.08
            closes[index_0107] = 95.68
            closes[index_0108] = 105.13

            rows = []
            prev_close = closes[0]
            for date, close in zip(dates, closes):
                change = close - prev_close
                pct_chg = 0.0 if prev_close == 0 else change / prev_close * 100.0
                rows.append(
                    {
                        "date": date,
                        "open": close,
                        "close": close,
                        "high": close,
                        "low": close,
                        "volume": 1000.0,
                        "amount": 1000.0 * close,
                        "amplitude": 0.0,
                        "pct_chg": pct_chg,
                        "change": change,
                        "turnover_rate": 1.0,
                    }
                )
                prev_close = close
            frame = pd.DataFrame(rows)

        features = _build_single_stock_feature_frame("SZ002931", frame, config.data, include_target=True)
        self.assertIsNotNone(features)
        as_of_row = features.loc[features["date"].dt.strftime("%Y-%m-%d") == "2026-01-06"].iloc[0]

        self.assertEqual(int(as_of_row["open_limit_day1"]), 1)
        self.assertEqual(int(as_of_row["open_limit_day1_current"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_exact"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_hybrid"]), 1)
        self.assertEqual(int(as_of_row["one_word_day1"]), 1)

    def test_open_limit_detection_uses_exact_limit_price_for_gap_open(self) -> None:
        config = make_test_config(Path("."))
        config.data.min_listed_days = 5
        dates = pd.bdate_range("2025-11-10", periods=48)
        closes = [20.0 + i * 0.2 for i in range(len(dates))]
        custom = {
            "2026-01-07": (50.00, 50.00, 50.00, 50.00, 0.00, 0.00),
            "2026-01-08": (54.00, 55.00, 55.00, 52.00, 8.00, 4.00),
        }

        rows = []
        for date, fallback_close in zip(dates, closes):
            date_text = date.strftime("%Y-%m-%d")
            if date_text in custom:
                close, open_, high, low, pct_chg, change = custom[date_text]
            else:
                close = fallback_close
                open_ = close
                high = close
                low = close
                change = 0.0 if not rows else close - rows[-1]["close"]
                prev_close = close - change
                pct_chg = 0.0 if prev_close == 0 else change / prev_close * 100.0
            rows.append(
                {
                    "date": date,
                    "open": open_,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": 1000.0,
                    "amount": 1000.0 * close,
                    "amplitude": 0.0,
                    "pct_chg": pct_chg,
                    "change": change,
                    "turnover_rate": 1.0,
                }
            )
        frame = pd.DataFrame(rows)

        features = _build_single_stock_feature_frame("SZ002519", frame, config.data, include_target=True)
        self.assertIsNotNone(features)
        as_of_row = features.loc[features["date"].dt.strftime("%Y-%m-%d") == "2026-01-07"].iloc[0]

        self.assertEqual(int(as_of_row["open_limit_day1"]), 1)
        self.assertEqual(int(as_of_row["open_limit_day1_current"]), 1)
        self.assertEqual(int(as_of_row["open_limit_day1_exact"]), 1)
        self.assertEqual(int(as_of_row["open_limit_day1_hybrid"]), 1)
        self.assertEqual(int(as_of_row["one_word_day1"]), 0)

    def test_open_limit_detection_current_can_exceed_exact_on_sub_tick_gap(self) -> None:
        config = make_test_config(Path("."))
        config.data.min_listed_days = 5
        dates = pd.bdate_range("2025-11-10", periods=48)
        closes = [20.0 + i * 0.2 for i in range(len(dates))]
        custom = {
            "2026-01-07": (50.00, 50.00, 50.00, 50.00, 0.00, 0.00),
            "2026-01-08": (54.50, 54.99, 55.00, 53.50, 9.00, 4.50),
        }

        rows = []
        for date, fallback_close in zip(dates, closes):
            date_text = date.strftime("%Y-%m-%d")
            if date_text in custom:
                close, open_, high, low, pct_chg, change = custom[date_text]
            else:
                close = fallback_close
                open_ = close
                high = close
                low = close
                change = 0.0 if not rows else close - rows[-1]["close"]
                prev_close = close - change
                pct_chg = 0.0 if prev_close == 0 else change / prev_close * 100.0
            rows.append(
                {
                    "date": date,
                    "open": open_,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": 1000.0,
                    "amount": 1000.0 * close,
                    "amplitude": 0.0,
                    "pct_chg": pct_chg,
                    "change": change,
                    "turnover_rate": 1.0,
                }
            )
        frame = pd.DataFrame(rows)

        features = _build_single_stock_feature_frame("SZ002519", frame, config.data, include_target=True)
        self.assertIsNotNone(features)
        as_of_row = features.loc[features["date"].dt.strftime("%Y-%m-%d") == "2026-01-07"].iloc[0]

        self.assertEqual(int(as_of_row["open_limit_day1"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_current"]), 1)
        self.assertEqual(int(as_of_row["open_limit_day1_exact"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_hybrid"]), 0)
        self.assertEqual(int(as_of_row["one_word_day1"]), 0)

    def test_open_limit_detection_does_not_block_sub_limit_gap_high_days(self) -> None:
        config = make_test_config(Path("."))
        config.data.min_listed_days = 5
        dates = pd.bdate_range("2025-11-10", periods=48)
        closes = [20.0 + i * 0.2 for i in range(len(dates))]
        custom = {
            "2026-01-05": (81.87, 81.87, 81.87, 76.77, 8.81, 6.63),
            "2026-01-06": (89.11, 89.11, 89.11, 87.27, 8.84, 7.24),
            "2026-01-07": (97.06, 97.06, 97.06, 97.06, 8.92, 7.95),
            "2026-01-08": (105.84, 105.84, 105.84, 99.00, 9.05, 8.78),
            "2026-01-09": (115.53, 115.53, 115.53, 111.75, 9.16, 9.69),
        }

        rows = []
        for date, fallback_close in zip(dates, closes):
            date_text = date.strftime("%Y-%m-%d")
            if date_text in custom:
                close, open_, high, low, pct_chg, change = custom[date_text]
            else:
                close = fallback_close
                open_ = close
                high = close
                low = close
                change = 0.0 if not rows else close - rows[-1]["close"]
                prev_close = close - change
                pct_chg = 0.0 if prev_close == 0 else change / prev_close * 100.0
            rows.append(
                {
                    "date": date,
                    "open": open_,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": 1000.0,
                    "amount": 1000.0 * close,
                    "amplitude": 0.0,
                    "pct_chg": pct_chg,
                    "change": change,
                    "turnover_rate": 1.0,
                }
            )
        frame = pd.DataFrame(rows)

        features = _build_single_stock_feature_frame("SZ002519", frame, config.data, include_target=True)
        self.assertIsNotNone(features)
        as_of_row = features.loc[features["date"].dt.strftime("%Y-%m-%d") == "2026-01-07"].iloc[0]

        self.assertEqual(int(as_of_row["open_limit_day1"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_current"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_exact"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_hybrid"]), 0)
        self.assertEqual(int(as_of_row["one_word_day1"]), 0)

    def test_one_word_limit_detection_handles_sub_10_percent_adjusted_series(self) -> None:
        config = make_test_config(Path("."))
        config.data.min_listed_days = 5
        dates = pd.bdate_range("2025-12-01", periods=48)
        closes = [30.0 + i * 0.2 for i in range(len(dates))]
        custom = {
            "2026-01-26": (86.51, 86.51, 86.51, 86.51, 9.66, 7.62),
            "2026-01-27": (94.89, 94.89, 94.89, 94.89, 9.69, 8.38),
            "2026-01-28": (104.10, 104.10, 104.10, 104.10, 9.71, 9.21),
            "2026-01-29": (114.23, 114.23, 114.23, 114.23, 9.73, 10.13),
        }

        rows = []
        for date, fallback_close in zip(dates, closes):
            date_text = date.strftime("%Y-%m-%d")
            if date_text in custom:
                close, open_, high, low, pct_chg, change = custom[date_text]
            else:
                close = fallback_close
                open_ = close
                high = close
                low = close
                change = 0.0 if not rows else close - rows[-1]["close"]
                prev_close = close - change
                pct_chg = 0.0 if prev_close == 0 else change / prev_close * 100.0
            rows.append(
                {
                    "date": date,
                    "open": open_,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": 1000.0,
                    "amount": 1000.0 * close,
                    "amplitude": 0.0,
                    "pct_chg": pct_chg,
                    "change": change,
                    "turnover_rate": 1.0,
                }
            )
        frame = pd.DataFrame(rows)

        features = _build_single_stock_feature_frame("SZ002155", frame, config.data, include_target=True)
        self.assertIsNotNone(features)
        as_of_row = features.loc[features["date"].dt.strftime("%Y-%m-%d") == "2026-01-27"].iloc[0]

        self.assertEqual(int(as_of_row["open_limit_day1"]), 1)
        self.assertEqual(int(as_of_row["open_limit_day1_current"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_exact"]), 0)
        self.assertEqual(int(as_of_row["open_limit_day1_hybrid"]), 1)
        self.assertEqual(int(as_of_row["one_word_day1"]), 1)

    def test_realized_day_detail_lookup_exposes_execution_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_test_config(root)
            make_synthetic_market_parquet(config.paths.merged_parquet, num_assets=6, num_days=70)
            build_market_cache(config, force=True)
            bundle = load_market_bundle(config)
            detail_map = realized_day_detail_lookup(bundle, bundle["dates"][-5])

            self.assertIsNotNone(detail_map)
            sample = next(iter(detail_map.values()))
            self.assertIn("open_limit_day1", sample)
            self.assertIn("open_limit_day1_current", sample)
            self.assertIn("open_limit_day1_exact", sample)
            self.assertIn("open_limit_day1_hybrid", sample)
            self.assertIn("strict_open_return", sample)
            self.assertIn("strict_open_return_current", sample)
            self.assertIn("strict_open_return_exact", sample)
            self.assertIn("strict_open_return_hybrid", sample)
            self.assertEqual(sample["open_limit_day1"], sample["open_limit_day1_hybrid"])
            self.assertEqual(sample["strict_open_return"], sample["strict_open_return_hybrid"])


if __name__ == "__main__":
    unittest.main()
