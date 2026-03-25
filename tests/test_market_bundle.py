from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_impl.data.market import (
    _build_single_stock_feature_frame,
    build_market_cache,
    build_walk_forward_splits,
    canonicalize_raw_frame,
    load_market_bundle,
)

from tests.helpers import make_synthetic_market_parquet, make_test_config


class MarketBundleTest(unittest.TestCase):
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
            self.assertIn("one_word_day1", bundle)
            self.assertEqual(bundle["open_limit_day1"].shape[0], bundle["targets"].shape[0])
            self.assertEqual(bundle["one_word_day1"].shape[0], bundle["targets"].shape[0])
            self.assertGreaterEqual(len(splits), 1)

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
            self.assertEqual(int(as_of_row["one_word_day1"]), 1)

    def test_open_limit_detection_handles_gap_open_limit_that_breaks_intraday(self) -> None:
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

        self.assertEqual(int(as_of_row["open_limit_day1"]), 1)
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
        self.assertEqual(int(as_of_row["one_word_day1"]), 1)


if __name__ == "__main__":
    unittest.main()
