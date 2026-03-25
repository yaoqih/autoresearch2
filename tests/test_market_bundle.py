from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_impl.data.market import (
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
            self.assertGreaterEqual(len(splits), 1)


if __name__ == "__main__":
    unittest.main()
