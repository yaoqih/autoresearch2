from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_impl.data.market import build_market_cache, build_walk_forward_splits, load_market_bundle

from tests.helpers import make_synthetic_market_parquet, make_test_config


class MarketBundleTest(unittest.TestCase):
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
