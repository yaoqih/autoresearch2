from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_impl.settings import AppConfig


def make_test_config(root: Path) -> AppConfig:
    config = AppConfig().resolve()
    config.paths.raw_daily_dir = root / "data" / "daily"
    config.paths.merged_parquet = root / "data" / "market_daily.parquet"
    config.paths.cache_dir = root / "artifacts" / "cache"
    config.paths.models_dir = root / "artifacts" / "models"
    config.paths.predictions_dir = root / "artifacts" / "predictions"
    config.paths.validation_dir = root / "artifacts" / "validation"
    config.paths.logs_dir = root / "artifacts" / "logs"
    config.paths.metrics_dir = root / "artifacts" / "metrics"
    config.paths.ensure()

    config.data.cache_version = "test_quant"
    config.data.min_listed_days = 5
    config.data.min_daily_universe = 4
    config.data.train_days = 20
    config.data.valid_days = 10
    config.data.holdout_days = 10
    config.data.step_days = 10
    config.data.rolling_train = True
    config.data.ret_windows = (2, 3, 5)
    config.data.ma_windows = (3, 5)
    config.data.vol_windows = (3, 5)
    config.data.flow_windows = (3, 5)

    config.model.hidden_dims = (32, 16, 8)
    config.model.dropout = 0.0
    config.model.num_residual_blocks = 1
    config.model.shortlist_size = 4
    config.model.rerank_dim = 8
    config.model.rerank_blocks = 1
    config.model.rerank_heads = 2
    config.model.rerank_mix = 0.25

    config.training.epochs = 2
    config.training.batch_days = 5
    config.training.early_stopping_patience = 2
    config.training.pair_samples_per_day = 8
    config.training.recent_holdout_folds = 2

    config.inference.archive_top_n = 3
    return config


def make_synthetic_market_parquet(path: Path, num_assets: int = 6, num_days: int = 60) -> None:
    rng = np.random.default_rng(7)
    writer = None
    dates = pd.bdate_range("2020-01-01", periods=num_days)
    try:
        for asset_idx in range(num_assets):
            base = 10.0 + asset_idx * 0.7
            trend = np.linspace(0.0, 1.5 + asset_idx * 0.15, num_days)
            seasonal = 0.2 * np.sin(np.arange(num_days) / 4.0 + asset_idx)
            noise = rng.normal(0.0, 0.03, num_days).cumsum() * 0.2
            close = base + trend + seasonal + noise
            open_ = close * (1.0 + rng.normal(0.0, 0.005, num_days))
            high = np.maximum(open_, close) * (1.0 + 0.01 + rng.random(num_days) * 0.01)
            low = np.minimum(open_, close) * (1.0 - 0.01 - rng.random(num_days) * 0.01)
            volume = 5e5 + asset_idx * 1e4 + rng.normal(0.0, 2e4, num_days)
            amount = volume * close * 100
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            pct_chg = (close / prev_close - 1.0) * 100
            change = close - prev_close
            amplitude = (high / low - 1.0) * 100
            turnover_rate = 1.5 + asset_idx * 0.1 + rng.random(num_days) * 0.2
            code = f"SZ{asset_idx + 1:06d}"
            frame = pd.DataFrame(
                {
                    "code": code,
                    "date": dates,
                    "open": open_,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "amount": amount,
                    "amplitude": amplitude,
                    "pct_chg": pct_chg,
                    "change": change,
                    "turnover_rate": turnover_rate,
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
