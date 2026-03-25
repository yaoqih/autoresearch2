from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from quant_impl.settings import AppConfig, DataSettings, TrainingSettings
from quant_impl.utils.io import write_json


LOGGER = logging.getLogger(__name__)

BASE_COLUMNS = (
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "change",
    "turnover_rate",
)
MERGED_COLUMNS = ("code", *BASE_COLUMNS)
NUMERIC_COLUMNS = tuple(column for column in BASE_COLUMNS if column != "date")
TARGET_NAME = "open_t_plus_2_vs_open_t_plus_1"
PRICE_TOL = 0.011


@dataclass(frozen=True)
class WalkForwardSplit:
    fold_id: int
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int
    holdout_start: int
    holdout_end: int
    train_start_date: str
    train_end_date: str
    valid_start_date: str
    valid_end_date: str
    holdout_start_date: str
    holdout_end_date: str


def round_half_up(value: float, decimals: int = 2) -> float:
    quant = "1." + ("0" * decimals)
    return float(Decimal(str(value)).quantize(Decimal(quant), rounding=ROUND_HALF_UP))


def _code_digits(code: str) -> str:
    return "".join(ch for ch in str(code) if ch.isdigit())


def limit_pct_for_date(code: str, date_str: str) -> float:
    digits = _code_digits(code)
    if digits.startswith("688"):
        return 0.20
    if digits.startswith(("300", "301")):
        return 0.20 if date_str >= "2020-08-24" else 0.10
    if digits.startswith(("8", "4", "920")) and date_str >= "2021-11-15":
        return 0.30
    return 0.10


def feature_columns(data_cfg: DataSettings) -> list[str]:
    return [
        "gap_prev_close",
        "intraday_return",
        "high_close_spread",
        "close_low_spread",
        "day_range",
        "amplitude_pct",
        "pct_chg_pct",
        *[f"ret_{window}" for window in data_cfg.ret_windows],
        *[f"ma_gap_{window}" for window in data_cfg.ma_windows],
        *[f"volatility_{window}" for window in data_cfg.vol_windows],
        *[f"volume_ratio_{window}" for window in data_cfg.flow_windows],
        *[f"amount_ratio_{window}" for window in data_cfg.flow_windows],
        *[f"turnover_ratio_{window}" for window in data_cfg.flow_windows],
        "price_position_20",
        "momentum_spread_5_20",
        "volume_momentum_5",
        "turnover_volatility_20",
    ]


def resolve_merged_path(config: AppConfig) -> Path:
    if config.paths.merged_parquet.exists():
        return config.paths.merged_parquet
    if config.paths.reference_merged_parquet and config.paths.reference_merged_parquet.exists():
        return config.paths.reference_merged_parquet
    raise FileNotFoundError(
        f"Missing merged parquet. Expected {config.paths.merged_parquet}"
        + (
            f" or {config.paths.reference_merged_parquet}"
            if config.paths.reference_merged_parquet
            else ""
        )
    )


def canonical_code(symbol: str) -> str:
    token = str(symbol).strip().upper()
    if "." in token:
        left, right = token.split(".", 1)
        if left in {"SH", "SZ", "BJ"}:
            market, code = left, right
        else:
            code, market = left, right
    elif token.startswith(("SH", "SZ", "BJ")):
        market, code = token[:2], token[2:]
    else:
        code = "".join(ch for ch in token if ch.isdigit())
        if code.startswith("6"):
            market = "SH"
        elif code.startswith(("4", "8")):
            market = "BJ"
        else:
            market = "SZ"
    return f"{market}{code.zfill(6)}"


def canonicalize_raw_frame(frame: pd.DataFrame, fallback_symbol: str) -> pd.DataFrame:
    working = frame.copy()
    if "symbol" in working.columns:
        symbol = str(working["symbol"].iloc[0])
    else:
        symbol = fallback_symbol
    rename_map = {
        "pct_change": "pct_chg",
        "turnover": "turnover_rate",
    }
    if "money" in working.columns:
        if "amount" in working.columns:
            working = working.drop(columns=["money"])
        else:
            rename_map["money"] = "amount"
    working = working.rename(columns=rename_map)
    missing = [column for column in BASE_COLUMNS if column not in working.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {fallback_symbol}")
    working["code"] = canonical_code(symbol or fallback_symbol)
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.loc[:, MERGED_COLUMNS].dropna(subset=["date", "open", "close"]).copy()
    working.sort_values("date", inplace=True)
    return working.reset_index(drop=True)


def merge_daily_parquets(config: AppConfig, force: bool = True) -> dict[str, int | str]:
    source_dir = config.paths.raw_daily_dir
    output_path = config.paths.merged_parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        LOGGER.info("Merged parquet already exists, skipping rebuild path=%s", output_path)
        return {"symbols": 0, "rows": 0, "path": str(output_path)}

    file_paths = sorted(source_dir.glob("*.parquet"))
    if not file_paths:
        raise RuntimeError(f"No raw parquet files found in {source_dir}")
    LOGGER.info(
        "Merging raw parquet files source_dir=%s output_path=%s files=%s force=%s",
        source_dir,
        output_path,
        len(file_paths),
        force,
    )
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_symbols = 0
    try:
        for index, path in enumerate(file_paths, start=1):
            frame = pd.read_parquet(path)
            normalized = canonicalize_raw_frame(frame, path.stem)
            if normalized.empty:
                LOGGER.debug("Skipping empty normalized parquet path=%s", path)
                continue
            table = pa.Table.from_pandas(normalized, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(normalized)
            total_symbols += 1
            if index % 200 == 0 or index == len(file_paths):
                LOGGER.info(
                    "Merge progress processed_files=%s/%s merged_symbols=%s merged_rows=%s",
                    index,
                    len(file_paths),
                    total_symbols,
                    total_rows,
                )
    finally:
        if writer is not None:
            writer.close()
    LOGGER.info("Merged parquet ready path=%s symbols=%s rows=%s", output_path, total_symbols, total_rows)
    return {"symbols": total_symbols, "rows": total_rows, "path": str(output_path)}


def _iter_stock_frames(merged_path: Path, limit_stocks: int | None = None):
    parquet_file = pq.ParquetFile(merged_path)
    total = parquet_file.num_row_groups
    end = total if limit_stocks is None else min(total, limit_stocks)
    for index in range(end):
        frame = parquet_file.read_row_group(index, columns=list(MERGED_COLUMNS)).to_pandas()
        if frame.empty:
            continue
        code = str(frame["code"].iloc[0])
        yield index + 1, end, code, frame.loc[:, BASE_COLUMNS]


def _build_single_stock_feature_frame(
    code: str,
    df: pd.DataFrame,
    data_cfg: DataSettings,
    *,
    include_target: bool,
) -> pd.DataFrame | None:
    if df.empty:
        return None
    feature_names = feature_columns(data_cfg)
    frame = df.loc[:, BASE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(BASE_COLUMNS)).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return None

    valid_price_mask = (
        (frame["open"] > 0)
        & (frame["close"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["volume"] > 0)
        & (frame["amount"] > 0)
    )
    frame = frame.loc[valid_price_mask].reset_index(drop=True)
    min_rows = (
        data_cfg.min_listed_days
        + max(max(data_cfg.ret_windows), max(data_cfg.ma_windows), max(data_cfg.vol_windows), 20)
        + data_cfg.exit_offset_days
    )
    if len(frame) < min_rows:
        return None

    open_price = frame["open"].astype(float)
    close_price = frame["close"].astype(float)
    high_price = frame["high"].astype(float)
    low_price = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    amount = frame["amount"].astype(float)
    turnover = frame["turnover_rate"].astype(float)

    prev_close = close_price.shift(1)
    ret_1 = close_price.pct_change()
    day_range = high_price / low_price - 1.0

    feature_map: dict[str, pd.Series] = {
        "gap_prev_close": open_price / prev_close - 1.0,
        "intraday_return": close_price / open_price - 1.0,
        "high_close_spread": high_price / close_price - 1.0,
        "close_low_spread": close_price / low_price - 1.0,
        "day_range": day_range,
        "amplitude_pct": frame["amplitude"].astype(float) / 100.0,
        "pct_chg_pct": frame["pct_chg"].astype(float) / 100.0,
    }
    for window in data_cfg.ret_windows:
        feature_map[f"ret_{window}"] = close_price.pct_change(window)
    for window in data_cfg.ma_windows:
        ma = close_price.rolling(window).mean()
        feature_map[f"ma_gap_{window}"] = close_price / ma - 1.0
    for window in data_cfg.vol_windows:
        feature_map[f"volatility_{window}"] = ret_1.rolling(window).std()
    for window in data_cfg.flow_windows:
        feature_map[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean() - 1.0
        feature_map[f"amount_ratio_{window}"] = amount / amount.rolling(window).mean() - 1.0
        feature_map[f"turnover_ratio_{window}"] = turnover / turnover.rolling(window).mean() - 1.0

    rolling_high_20 = high_price.rolling(20).max()
    rolling_low_20 = low_price.rolling(20).min()
    feature_map["price_position_20"] = (
        (close_price - rolling_low_20) / (rolling_high_20 - rolling_low_20 + 1e-6)
    )
    short_ret_window = "ret_5" if "ret_5" in feature_map else f"ret_{data_cfg.ret_windows[0]}"
    long_ret_window = "ret_20" if "ret_20" in feature_map else f"ret_{data_cfg.ret_windows[-1]}"
    flow_window = "volume_ratio_20" if "volume_ratio_20" in feature_map else f"volume_ratio_{data_cfg.flow_windows[-1]}"
    turnover_window = "turnover_ratio_20" if "turnover_ratio_20" in feature_map else f"turnover_ratio_{data_cfg.flow_windows[-1]}"
    volatility_window = "volatility_20" if "volatility_20" in feature_map else f"volatility_{data_cfg.vol_windows[-1]}"
    feature_map["momentum_spread_5_20"] = feature_map[short_ret_window] - feature_map[long_ret_window]
    feature_map["volume_momentum_5"] = feature_map[short_ret_window] * feature_map[flow_window]
    feature_map["turnover_volatility_20"] = feature_map[turnover_window] * feature_map[volatility_window]

    result = pd.DataFrame({name: feature_map[name] for name in feature_names})
    result["date"] = frame["date"]
    result["code"] = code
    result["listed_days"] = np.arange(len(result), dtype=np.int32)
    if include_target:
        target = open_price.shift(-data_cfg.exit_offset_days) / open_price.shift(-data_cfg.entry_offset_days) - 1.0
        result[TARGET_NAME] = target
    result = result.replace([np.inf, -np.inf], np.nan)

    valid_mask = result["listed_days"] >= data_cfg.min_listed_days
    valid_mask &= result[feature_names].notna().all(axis=1)
    if include_target:
        valid_mask &= result[TARGET_NAME].notna()
        open_limit_day1 = np.zeros(len(result), dtype=np.uint8)
        one_word_day1 = np.zeros(len(result), dtype=np.uint8)
        day_strings = frame["date"].dt.strftime("%Y-%m-%d").tolist()
        open_values = open_price.to_numpy(dtype=np.float64)
        high_values = high_price.to_numpy(dtype=np.float64)
        low_values = low_price.to_numpy(dtype=np.float64)
        close_values = close_price.to_numpy(dtype=np.float64)
        for row_index in np.flatnonzero(valid_mask.to_numpy(dtype=bool)):
            trade_index = row_index + data_cfg.entry_offset_days
            if trade_index >= len(frame):
                continue
            trade_date = day_strings[trade_index]
            prev_close = float(close_values[trade_index - 1])
            upper_price = round_half_up(prev_close * (1.0 + limit_pct_for_date(code, trade_date)))
            open_at_entry = float(open_values[trade_index])
            high_at_entry = float(high_values[trade_index])
            low_at_entry = float(low_values[trade_index])
            close_at_entry = float(close_values[trade_index])
            is_open_limit = open_at_entry >= upper_price - PRICE_TOL
            is_one_word = (
                is_open_limit
                and abs(high_at_entry - upper_price) <= PRICE_TOL
                and abs(low_at_entry - upper_price) <= PRICE_TOL
                and abs(close_at_entry - upper_price) <= PRICE_TOL
            )
            open_limit_day1[row_index] = 1 if is_open_limit else 0
            one_word_day1[row_index] = 1 if is_one_word else 0
        result["open_limit_day1"] = open_limit_day1
        result["one_word_day1"] = one_word_day1
    columns = ["code", "date", *feature_names]
    if include_target:
        columns.extend([TARGET_NAME, "open_limit_day1", "one_word_day1"])
    return result.loc[valid_mask, columns].reset_index(drop=True)


def _day_values_to_strings(day_values: np.ndarray) -> list[str]:
    date_values = day_values.astype("datetime64[D]")
    return [np.datetime_as_string(value, unit="D") for value in date_values]


def build_market_cache(config: AppConfig, force: bool = False, limit_stocks: int | None = None) -> Path:
    config.paths.ensure()
    if config.bundle_path.exists() and not force:
        LOGGER.info("Bundle already exists, skipping rebuild path=%s", config.bundle_path)
        return config.bundle_path

    merged_path = resolve_merged_path(config)
    LOGGER.info(
        "Building market bundle merged_path=%s bundle_path=%s limit_stocks=%s force=%s",
        merged_path,
        config.bundle_path,
        limit_stocks,
        force,
    )
    feature_names = feature_columns(config.data)
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    open_limit_blocks: list[np.ndarray] = []
    one_word_blocks: list[np.ndarray] = []
    date_blocks: list[np.ndarray] = []
    asset_blocks: list[np.ndarray] = []
    asset_codes: list[str] = []

    processed_assets = 0
    for index, total, code, frame in _iter_stock_frames(merged_path, limit_stocks=limit_stocks):
        stock_frame = _build_single_stock_feature_frame(code, frame, config.data, include_target=True)
        if stock_frame is None or stock_frame.empty:
            continue
        asset_id = len(asset_codes)
        asset_codes.append(code)
        processed_assets += 1

        features = stock_frame[feature_names].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(features, -config.data.feature_clip, config.data.feature_clip, out=features)

        targets = stock_frame[TARGET_NAME].to_numpy(dtype=np.float32, copy=True)
        np.clip(targets, config.data.target_clip[0], config.data.target_clip[1], out=targets)
        open_limit_day1 = stock_frame["open_limit_day1"].to_numpy(dtype=np.uint8, copy=True)
        one_word_day1 = stock_frame["one_word_day1"].to_numpy(dtype=np.uint8, copy=True)

        dates = (
            stock_frame["date"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("datetime64[D]")
            .astype(np.int32, copy=False)
        )
        feature_blocks.append(features)
        target_blocks.append(targets)
        open_limit_blocks.append(open_limit_day1)
        one_word_blocks.append(one_word_day1)
        date_blocks.append(dates)
        asset_blocks.append(np.full(len(stock_frame), asset_id, dtype=np.int32))
        if index % 200 == 0 or index == total:
            LOGGER.info(
                "Bundle progress processed_assets=%s accepted_assets=%s/%s",
                index,
                processed_assets,
                total,
            )

    if not feature_blocks:
        raise RuntimeError("No usable stock files found after feature engineering.")

    features = np.concatenate(feature_blocks, axis=0)
    targets = np.concatenate(target_blocks, axis=0)
    open_limit_day1 = np.concatenate(open_limit_blocks, axis=0)
    one_word_day1 = np.concatenate(one_word_blocks, axis=0)
    dates = np.concatenate(date_blocks, axis=0)
    asset_ids = np.concatenate(asset_blocks, axis=0)

    order = np.lexsort((asset_ids, dates))
    features = features[order]
    targets = targets[order]
    open_limit_day1 = open_limit_day1[order]
    one_word_day1 = one_word_day1[order]
    dates = dates[order]
    asset_ids = asset_ids[order]

    day_values, counts = np.unique(dates, return_counts=True)
    row_mask = np.ones(len(dates), dtype=bool)
    valid_day_values: list[int] = []
    valid_counts: list[int] = []
    cursor = 0
    for day_value, count in zip(day_values, counts):
        next_cursor = cursor + int(count)
        if count < config.data.min_daily_universe:
            row_mask[cursor:next_cursor] = False
        else:
            valid_day_values.append(int(day_value))
            valid_counts.append(int(count))
        cursor = next_cursor

    if not valid_day_values:
        raise RuntimeError("No trading days remain after enforcing min_daily_universe")

    features = features[row_mask]
    targets = targets[row_mask]
    open_limit_day1 = open_limit_day1[row_mask]
    one_word_day1 = one_word_day1[row_mask]
    asset_ids = asset_ids[row_mask]
    counts = np.asarray(valid_counts, dtype=np.int64)
    day_ptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))

    bundle = {
        "version": config.data.cache_version,
        "target_name": TARGET_NAME,
        "feature_names": feature_names,
        "features": torch.tensor(features, dtype=torch.float32),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "open_limit_day1": torch.tensor(open_limit_day1, dtype=torch.uint8),
        "one_word_day1": torch.tensor(one_word_day1, dtype=torch.uint8),
        "asset_ids": torch.tensor(asset_ids, dtype=torch.int32),
        "day_ptr": torch.tensor(day_ptr, dtype=torch.int64),
        "dates": _day_values_to_strings(np.asarray(valid_day_values, dtype=np.int32)),
        "assets": asset_codes,
        "config": {
            "merged_parquet_path": str(merged_path),
            **asdict(config.data),
        },
    }

    torch.save(bundle, config.bundle_path)
    summary = get_bundle_summary(bundle)
    write_json(config.summary_path, summary)
    LOGGER.info(
        "Market bundle ready path=%s assets=%s rows=%s days=%s summary_path=%s",
        config.bundle_path,
        summary["num_assets"],
        summary["num_rows"],
        summary["num_days"],
        config.summary_path,
    )
    return config.bundle_path


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_market_bundle(config: AppConfig, force: bool = False, limit_stocks: int | None = None):
    if force or not config.bundle_path.exists():
        LOGGER.info(
            "Loading bundle requires rebuild force=%s bundle_exists=%s",
            force,
            config.bundle_path.exists(),
        )
        build_market_cache(config, force=True, limit_stocks=limit_stocks)
    bundle = _torch_load(config.bundle_path)
    if bundle.get("version") != config.data.cache_version or "open_limit_day1" not in bundle or "one_word_day1" not in bundle:
        LOGGER.warning(
            "Bundle contract mismatch bundle_version=%s expected=%s has_open_limit=%s has_one_word=%s, rebuilding",
            bundle.get("version"),
            config.data.cache_version,
            "open_limit_day1" in bundle,
            "one_word_day1" in bundle,
        )
        build_market_cache(config, force=True, limit_stocks=limit_stocks)
        bundle = _torch_load(config.bundle_path)
    LOGGER.info(
        "Loaded market bundle path=%s assets=%s days=%s rows=%s",
        config.bundle_path,
        len(bundle["assets"]),
        len(bundle["dates"]),
        int(bundle["targets"].shape[0]),
    )
    return bundle


def get_bundle_summary(bundle) -> dict[str, object]:
    day_ptr = bundle["day_ptr"]
    daily_counts = day_ptr[1:] - day_ptr[:-1]
    mean_assets_per_day = float(daily_counts.float().mean().item()) if len(daily_counts) else 0.0
    return {
        "version": bundle["version"],
        "target_name": bundle["target_name"],
        "num_assets": len(bundle["assets"]),
        "num_rows": int(bundle["targets"].shape[0]),
        "num_days": len(bundle["dates"]),
        "num_features": len(bundle["feature_names"]),
        "start_date": bundle["dates"][0],
        "end_date": bundle["dates"][-1],
        "mean_assets_per_day": mean_assets_per_day,
    }


def _window_start(train_end: int, data_cfg: DataSettings) -> int:
    if not data_cfg.rolling_train:
        return 0
    return max(0, train_end - data_cfg.train_days)


def build_walk_forward_splits(bundle, data_cfg: DataSettings) -> list[WalkForwardSplit]:
    n_days = len(bundle["dates"])
    min_required = data_cfg.train_days + data_cfg.valid_days + data_cfg.holdout_days
    if n_days < min_required:
        raise RuntimeError(f"Need at least {min_required} labeled days, found {n_days}")
    splits: list[WalkForwardSplit] = []
    fold_id = 0
    train_end = data_cfg.train_days
    while train_end + data_cfg.valid_days + data_cfg.holdout_days <= n_days:
        train_start = _window_start(train_end, data_cfg)
        valid_start = train_end
        valid_end = valid_start + data_cfg.valid_days
        holdout_start = valid_end
        holdout_end = holdout_start + data_cfg.holdout_days
        splits.append(
            WalkForwardSplit(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
                train_start_date=bundle["dates"][train_start],
                train_end_date=bundle["dates"][train_end - 1],
                valid_start_date=bundle["dates"][valid_start],
                valid_end_date=bundle["dates"][valid_end - 1],
                holdout_start_date=bundle["dates"][holdout_start],
                holdout_end_date=bundle["dates"][holdout_end - 1],
            )
        )
        fold_id += 1
        train_end += data_cfg.step_days
    return splits


def get_day_slice(bundle, day_index: int) -> tuple[int, int]:
    start = int(bundle["day_ptr"][day_index].item())
    end = int(bundle["day_ptr"][day_index + 1].item())
    return start, end


def get_day_data(
    bundle,
    day_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start, end = get_day_slice(bundle, day_index)
    open_limit = bundle.get("open_limit_day1")
    one_word = bundle.get("one_word_day1")
    if open_limit is None:
        open_limit = torch.zeros(end - start, dtype=torch.uint8)
    else:
        open_limit = open_limit[start:end]
    if one_word is None:
        one_word = torch.zeros(end - start, dtype=torch.uint8)
    else:
        one_word = one_word[start:end]
    return (
        bundle["features"][start:end],
        bundle["targets"][start:end],
        bundle["asset_ids"][start:end],
        open_limit,
        one_word,
    )


def normalize_cross_section(features: torch.Tensor, clip_value: float) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.clamp_(-clip_value, clip_value)


def centered_rank(values: torch.Tensor) -> torch.Tensor:
    size = int(values.numel())
    if size <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(size, device=values.device, dtype=torch.float32)
    pct_rank = ranks / float(max(size - 1, 1))
    return 1.0 - 2.0 * pct_rank


def transform_training_targets(
    targets: torch.Tensor,
    group_sizes: Sequence[int],
    training_cfg: TrainingSettings,
    blocked_flags: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = str(training_cfg.target_transform).lower()
    if mode == "raw":
        return targets.float()
    if mode not in {"rank_center", "exec_fillable_rank_neg1"}:
        raise ValueError(f"Unsupported training target transform: {training_cfg.target_transform}")

    transformed_chunks: list[torch.Tensor] = []
    offset = 0
    for group_size in group_sizes:
        next_offset = offset + group_size
        day_targets = targets[offset:next_offset].float()
        day_blocked = None
        if blocked_flags is not None:
            day_blocked = blocked_flags[offset:next_offset].bool()
        offset = next_offset
        if group_size <= 1:
            transformed_chunks.append(torch.zeros_like(day_targets))
            continue
        if mode == "rank_center":
            transformed_chunks.append(centered_rank(day_targets))
            continue
        if day_blocked is None:
            raise ValueError("exec_fillable_rank_neg1 requires blocked_flags")
        fillable_mask = ~day_blocked
        if not bool(fillable_mask.any().item()):
            transformed_chunks.append(torch.zeros_like(day_targets))
            continue
        transformed = torch.full_like(day_targets, -1.0)
        fillable_targets = day_targets[fillable_mask]
        if fillable_targets.numel() == 1:
            transformed[fillable_mask] = 1.0
        else:
            transformed[fillable_mask] = centered_rank(fillable_targets)
        transformed_chunks.append(transformed)
    return torch.cat(transformed_chunks, dim=0) if transformed_chunks else targets.float()


def make_day_batches(
    bundle,
    day_indices: Sequence[int],
    batch_days: int,
    clip_value: float,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    target_abs_cap: float | None = None,
):
    ordered = list(day_indices)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ordered)
    for start in range(0, len(ordered), batch_days):
        batch_day_indices = ordered[start : start + batch_days]
        feature_chunks = []
        target_chunks = []
        asset_chunks = []
        open_limit_chunks = []
        one_word_chunks = []
        group_sizes = []
        dates = []
        kept_day_indices = []
        for day_index in batch_day_indices:
            features, targets, asset_ids, open_limit_day1, one_word_day1 = get_day_data(bundle, day_index)
            if target_abs_cap is not None and target_abs_cap > 0:
                keep_mask = targets.abs() <= float(target_abs_cap)
                if not bool(keep_mask.all().item()):
                    features = features[keep_mask]
                    targets = targets[keep_mask]
                    asset_ids = asset_ids[keep_mask]
                    open_limit_day1 = open_limit_day1[keep_mask]
                    one_word_day1 = one_word_day1[keep_mask]
                if targets.numel() <= 0:
                    continue
            feature_chunks.append(normalize_cross_section(features.float(), clip_value))
            target_chunks.append(targets.float())
            asset_chunks.append(asset_ids)
            open_limit_chunks.append(open_limit_day1)
            one_word_chunks.append(one_word_day1)
            group_sizes.append(int(targets.shape[0]))
            dates.append(bundle["dates"][day_index])
            kept_day_indices.append(day_index)
        if not feature_chunks:
            continue
        yield {
            "features": torch.cat(feature_chunks, dim=0),
            "targets": torch.cat(target_chunks, dim=0),
            "asset_ids": torch.cat(asset_chunks, dim=0),
            "open_limit_day1": torch.cat(open_limit_chunks, dim=0),
            "one_word_day1": torch.cat(one_word_chunks, dim=0),
            "group_sizes": group_sizes,
            "dates": dates,
            "day_indices": kept_day_indices,
        }


def portfolio_size(group_size: int, data_cfg: DataSettings, fixed_top_k: int | None = None) -> int:
    if group_size <= 1:
        return max(1, group_size)
    if fixed_top_k is not None:
        return max(1, min(group_size, fixed_top_k))
    raw_k = int(round(group_size * data_cfg.portfolio_top_fraction))
    target_k = max(data_cfg.portfolio_min_k, raw_k)
    target_k = min(data_cfg.portfolio_max_k, target_k)
    return max(1, min(group_size, target_k))


def build_rank_weights(size: int, decay: float, *, device=None, dtype=None) -> torch.Tensor:
    rank = torch.arange(1, size + 1, device=device, dtype=dtype or torch.float32)
    weights = torch.ones_like(rank) if decay == 0 else rank.pow(-decay)
    return weights / weights.sum().clamp_min(1e-12)


def simulate_sleeve_equity(returns: Sequence[float], sleeves: int) -> np.ndarray:
    returns_array = np.asarray(list(returns), dtype=np.float64)
    if returns_array.size == 0:
        return np.ones(1, dtype=np.float64)
    sleeve_values = np.ones(max(1, sleeves), dtype=np.float64)
    combined = np.empty_like(returns_array)
    for index, trade_return in enumerate(returns_array):
        sleeve_id = index % len(sleeve_values)
        sleeve_values[sleeve_id] *= 1.0 + trade_return
        combined[index] = sleeve_values.mean()
    return combined


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    equity_array = np.asarray(list(equity_curve), dtype=np.float64)
    if equity_array.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity_array)
    drawdowns = equity_array / np.maximum(running_peak, 1e-12) - 1.0
    return float(drawdowns.min())


def _summarize_by_period(
    dates: Sequence[str],
    selected: np.ndarray,
    alpha: np.ndarray,
    width: int,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    periods = np.asarray([date[:width] for date in dates])
    for period in np.unique(periods):
        mask = periods == period
        summary[str(period)] = {
            "mean_return": float(selected[mask].mean()),
            "mean_alpha": float(alpha[mask].mean()),
            "hit_rate": float((selected[mask] > 0).mean()),
        }
    return summary


def summarize_period(
    data_cfg: DataSettings,
    dates: Sequence[str],
    selected_returns: Sequence[float],
    universe_returns: Sequence[float],
    oracle_returns: Sequence[float],
):
    selected = np.asarray(list(selected_returns), dtype=np.float64)
    universe = np.asarray(list(universe_returns), dtype=np.float64)
    oracle = np.asarray(list(oracle_returns), dtype=np.float64)
    if selected.size == 0:
        return {"metrics": {"selection_score": 0.0, "robust_score": 0.0, "mean_return": 0.0, "mean_alpha": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0}}

    alpha = selected - universe
    monthly = _summarize_by_period(dates, selected, alpha, width=7)
    yearly = _summarize_by_period(dates, selected, alpha, width=4)
    yearly_alpha = np.asarray([item["mean_alpha"] for item in yearly.values()], dtype=np.float64) if yearly else np.empty(0)
    monthly_alpha = np.asarray([item["mean_alpha"] for item in monthly.values()], dtype=np.float64) if monthly else np.empty(0)
    if yearly_alpha.size >= 2:
        stability = yearly_alpha
    elif monthly_alpha.size >= 2:
        stability = monthly_alpha
    else:
        stability = np.asarray([float(alpha.mean())], dtype=np.float64)

    equity_curve = simulate_sleeve_equity(selected, data_cfg.holding_days)
    max_drawdown = compute_max_drawdown(equity_curve)
    mean_return = float(selected.mean())
    mean_alpha = float(alpha.mean())
    hit_rate = float((selected > 0).mean())
    selection_score = mean_return
    selection_score += data_cfg.selection_alpha_weight * mean_alpha
    selection_score += data_cfg.selection_hit_weight * (hit_rate - 0.5)
    robust_score = mean_alpha + float(stability.min() - stability.std(ddof=0))
    robust_score += data_cfg.robust_drawdown_weight * max_drawdown
    return {
        "metrics": {
            "selection_score": float(selection_score),
            "robust_score": float(robust_score),
            "mean_return": mean_return,
            "mean_alpha": mean_alpha,
            "hit_rate": hit_rate,
            "max_drawdown": float(max_drawdown),
            "num_days": int(selected.size),
            "start_date": dates[0],
            "end_date": dates[-1],
        },
        "daily": {
            "dates": list(dates),
            "selected_returns": selected.tolist(),
            "universe_returns": universe.tolist(),
            "oracle_returns": oracle.tolist(),
            "alpha_returns": alpha.tolist(),
        },
        "monthly": monthly,
        "yearly": yearly,
    }


@torch.no_grad()
def evaluate_ranker(
    model,
    bundle,
    data_cfg: DataSettings,
    day_indices: Sequence[int],
    device: torch.device,
    *,
    top_k: int | None = None,
    batch_days: int = 32,
):
    from quant_impl.modeling.ranker import rerank_shortlist_scores

    model.eval()
    dates: list[str] = []
    selected_returns: list[float] = []
    universe_returns: list[float] = []
    oracle_returns: list[float] = []
    ideal_selected_returns: list[float] = []
    ideal_oracle_returns: list[float] = []
    one_word_selected_returns: list[float] = []
    one_word_oracle_returns: list[float] = []
    open_block_flags: list[bool] = []
    one_word_flags: list[bool] = []

    for batch in make_day_batches(
        bundle,
        day_indices=day_indices,
        batch_days=batch_days,
        clip_value=data_cfg.normalized_clip,
        shuffle=False,
    ):
        components = model.forward_components(batch["features"].to(device))
        predictions, _ = rerank_shortlist_scores(
            model,
            components["broad_score"].detach().float(),
            components["linear_score"].detach().float(),
            components["rerank_latent"].detach().float(),
            batch["group_sizes"],
        )
        predictions = predictions.cpu()
        offset = 0
        for group_size, date in zip(batch["group_sizes"], batch["dates"]):
            next_offset = offset + group_size
            day_scores = predictions[offset:next_offset]
            day_targets = batch["targets"][offset:next_offset]
            day_open_flags = batch["open_limit_day1"][offset:next_offset].bool()
            day_one_word_flags = batch["one_word_day1"][offset:next_offset].bool()
            offset = next_offset

            k = portfolio_size(group_size, data_cfg, fixed_top_k=top_k or data_cfg.eval_top_k)
            if k != 1:
                weights = build_rank_weights(k, data_cfg.portfolio_rank_decay, device=day_targets.device, dtype=day_targets.dtype)
                top_indices = torch.topk(day_scores, k=k, largest=True).indices
                oracle_indices = torch.topk(day_targets, k=k, largest=True).indices
                selected_return = float((day_targets[top_indices] * weights).sum().item())
                oracle_return = float((day_targets[oracle_indices] * weights).sum().item())
                selected_returns.append(selected_return)
                universe_returns.append(float(day_targets.mean().item()))
                oracle_returns.append(oracle_return)
                ideal_selected_returns.append(selected_return)
                ideal_oracle_returns.append(oracle_return)
                one_word_selected_returns.append(selected_return)
                one_word_oracle_returns.append(oracle_return)
                open_block_flags.append(False)
                one_word_flags.append(False)
                dates.append(date)
                continue

            top_index = int(torch.argmax(day_scores).item())
            selected_return_ideal = float(day_targets[top_index].item())
            selected_return_open = 0.0 if bool(day_open_flags[top_index].item()) else selected_return_ideal
            selected_return_one_word = 0.0 if bool(day_one_word_flags[top_index].item()) else selected_return_ideal
            open_fillable_mask = ~day_open_flags
            one_word_fillable_mask = ~day_one_word_flags
            oracle_ideal_return = float(day_targets.max().item())
            oracle_open_return = (
                float(day_targets[open_fillable_mask].max().item())
                if bool(open_fillable_mask.any().item())
                else 0.0
            )
            oracle_one_word_return = (
                float(day_targets[one_word_fillable_mask].max().item())
                if bool(one_word_fillable_mask.any().item())
                else 0.0
            )

            selected_returns.append(selected_return_open)
            universe_returns.append(float(day_targets.mean().item()))
            oracle_returns.append(oracle_open_return)
            ideal_selected_returns.append(selected_return_ideal)
            ideal_oracle_returns.append(oracle_ideal_return)
            one_word_selected_returns.append(selected_return_one_word)
            one_word_oracle_returns.append(oracle_one_word_return)
            open_block_flags.append(bool(day_open_flags[top_index].item()))
            one_word_flags.append(bool(day_one_word_flags[top_index].item()))
            dates.append(date)

    primary_report = summarize_period(data_cfg, dates, selected_returns, universe_returns, oracle_returns)
    ideal_report = summarize_period(data_cfg, dates, ideal_selected_returns, universe_returns, ideal_oracle_returns)
    one_word_report = summarize_period(
        data_cfg,
        dates,
        one_word_selected_returns,
        universe_returns,
        one_word_oracle_returns,
    )
    primary_report["trade_rate"] = 1.0 - float(np.mean(open_block_flags)) if open_block_flags else 1.0
    primary_report["block_rate_open_limit"] = float(np.mean(open_block_flags)) if open_block_flags else 0.0
    primary_report["block_rate_one_word"] = float(np.mean(one_word_flags)) if one_word_flags else 0.0
    primary_report["supplemental"] = {
        "ideal_metrics": ideal_report["metrics"],
        "one_word_metrics": one_word_report["metrics"],
        "ideal_daily": ideal_report.get("daily", {}),
        "one_word_daily": one_word_report.get("daily", {}),
    }
    return primary_report


def compute_linear_ic_weights(
    bundle,
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
    day_indices: Sequence[int],
) -> torch.Tensor:
    feature_dim = bundle["features"].shape[1]
    accumulator = np.zeros(feature_dim, dtype=np.float64)
    total_weight = 0.0
    for day_index in day_indices:
        batch = next(
            make_day_batches(
                bundle,
                day_indices=[day_index],
                batch_days=1,
                clip_value=data_cfg.normalized_clip,
                shuffle=False,
                target_abs_cap=(
                    training_cfg.train_target_abs_cap
                    if training_cfg.train_target_cap_applies_to_linear_head
                    else None
                ),
            ),
            None,
        )
        if batch is None:
            continue
        features = batch["features"].numpy()
        targets = transform_training_targets(
            batch["targets"],
            batch["group_sizes"],
            training_cfg,
            blocked_flags=batch.get("open_limit_day1"),
        ).numpy()
        centered_targets = targets - targets.mean()
        target_std = centered_targets.std()
        if target_std < 1e-8:
            continue
        centered_features = features - features.mean(axis=0, keepdims=True)
        feature_std = features.std(axis=0)
        denom = feature_std * target_std
        accumulator += np.divide(
            (centered_features * centered_targets[:, None]).mean(axis=0),
            denom,
            out=np.zeros(feature_dim, dtype=np.float64),
            where=denom > 1e-8,
        )
        total_weight += 1.0
    if total_weight <= 0:
        return torch.zeros(bundle["features"].shape[1], dtype=torch.float32)
    weight = accumulator / total_weight
    norm = np.linalg.norm(weight)
    if norm > 0:
        weight = weight / norm
    return torch.from_numpy(weight.astype(np.float32, copy=False))


def latest_market_date(config: AppConfig) -> str:
    parquet_file = pq.ParquetFile(resolve_merged_path(config))
    latest = None
    for row_group in range(parquet_file.num_row_groups):
        frame = parquet_file.read_row_group(row_group, columns=["date"]).to_pandas()
        if frame.empty:
            continue
        current = pd.to_datetime(frame["date"], errors="coerce").max()
        if pd.notna(current) and (latest is None or current > latest):
            latest = current
    if latest is None:
        raise RuntimeError("Could not determine latest market date")
    latest_text = latest.strftime("%Y-%m-%d")
    LOGGER.info("Latest market date resolved=%s", latest_text)
    return latest_text


def build_scoring_snapshot(
    config: AppConfig,
    *,
    as_of_date: str | None = None,
    limit_stocks: int | None = None,
) -> dict[str, object]:
    snapshot_date = pd.Timestamp(str(as_of_date)) if as_of_date else pd.Timestamp(latest_market_date(config))
    merged_path = resolve_merged_path(config)
    LOGGER.info(
        "Building scoring snapshot date=%s merged_path=%s limit_stocks=%s",
        snapshot_date.strftime("%Y-%m-%d"),
        merged_path,
        limit_stocks,
    )
    feature_names = feature_columns(config.data)
    records: list[tuple[str, np.ndarray]] = []
    for _, _, code, frame in _iter_stock_frames(merged_path, limit_stocks=limit_stocks):
        feature_frame = _build_single_stock_feature_frame(code, frame, config.data, include_target=False)
        if feature_frame is None or feature_frame.empty:
            continue
        row = feature_frame.loc[feature_frame["date"] == snapshot_date]
        if row.empty:
            continue
        vector = row.iloc[0][feature_names].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        vector = np.clip(vector, -config.data.feature_clip, config.data.feature_clip)
        records.append((code, vector))
    if not records:
        raise RuntimeError(f"No scoreable universe found for {snapshot_date.date()}")
    codes = [item[0] for item in records]
    features = np.stack([item[1] for item in records], axis=0)
    LOGGER.info(
        "Scoring snapshot ready date=%s universe_size=%s feature_count=%s",
        snapshot_date.strftime("%Y-%m-%d"),
        len(codes),
        len(feature_names),
    )
    return {
        "date": snapshot_date.strftime("%Y-%m-%d"),
        "codes": codes,
        "features": torch.tensor(features, dtype=torch.float32),
        "feature_names": feature_names,
    }


def locate_day_index(bundle, date_text: str) -> int | None:
    try:
        return bundle["dates"].index(date_text)
    except ValueError:
        return None


def realized_day_lookup(bundle, date_text: str) -> dict[str, float] | None:
    day_index = locate_day_index(bundle, date_text)
    if day_index is None:
        return None
    _, targets, asset_ids, _, _ = get_day_data(bundle, day_index)
    codes = [bundle["assets"][int(asset_id)] for asset_id in asset_ids.tolist()]
    return {code: float(target) for code, target in zip(codes, targets.tolist())}


def realized_day_detail_lookup(bundle, date_text: str) -> dict[str, dict[str, float | bool]] | None:
    day_index = locate_day_index(bundle, date_text)
    if day_index is None:
        return None
    _, targets, asset_ids, open_limit_day1, one_word_day1 = get_day_data(bundle, day_index)
    detail_map: dict[str, dict[str, float | bool]] = {}
    for asset_id, target, open_limit, one_word in zip(
        asset_ids.tolist(),
        targets.tolist(),
        open_limit_day1.tolist(),
        one_word_day1.tolist(),
    ):
        code = bundle["assets"][int(asset_id)]
        ideal_return = float(target)
        blocked_open = bool(open_limit)
        blocked_one_word = bool(one_word)
        detail_map[code] = {
            "ideal_return": ideal_return,
            "open_limit_day1": blocked_open,
            "one_word_day1": blocked_one_word,
            "strict_open_return": 0.0 if blocked_open else ideal_return,
            "strict_one_word_return": 0.0 if blocked_one_word else ideal_return,
        }
    return detail_map
