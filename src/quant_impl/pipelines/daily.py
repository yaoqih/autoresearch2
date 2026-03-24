from __future__ import annotations

import logging
import subprocess
import sys

import pandas as pd

from quant_impl.data.market import build_market_cache, latest_market_date, merge_daily_parquets
from quant_impl.pipelines.predict import predict_pipeline
from quant_impl.pipelines.train import train_pipeline
from quant_impl.pipelines.validate import validate_pipeline
from quant_impl.settings import AppConfig, REPO_ROOT
from quant_impl.utils.logging_utils import resolve_log_file_path, resolve_runtime_path


LOGGER = logging.getLogger(__name__)


def _refresh_start_date(config: AppConfig) -> str:
    if config.paths.merged_parquet.exists():
        latest = pd.Timestamp(latest_market_date(config))
        refreshed = latest - pd.Timedelta(days=config.download.refresh_lookback_days)
        return max(refreshed, pd.Timestamp(config.download.start_date)).strftime("%Y-%m-%d")
    return config.download.start_date


def run_download_step(config: AppConfig) -> dict[str, object]:
    script_path = REPO_ROOT / "download_stock.py"
    refresh_start_date = _refresh_start_date(config)
    log_file = resolve_log_file_path(config, "download")
    report_file = resolve_runtime_path(config.download.report_file)
    cookie_cache_file = resolve_runtime_path(config.download.eastmoney_cookie_cache_file)
    cookie_script = resolve_runtime_path(config.download.eastmoney_cookie_script)
    cmd = [
        sys.executable,
        str(script_path),
        "--parquet-dir",
        str(config.paths.raw_daily_dir),
        "--start-date",
        refresh_start_date,
        "--adjust",
        config.download.adjust,
        "--max-workers",
        str(config.download.max_workers),
        "--host-max-workers",
        str(config.download.host_max_workers),
        "--max-retries",
        str(config.download.max_retries),
        "--timeout",
        str(config.download.timeout),
        "--request-interval",
        str(config.download.request_interval),
        "--request-jitter",
        str(config.download.request_jitter),
        "--retry-sleep",
        str(config.download.retry_sleep),
        "--log-level",
        str(config.logging.level).upper(),
    ]
    cmd.append("--use-env-proxy" if config.download.use_env_proxy else "--no-use-env-proxy")
    if not config.logging.console:
        cmd.append("--no-console-log")
    if config.download.end_date:
        cmd.extend(["--end-date", config.download.end_date])
    if config.download.limit is not None:
        cmd.extend(["--limit", str(config.download.limit)])
    if log_file is not None:
        cmd.extend(["--log-file", str(log_file)])
    if report_file is not None:
        cmd.extend(["--report-file", str(report_file)])
    if config.download.eastmoney_cookie_warmup:
        cmd.append("--eastmoney-cookie-warmup")
        if cookie_cache_file is not None:
            cmd.extend(["--eastmoney-cookie-cache-file", str(cookie_cache_file)])
        cmd.extend(["--eastmoney-cookie-max-age-seconds", str(config.download.eastmoney_cookie_max_age_seconds)])
        cmd.extend(["--eastmoney-cookie-node-binary", str(config.download.eastmoney_cookie_node_binary)])
        if cookie_script is not None:
            cmd.extend(["--eastmoney-cookie-script", str(cookie_script)])
        if config.download.eastmoney_browser_path:
            cmd.extend(["--eastmoney-browser-path", str(config.download.eastmoney_browser_path)])
        if config.download.eastmoney_browser_proxy:
            cmd.extend(["--eastmoney-browser-proxy", str(config.download.eastmoney_browser_proxy)])
        cmd.extend(["--eastmoney-cookie-timeout-ms", str(config.download.eastmoney_cookie_timeout_ms)])
    if config.download.shuffle_symbols:
        cmd.append("--shuffle-symbols")
    if config.download.symbols_file:
        cmd.extend(["--symbols-file", config.download.symbols_file])
    elif config.download.extra_symbols:
        cmd.append("--symbols")
        cmd.extend(config.download.extra_symbols)
    LOGGER.info(
        "Running download step start_date=%s end_date=%s raw_dir=%s workers=%s log_file=%s report_file=%s",
        refresh_start_date,
        config.download.end_date,
        config.paths.raw_daily_dir,
        config.download.max_workers,
        log_file,
        report_file,
    )
    subprocess.run(cmd, check=True)
    LOGGER.info("Download step finished raw_dir=%s", config.paths.raw_daily_dir)
    return {
        "command": cmd,
        "raw_daily_dir": str(config.paths.raw_daily_dir),
        "refresh_start_date": refresh_start_date,
        "log_file": str(log_file) if log_file is not None else None,
        "report_file": str(report_file) if report_file is not None else None,
        "eastmoney_cookie_cache_file": str(cookie_cache_file) if cookie_cache_file is not None else None,
    }


def daily_pipeline(
    config: AppConfig,
    *,
    device: str | None = None,
    retrain: bool = False,
    profile: str = "screen",
) -> dict[str, object]:
    LOGGER.info("Daily pipeline start retrain=%s profile=%s device=%s", retrain, profile, device)
    download_result = run_download_step(config)
    LOGGER.info("Daily pipeline merging raw parquet files")
    merge_result = merge_daily_parquets(config, force=True)
    LOGGER.info("Daily pipeline rebuilding bundle cache")
    bundle_path = build_market_cache(config, force=True)
    if retrain or not config.model_path.exists():
        LOGGER.info("Daily pipeline training model retrain=%s model_exists=%s", retrain, config.model_path.exists())
        train_result = train_pipeline(config, device=device, profile=profile, force_prepare=False)
    else:
        LOGGER.info("Daily pipeline skipping training existing_model=%s", config.model_path)
        train_result = None
    LOGGER.info("Daily pipeline generating prediction")
    prediction_result = predict_pipeline(config, device=device)
    LOGGER.info("Daily pipeline validating archives")
    validation_result = validate_pipeline(config)
    LOGGER.info("Daily pipeline finished prediction_archive=%s", prediction_result["archive_id"])
    return {
        "download": download_result,
        "merge": merge_result,
        "prepare": {"bundle_path": str(bundle_path)},
        "train": train_result,
        "prediction": prediction_result,
        "validation": validation_result,
    }
