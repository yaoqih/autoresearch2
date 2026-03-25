from __future__ import annotations

import argparse
import json
import logging

from quant_impl.data.market import build_market_cache, merge_daily_parquets
from quant_impl.pipelines.daily import daily_pipeline, run_download_step
from quant_impl.pipelines.predict_history import predict_history_pipeline
from quant_impl.pipelines.predict import predict_pipeline
from quant_impl.pipelines.train import train_pipeline
from quant_impl.pipelines.validate import validate_pipeline
from quant_impl.settings import load_config
from quant_impl.utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default=None, help="Override global log level for this command")
    parser.add_argument("--log-file", default=None, help="Override log file path for this command")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant implementation CLI")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download raw daily stock data")
    add_logging_args(download)
    download.add_argument("--report-file", default=None)
    download.set_defaults(command="download")

    merge = subparsers.add_parser("merge", help="Merge raw per-symbol parquet files")
    add_logging_args(merge)
    merge.add_argument("--force", action="store_true", help="Rebuild merged parquet")
    merge.set_defaults(command="merge")

    prepare = subparsers.add_parser("prepare", help="Build merged parquet cache bundle")
    add_logging_args(prepare)
    prepare.add_argument("--force", action="store_true", help="Rebuild bundle")
    prepare.add_argument("--limit-stocks", type=int, default=None)
    prepare.set_defaults(command="prepare")

    train = subparsers.add_parser("train", help="Train the ranking model")
    add_logging_args(train)
    train.add_argument("--device", default=None)
    train.add_argument("--profile", choices=["full", "probe", "screen"], default="full")
    train.add_argument("--deploy-only", action="store_true")
    train.add_argument("--force-prepare", action="store_true")
    train.add_argument("--limit-stocks", type=int, default=None)
    train.add_argument("--deployment-start-date", default=None)
    train.add_argument("--deployment-end-date", default=None)
    train.add_argument("--deployment-anchor-date", default=None)
    train.add_argument("--deployment-lookback-years", type=int, default=None)
    train.set_defaults(command="train")

    predict = subparsers.add_parser("predict", help="Score the latest available market date")
    add_logging_args(predict)
    predict.add_argument("--device", default=None)
    predict.add_argument("--as-of-date", default=None)
    predict.add_argument("--limit-stocks", type=int, default=None)
    predict.set_defaults(command="predict")

    predict_history = subparsers.add_parser("predict-history", help="Score a historical date range")
    add_logging_args(predict_history)
    predict_history.add_argument("--device", default=None)
    predict_history.add_argument("--start-date", default=None)
    predict_history.add_argument("--end-date", default=None)
    predict_history.add_argument("--anchor-date", default=None)
    predict_history.add_argument("--lookback-months", type=int, default=None)
    predict_history.add_argument("--validate", action="store_true")
    predict_history.add_argument("--limit-stocks", type=int, default=None)
    predict_history.set_defaults(command="predict-history")

    validate = subparsers.add_parser("validate", help="Validate historical prediction archives")
    add_logging_args(validate)
    validate.set_defaults(command="validate")

    daily = subparsers.add_parser("daily", help="Run the full daily pipeline")
    add_logging_args(daily)
    daily.add_argument("--device", default=None)
    daily.add_argument("--retrain", action="store_true")
    daily.add_argument("--profile", choices=["full", "probe", "screen"], default="screen")
    daily.set_defaults(command="daily")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.log_level is not None:
        config.logging.level = args.log_level
    if args.log_file is not None:
        config.logging.log_file = args.log_file
    if args.command == "download" and args.report_file is not None:
        config.download.report_file = args.report_file
    log_path = setup_logging(config, args.command)
    LOGGER.info(
        "Starting command=%s config=%s log_file=%s",
        args.command,
        args.config or "configs/default.yaml",
        log_path,
    )

    if args.command == "download":
        result = run_download_step(config)
    elif args.command == "merge":
        result = merge_daily_parquets(config, force=args.force)
    elif args.command == "prepare":
        result = {"bundle_path": str(build_market_cache(config, force=args.force, limit_stocks=args.limit_stocks))}
    elif args.command == "train":
        result = train_pipeline(
            config,
            device=args.device,
            profile=args.profile,
            deploy_only=args.deploy_only,
            force_prepare=args.force_prepare,
            limit_stocks=args.limit_stocks,
            deployment_start_date=args.deployment_start_date,
            deployment_end_date=args.deployment_end_date,
            deployment_anchor_date=args.deployment_anchor_date,
            deployment_lookback_years=args.deployment_lookback_years,
        )
    elif args.command == "predict":
        result = predict_pipeline(
            config,
            device=args.device,
            as_of_date=args.as_of_date,
            limit_stocks=args.limit_stocks,
        )
    elif args.command == "predict-history":
        result = predict_history_pipeline(
            config,
            device=args.device,
            start_date=args.start_date,
            end_date=args.end_date,
            anchor_date=args.anchor_date,
            lookback_months=args.lookback_months,
            validate=args.validate,
            limit_stocks=args.limit_stocks,
        )
    elif args.command == "validate":
        result = validate_pipeline(config)
    elif args.command == "daily":
        result = daily_pipeline(config, device=args.device, retrain=args.retrain, profile=args.profile)
    else:
        raise ValueError(f"Unsupported command {args.command}")

    if isinstance(result, dict) and log_path is not None:
        result.setdefault("log_file", str(log_path))
    LOGGER.info("Completed command=%s", args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
