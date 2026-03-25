from __future__ import annotations

import logging

from download_stock import TradingCalendar
import pandas as pd
from pandas.tseries.offsets import BDay
import torch

from quant_impl.data.market import (
    available_market_dates,
    build_scoring_snapshot,
    get_day_data,
    load_market_bundle,
    locate_day_index,
    normalize_cross_section,
)
from quant_impl.modeling.ranker import CrossSectionalRanker
from quant_impl.pipelines.model_contract import (
    contract_config_from_artifact,
    load_model_artifact,
    model_settings_from_artifact,
)
from quant_impl.pipelines.prediction_archive import refresh_prediction_views, write_canonical_daily_prediction
from quant_impl.settings import AppConfig
from quant_impl.utils.io import utc_timestamp, write_json


LOGGER = logging.getLogger(__name__)


def _load_model(config: AppConfig, device: torch.device) -> tuple[CrossSectionalRanker, dict]:
    artifact = load_model_artifact(config.model_path)
    model_cfg = model_settings_from_artifact(artifact)
    artifact_feature_names = artifact.get("feature_names") or []
    linear_weights = artifact["state_dict"]["linear_head.weight"].squeeze(0)
    model = CrossSectionalRanker(len(artifact_feature_names), model_cfg, linear_weights)
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()
    LOGGER.info(
        "Loaded model path=%s created_at=%s device=%s",
        config.model_path,
        artifact.get("created_at"),
        device,
    )
    return model, artifact


def _rank_top_candidates(codes: list[str], scores: torch.Tensor, top_n: int) -> list[dict[str, object]]:
    order = torch.argsort(scores, descending=True)
    records = []
    for rank, idx in enumerate(order[:top_n].tolist(), start=1):
        records.append(
            {
                "rank": rank,
                "code": codes[idx],
                "score": float(scores[idx].item()),
            }
        )
    return records


def _write_prediction_payload(
    config: AppConfig,
    artifact: dict,
    *,
    as_of_date: str,
    entry_date: str,
    exit_date: str,
    universe_size: int,
    records: list[dict[str, object]],
) -> dict[str, object]:
    archive_id = f"{as_of_date.replace('-', '')}_{utc_timestamp()}_{config.inference.prediction_name}"
    archive_dir = config.paths.predictions_dir / archive_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    prediction_payload = {
        "archive_id": archive_id,
        "prediction_name": config.inference.prediction_name,
        "model_path": str(config.model_path),
        "model_created_at": artifact["created_at"],
        "as_of_date": as_of_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "status": "pending",
        "universe_size": universe_size,
        "top_k": config.inference.top_k,
        "top_candidates": records,
        "selected_code": records[0]["code"],
        "selected_score": records[0]["score"],
    }
    pd.DataFrame(records).to_csv(archive_dir / "top_candidates.csv", index=False)
    write_json(archive_dir / "prediction.json", prediction_payload)
    write_canonical_daily_prediction(config, prediction_payload)
    refresh_prediction_views(config)
    LOGGER.info("Prediction archive saved archive_id=%s archive_dir=%s", archive_id, archive_dir)
    return prediction_payload


def predict_bundle_day(
    config: AppConfig,
    model: CrossSectionalRanker,
    artifact: dict,
    contract_config: AppConfig,
    bundle,
    *,
    device: torch.device,
    day_index: int,
) -> dict[str, object]:
    as_of_date = str(bundle["dates"][day_index])
    features, _, asset_ids, _, _ = get_day_data(bundle, day_index)
    codes = [bundle["assets"][int(asset_id)] for asset_id in asset_ids.tolist()]
    normalized = normalize_cross_section(features.float(), contract_config.data.normalized_clip).to(device)
    scores = model.score_batch(normalized, [normalized.shape[0]]).detach().cpu()
    records = _rank_top_candidates(codes, scores, min(config.inference.archive_top_n, len(codes)))
    LOGGER.info(
        "Prediction ranking complete date=%s universe_size=%s top_n=%s selected_code=%s selected_score=%.6f bundle_fast_path=true",
        as_of_date,
        len(codes),
        len(records),
        records[0]["code"],
        records[0]["score"],
    )
    return _write_prediction_payload(
        config,
        artifact,
        as_of_date=as_of_date,
        entry_date=_future_trading_day(
            as_of_date,
            contract_config.data.entry_offset_days,
            bundle=bundle,
        ),
        exit_date=_future_trading_day(
            as_of_date,
            contract_config.data.exit_offset_days,
            bundle=bundle,
        ),
        universe_size=len(codes),
        records=records,
    )


def _future_trading_day(
    date_text: str,
    offset: int,
    *,
    calendar: TradingCalendar | None = None,
    bundle=None,
    available_dates: list[str] | None = None,
) -> str:
    current = pd.Timestamp(date_text).normalize()
    if offset <= 0:
        return current.strftime("%Y-%m-%d")

    def _consume_known_dates(
        current_ts: pd.Timestamp,
        remaining_steps: int,
        known_dates: list[str] | None,
    ) -> tuple[pd.Timestamp, int]:
        if not known_dates or remaining_steps <= 0:
            return current_ts, remaining_steps
        future = pd.DatetimeIndex(pd.to_datetime(known_dates)).normalize()
        future = future[future > current_ts]
        if not len(future):
            return current_ts, remaining_steps
        consumed = min(remaining_steps, len(future))
        return future[consumed - 1], remaining_steps - consumed

    remaining = int(offset)
    if bundle is not None:
        current, remaining = _consume_known_dates(current, remaining, list(bundle["dates"]))
        if remaining == 0:
            return current.strftime("%Y-%m-%d")
    if available_dates is not None:
        current, remaining = _consume_known_dates(current, remaining, available_dates)
        if remaining == 0:
            return current.strftime("%Y-%m-%d")

    try:
        trading_calendar = calendar or TradingCalendar()
        current_date = current.date()
        for _ in range(remaining):
            current_date = trading_calendar.next_session(current_date)
        return current_date.strftime("%Y-%m-%d")
    except Exception:
        fallback = current
        for _ in range(remaining):
            fallback = (fallback + BDay(1)).normalize()
        return fallback.strftime("%Y-%m-%d")


def predict_pipeline(
    config: AppConfig,
    *,
    device: str | None = None,
    as_of_date: str | None = None,
    limit_stocks: int | None = None,
    bundle=None,
) -> dict[str, object]:
    device_obj = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(
        "Predict pipeline start model_path=%s device=%s as_of_date=%s limit_stocks=%s",
        config.model_path,
        device_obj,
        as_of_date,
        limit_stocks,
    )
    model, artifact = _load_model(config, device_obj)
    contract_config = contract_config_from_artifact(config, artifact)
    prediction_bundle = bundle
    if prediction_bundle is None and as_of_date is not None:
        prediction_bundle = load_market_bundle(contract_config, force=False)
    market_dates = None
    if as_of_date is not None and locate_day_index(prediction_bundle, as_of_date) is None:
        market_dates = available_market_dates(contract_config)
    snapshot = build_scoring_snapshot(contract_config, as_of_date=as_of_date, limit_stocks=limit_stocks)
    artifact_feature_names = list(artifact.get("feature_names") or [])
    if artifact_feature_names and snapshot["feature_names"] != artifact_feature_names:
        raise ValueError(
            "Prediction feature contract mismatch between model artifact and runtime feature builder"
        )
    features = normalize_cross_section(snapshot["features"].float(), contract_config.data.normalized_clip).to(device_obj)
    scores = model.score_batch(features, [features.shape[0]]).detach().cpu()
    records = _rank_top_candidates(snapshot["codes"], scores, min(config.inference.archive_top_n, len(snapshot["codes"])))
    LOGGER.info(
        "Prediction ranking complete date=%s universe_size=%s top_n=%s selected_code=%s selected_score=%.6f",
        snapshot["date"],
        len(snapshot["codes"]),
        len(records),
        records[0]["code"],
        records[0]["score"],
    )
    return _write_prediction_payload(
        config,
        artifact,
        as_of_date=snapshot["date"],
        entry_date=_future_trading_day(
            snapshot["date"],
            contract_config.data.entry_offset_days,
            bundle=prediction_bundle,
            available_dates=market_dates,
        ),
        exit_date=_future_trading_day(
            snapshot["date"],
            contract_config.data.exit_offset_days,
            bundle=prediction_bundle,
            available_dates=market_dates,
        ),
        universe_size=len(snapshot["codes"]),
        records=records,
    )
