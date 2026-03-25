from __future__ import annotations

import logging

from download_stock import TradingCalendar
import pandas as pd
import torch

from quant_impl.data.market import build_scoring_snapshot, normalize_cross_section
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


def _future_trading_day(date_text: str, offset: int, *, calendar: TradingCalendar | None = None) -> str:
    current = pd.Timestamp(date_text).date()
    if offset <= 0:
        return current.strftime("%Y-%m-%d")
    trading_calendar = calendar or TradingCalendar()
    for _ in range(int(offset)):
        current = trading_calendar.next_session(current)
    return current.strftime("%Y-%m-%d")


def predict_pipeline(
    config: AppConfig,
    *,
    device: str | None = None,
    as_of_date: str | None = None,
    limit_stocks: int | None = None,
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
    snapshot = build_scoring_snapshot(contract_config, as_of_date=as_of_date, limit_stocks=limit_stocks)
    artifact_feature_names = list(artifact.get("feature_names") or [])
    if artifact_feature_names and snapshot["feature_names"] != artifact_feature_names:
        raise ValueError(
            "Prediction feature contract mismatch between model artifact and runtime feature builder"
        )
    features = normalize_cross_section(snapshot["features"].float(), contract_config.data.normalized_clip).to(device_obj)
    scores = model.score_batch(features, [features.shape[0]]).detach().cpu()
    order = torch.argsort(scores, descending=True)

    top_n = min(config.inference.archive_top_n, len(order))
    records = []
    for rank, idx in enumerate(order[:top_n].tolist(), start=1):
        records.append(
            {
                "rank": rank,
                "code": snapshot["codes"][idx],
                "score": float(scores[idx].item()),
            }
        )
    LOGGER.info(
        "Prediction ranking complete date=%s universe_size=%s top_n=%s selected_code=%s selected_score=%.6f",
        snapshot["date"],
        len(snapshot["codes"]),
        top_n,
        records[0]["code"],
        records[0]["score"],
    )

    archive_id = f"{snapshot['date'].replace('-', '')}_{utc_timestamp()}_{config.inference.prediction_name}"
    archive_dir = config.paths.predictions_dir / archive_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    prediction_payload = {
        "archive_id": archive_id,
        "prediction_name": config.inference.prediction_name,
        "model_path": str(config.model_path),
        "model_created_at": artifact["created_at"],
        "as_of_date": snapshot["date"],
        "entry_date": _future_trading_day(snapshot["date"], contract_config.data.entry_offset_days),
        "exit_date": _future_trading_day(snapshot["date"], contract_config.data.exit_offset_days),
        "status": "pending",
        "universe_size": len(snapshot["codes"]),
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
