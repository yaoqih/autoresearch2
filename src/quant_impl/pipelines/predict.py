from __future__ import annotations

import logging

import pandas as pd
import torch

from quant_impl.data.market import build_scoring_snapshot, normalize_cross_section
from quant_impl.modeling.ranker import CrossSectionalRanker
from quant_impl.settings import AppConfig
from quant_impl.utils.io import utc_timestamp, write_json


LOGGER = logging.getLogger(__name__)


def _load_model(config: AppConfig, device: torch.device) -> tuple[CrossSectionalRanker, dict]:
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {config.model_path}")
    artifact = torch.load(config.model_path, map_location="cpu", weights_only=False)
    linear_weights = artifact["state_dict"]["linear_head.weight"].squeeze(0)
    model = CrossSectionalRanker(len(artifact["feature_names"]), config.model, linear_weights)
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


def _future_business_day(date_text: str, offset: int) -> str:
    return (pd.Timestamp(date_text) + pd.offsets.BDay(offset)).strftime("%Y-%m-%d")


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
    snapshot = build_scoring_snapshot(config, as_of_date=as_of_date, limit_stocks=limit_stocks)
    features = normalize_cross_section(snapshot["features"].float(), config.data.normalized_clip).to(device_obj)
    scores = model.score_day(features).scores.detach().cpu()
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
        "entry_date": _future_business_day(snapshot["date"], config.data.entry_offset_days),
        "exit_date": _future_business_day(snapshot["date"], config.data.exit_offset_days),
        "status": "pending",
        "universe_size": len(snapshot["codes"]),
        "top_k": config.inference.top_k,
        "top_candidates": records,
        "selected_code": records[0]["code"],
        "selected_score": records[0]["score"],
    }
    pd.DataFrame(records).to_csv(archive_dir / "top_candidates.csv", index=False)
    write_json(archive_dir / "prediction.json", prediction_payload)
    LOGGER.info("Prediction archive saved archive_id=%s archive_dir=%s", archive_id, archive_dir)
    return prediction_payload
