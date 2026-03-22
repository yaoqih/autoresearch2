from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant_impl.data.market import load_market_bundle, realized_day_lookup
from quant_impl.settings import AppConfig
from quant_impl.utils.io import read_json, write_json


LOGGER = logging.getLogger(__name__)


def _prediction_files(predictions_dir: Path) -> list[Path]:
    return sorted(predictions_dir.glob("*/prediction.json"))


def validate_pipeline(config: AppConfig) -> dict[str, object]:
    bundle = load_market_bundle(config, force=False)
    history_path = config.validation_history_path
    existing = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    existing_ids = set(existing["archive_id"].astype(str).tolist()) if not existing.empty else set()
    prediction_files = _prediction_files(config.paths.predictions_dir)
    LOGGER.info(
        "Validate pipeline start predictions=%s existing_history_rows=%s history_path=%s",
        len(prediction_files),
        len(existing),
        history_path,
    )

    rows = []
    validated = 0
    pending = 0
    for path in prediction_files:
        payload = read_json(path)
        if not payload:
            continue
        if payload.get("archive_id") in existing_ids:
            continue
        realized_map = realized_day_lookup(bundle, payload["as_of_date"])
        if not realized_map or payload["selected_code"] not in realized_map:
            pending += 1
            continue

        values = list(realized_map.values())
        selected_return = float(realized_map[payload["selected_code"]])
        universe_return = float(sum(values) / len(values))
        oracle_return = float(max(values))
        alpha = selected_return - universe_return
        row = {
            "archive_id": payload["archive_id"],
            "prediction_name": payload["prediction_name"],
            "as_of_date": payload["as_of_date"],
            "entry_date": payload["entry_date"],
            "exit_date": payload["exit_date"],
            "selected_code": payload["selected_code"],
            "selected_score": payload["selected_score"],
            "selected_return": selected_return,
            "universe_return": universe_return,
            "oracle_return": oracle_return,
            "alpha": alpha,
            "hit": int(selected_return > 0),
        }
        payload["status"] = "validated"
        payload["validation"] = row
        write_json(path, payload)
        rows.append(row)
        validated += 1

    if rows:
        history = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
        history.to_csv(history_path, index=False)
        LOGGER.info("Validation history updated path=%s new_rows=%s total_rows=%s", history_path, len(rows), len(history))
    else:
        LOGGER.info("No new validations were written history_path=%s", history_path)
    LOGGER.info("Validate pipeline finished validated=%s pending=%s", validated, pending)
    return {
        "validated": validated,
        "still_pending": pending,
        "history_path": str(history_path),
    }
