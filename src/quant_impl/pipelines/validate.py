from __future__ import annotations

import logging

import pandas as pd

from quant_impl.data.market import load_market_bundle, realized_day_lookup
from quant_impl.pipelines.model_contract import contract_config_from_artifact, load_prediction_artifact
from quant_impl.pipelines.prediction_archive import (
    canonical_prediction_payload,
    refresh_prediction_views,
    sync_prediction_website_archives,
    update_legacy_run_archive,
)
from quant_impl.settings import AppConfig
from quant_impl.utils.io import write_json


LOGGER = logging.getLogger(__name__)


def validate_pipeline(config: AppConfig) -> dict[str, object]:
    history_path = config.validation_history_path
    existing = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    history_rows_by_date: dict[str, dict[str, object]] = {}
    if not existing.empty and "as_of_date" in existing.columns:
        for row in existing.to_dict("records"):
            history_rows_by_date[str(row["as_of_date"])] = row
    prediction_files = sync_prediction_website_archives(config)
    bundle_cache: dict[str, object] = {}
    LOGGER.info(
        "Validate pipeline start predictions=%s existing_history_rows=%s history_path=%s",
        len(prediction_files),
        len(existing),
        history_path,
    )

    rows = []
    validated = 0
    pending = 0
    for payload in prediction_files:
        if not payload:
            continue
        try:
            model_path, artifact = load_prediction_artifact(payload, default_model_path=config.model_path)
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping validation for as_of_date=%s missing_artifact=%s", payload.get("as_of_date"), exc)
            pending += 1
            continue
        bundle_key = str(model_path.resolve())
        bundle = bundle_cache.get(bundle_key)
        if bundle is None:
            contract_config = contract_config_from_artifact(config, artifact)
            bundle = load_market_bundle(contract_config, force=False)
            bundle_cache[bundle_key] = bundle
        realized_map = realized_day_lookup(bundle, payload["as_of_date"])
        selected_code = payload["selected_code"]
        if not realized_map or selected_code not in realized_map:
            pending += 1
            continue

        values = list(realized_map.values())
        selected_return = float(realized_map[selected_code])
        universe_return = float(sum(values) / len(values))
        oracle_return = float(max(values))
        alpha = selected_return - universe_return
        row = {
            "archive_id": payload["archive_id"],
            "prediction_name": payload["prediction_name"],
            "as_of_date": payload["as_of_date"],
            "entry_date": payload["entry_date"],
            "exit_date": payload["exit_date"],
            "selected_code": selected_code,
            "selected_score": payload["selected_score"],
            "selected_return": selected_return,
            "universe_return": universe_return,
            "oracle_return": oracle_return,
            "alpha": alpha,
            "hit": int(selected_return > 0),
        }
        history_rows_by_date[payload["as_of_date"]] = row
        if payload.get("status") == "validated" and payload.get("validation") == row:
            continue

        payload["status"] = "validated"
        payload["validation"] = row
        canonical = canonical_prediction_payload(payload)
        write_json(config.prediction_daily_path(payload["as_of_date"]), canonical)
        update_legacy_run_archive(config, payload.get("archive_id"), status="validated", validation=row)
        rows.append(row)
        validated += 1

    if history_rows_by_date:
        history = pd.DataFrame(history_rows_by_date.values()).sort_values("as_of_date").reset_index(drop=True)
        history.to_csv(history_path, index=False)
        LOGGER.info(
            "Validation history updated path=%s new_rows=%s total_rows=%s",
            history_path,
            len(rows),
            len(history),
        )
    else:
        LOGGER.info("No new validations were written history_path=%s", history_path)
    refresh_prediction_views(config)
    LOGGER.info("Validate pipeline finished validated=%s pending=%s", validated, pending)
    return {
        "validated": validated,
        "still_pending": pending,
        "history_path": str(history_path),
    }
