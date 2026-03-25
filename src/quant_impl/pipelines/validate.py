from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from quant_impl.data.market import load_market_bundle, realized_day_detail_lookup
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
VALIDATION_SCHEMA_VERSION = 2


def _candidate_validation_payload(
    detail: dict[str, float | bool] | None,
    *,
    executed: bool,
) -> dict[str, Any] | None:
    if detail is None:
        return None
    open_limit_day1 = bool(detail["open_limit_day1"])
    one_word_day1 = bool(detail["one_word_day1"])
    return {
        "ideal_return": float(detail["ideal_return"]),
        "strict_open_return": float(detail["strict_open_return"]),
        "strict_one_word_return": float(detail["strict_one_word_return"]),
        "open_limit_day1": int(open_limit_day1),
        "one_word_day1": int(one_word_day1),
        "tradeable": int(not open_limit_day1),
        "executed": bool(executed),
    }


def _enrich_top_candidates(
    top_candidates: list[dict[str, Any]],
    realized_map: dict[str, dict[str, float | bool]],
    *,
    max_fallback_rank: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    executed_index: int | None = None

    for index, candidate in enumerate(top_candidates):
        code = str(candidate["code"])
        detail = realized_map.get(code)
        tradeable = bool(detail is not None and not bool(detail["open_limit_day1"]))
        if executed_index is None and index < max_fallback_rank and tradeable:
            executed_index = index

    for index, candidate in enumerate(top_candidates):
        detail = realized_map.get(str(candidate["code"]))
        enriched.append(
            {
                **candidate,
                "validation": _candidate_validation_payload(detail, executed=index == executed_index),
            }
        )

    executed_candidate = enriched[executed_index] if executed_index is not None else None
    executed_validation = executed_candidate["validation"] if executed_candidate is not None else None
    fallback_window = min(max_fallback_rank, len(top_candidates))
    return enriched, {
        "executed_code": executed_candidate["code"] if executed_candidate is not None else None,
        "executed_rank": int(executed_candidate["rank"]) if executed_candidate is not None else None,
        "executed_score": float(executed_candidate["score"]) if executed_candidate is not None else None,
        "executed_return": (
            float(executed_validation["strict_open_return"])
            if executed_validation is not None
            else 0.0
        ),
        "executed_ideal_return": (
            float(executed_validation["ideal_return"])
            if executed_validation is not None
            else 0.0
        ),
        "fallback_applied": int(executed_index is not None and executed_index > 0),
        "all_top10_blocked": int(executed_index is None and fallback_window > 0),
    }


def _has_candidate_level_validation(payload: dict[str, Any]) -> bool:
    top_candidates = payload.get("top_candidates") or []
    if not top_candidates:
        return True
    return all(isinstance(candidate.get("validation"), dict) for candidate in top_candidates)


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
        realized_map = realized_day_detail_lookup(bundle, payload["as_of_date"])
        selected_code = payload["selected_code"]
        if not realized_map or selected_code not in realized_map:
            pending += 1
            continue

        ideal_values = [float(item["ideal_return"]) for item in realized_map.values()]
        strict_open_values = [float(item["strict_open_return"]) for item in realized_map.values()]
        top_candidates = list(payload.get("top_candidates") or [])
        enriched_top_candidates, executed = _enrich_top_candidates(top_candidates, realized_map)
        selected = realized_map[selected_code]
        selected_return = float(executed["executed_return"])
        universe_return = float(sum(ideal_values) / len(ideal_values))
        oracle_return = float(max(strict_open_values))
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
            "selected_ideal_return": float(executed["executed_ideal_return"]),
            "universe_return": universe_return,
            "oracle_return": oracle_return,
            "oracle_ideal_return": float(max(ideal_values)),
            "alpha": alpha,
            "hit": int(selected_return > 0),
            "open_limit_day1": int(bool(selected["open_limit_day1"])),
            "one_word_day1": int(bool(selected["one_word_day1"])),
            "tradeable": int(not bool(selected["open_limit_day1"])),
            "executed_code": executed["executed_code"],
            "executed_rank": executed["executed_rank"],
            "executed_score": executed["executed_score"],
            "fallback_applied": int(executed["fallback_applied"]),
            "all_top10_blocked": int(executed["all_top10_blocked"]),
            "schema_version": VALIDATION_SCHEMA_VERSION,
        }
        history_rows_by_date[payload["as_of_date"]] = row
        payload_validation = payload.get("validation") or {}
        schema_matches = payload_validation.get("schema_version") == VALIDATION_SCHEMA_VERSION
        if (
            payload.get("status") == "validated"
            and payload_validation == row
            and schema_matches
            and _has_candidate_level_validation(payload)
        ):
            continue

        payload["status"] = "validated"
        payload["validation"] = row
        payload["top_candidates"] = enriched_top_candidates
        canonical = canonical_prediction_payload(payload)
        write_json(config.prediction_daily_path(payload["as_of_date"]), canonical)
        update_legacy_run_archive(
            config,
            payload.get("archive_id"),
            status="validated",
            validation=row,
            top_candidates=enriched_top_candidates,
        )
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
