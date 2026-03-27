from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from quant_impl.settings import AppConfig
from quant_impl.utils.io import read_json, write_json, utc_timestamp


LOGGER = logging.getLogger(__name__)


def _selected_code(payload: dict[str, Any]) -> str | None:
    if payload.get("selected_code") is not None:
        return str(payload["selected_code"])
    selected = payload.get("selected")
    if isinstance(selected, dict) and selected.get("code") is not None:
        return str(selected["code"])
    return None


def _selected_score(payload: dict[str, Any]) -> float | None:
    if payload.get("selected_score") is not None:
        return float(payload["selected_score"])
    selected = payload.get("selected")
    if isinstance(selected, dict) and selected.get("score") is not None:
        return float(selected["score"])
    return None


def _summary_from_validation(validation: dict[str, Any] | None) -> dict[str, Any]:
    if not validation:
        return {
            "hit": None,
            "alpha": None,
            "selected_return": None,
            "universe_return": None,
            "oracle_return": None,
        }
    return {
        "hit": validation.get("hit"),
        "alpha": validation.get("alpha"),
        "selected_return": validation.get("selected_return"),
        "universe_return": validation.get("universe_return"),
        "oracle_return": validation.get("oracle_return"),
    }


def canonical_prediction_payload(
    payload: dict[str, Any],
    *,
    updated_at: str | None = None,
) -> dict[str, Any]:
    selected_code = _selected_code(payload)
    selected_score = _selected_score(payload)
    validation = payload.get("validation")
    top_candidates = payload.get("top_candidates") or []
    if selected_code is None and top_candidates:
        first = top_candidates[0]
        selected_code = str(first.get("code")) if first.get("code") is not None else None
        if selected_score is None and first.get("score") is not None:
            selected_score = float(first["score"])
    canonical = {
        "archive_id": payload.get("archive_id"),
        "prediction_name": payload.get("prediction_name"),
        "model_path": payload.get("model_path"),
        "model_created_at": payload.get("model_created_at"),
        "as_of_date": payload["as_of_date"],
        "entry_date": payload.get("entry_date"),
        "exit_date": payload.get("exit_date"),
        "status": payload.get("status", "pending"),
        "universe_size": payload.get("universe_size"),
        "top_k": payload.get("top_k"),
        "execution_fallback_top_k": payload.get("execution_fallback_top_k"),
        "execution_block_mode": payload.get("execution_block_mode"),
        "selected_code": selected_code,
        "selected_score": selected_score,
        "selected": {
            "code": selected_code,
            "score": selected_score,
        },
        "top_candidates": top_candidates,
        "validation": validation,
        "summary": _summary_from_validation(validation),
        "updated_at": updated_at or payload.get("updated_at") or utc_timestamp(),
    }
    return canonical


def prediction_index_record(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    validation = payload.get("validation") or {}
    return {
        "as_of_date": payload["as_of_date"],
        "archive_id": payload.get("archive_id"),
        "prediction_name": payload.get("prediction_name"),
        "status": payload.get("status"),
        "entry_date": payload.get("entry_date"),
        "exit_date": payload.get("exit_date"),
        "selected_code": _selected_code(payload),
        "selected_score": _selected_score(payload),
        "execution_fallback_top_k": payload.get("execution_fallback_top_k"),
        "execution_block_mode": payload.get("execution_block_mode"),
        "hit": summary.get("hit"),
        "alpha": summary.get("alpha"),
        "selected_return": summary.get("selected_return"),
        "executed_code": validation.get("executed_code"),
        "executed_rank": validation.get("executed_rank"),
        "fallback_applied": validation.get("fallback_applied"),
        "fallback_window_size": validation.get("fallback_window_size"),
        "all_fallback_blocked": validation.get("all_fallback_blocked"),
        "all_top10_blocked": validation.get("all_top10_blocked"),
        "updated_at": payload.get("updated_at"),
    }


def write_canonical_daily_prediction(config: AppConfig, payload: dict[str, Any]) -> Path:
    canonical = canonical_prediction_payload(payload, updated_at=utc_timestamp())
    path = config.prediction_daily_path(canonical["as_of_date"])
    write_json(path, canonical)
    return path


def _legacy_prediction_files(predictions_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in predictions_dir.glob("*/prediction.json")
        if path.parent.name != "daily"
    )


def backfill_daily_predictions_from_legacy(config: AppConfig) -> int:
    latest_by_date: dict[str, dict[str, Any]] = {}
    for path in _legacy_prediction_files(config.paths.predictions_dir):
        payload = read_json(path)
        if not payload or not payload.get("as_of_date"):
            continue
        as_of_date = str(payload["as_of_date"])
        existing = latest_by_date.get(as_of_date)
        if existing is None or str(payload.get("archive_id", "")) > str(existing.get("archive_id", "")):
            latest_by_date[as_of_date] = payload

    created = 0
    for as_of_date, payload in latest_by_date.items():
        daily_path = config.prediction_daily_path(as_of_date)
        if daily_path.exists():
            continue
        write_json(daily_path, canonical_prediction_payload(payload))
        created += 1

    if created:
        LOGGER.info("Backfilled canonical daily predictions from legacy archives count=%s", created)
    return created


def load_canonical_daily_predictions(config: AppConfig) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(config.prediction_daily_dir.glob("*.json")):
        payload = read_json(path)
        if not payload:
            continue
        canonical = canonical_prediction_payload(payload)
        if canonical != payload:
            write_json(path, canonical)
        payloads.append(canonical)
    payloads.sort(key=lambda item: item["as_of_date"], reverse=True)
    return payloads


def refresh_prediction_views(config: AppConfig) -> list[dict[str, Any]]:
    payloads = load_canonical_daily_predictions(config)
    index_payload = [prediction_index_record(payload) for payload in payloads]
    write_json(config.prediction_index_path, index_payload)
    write_json(config.prediction_latest_path, payloads[0] if payloads else {})
    return payloads


def sync_prediction_website_archives(config: AppConfig) -> list[dict[str, Any]]:
    backfill_daily_predictions_from_legacy(config)
    return refresh_prediction_views(config)


def update_legacy_run_archive(
    config: AppConfig,
    archive_id: str | None,
    *,
    status: str,
    validation: dict[str, Any] | None,
    top_candidates: list[dict[str, Any]] | None = None,
) -> None:
    if not archive_id:
        return
    path = config.paths.predictions_dir / archive_id / "prediction.json"
    payload = read_json(path)
    if not payload:
        return
    payload["status"] = status
    if validation is not None:
        payload["validation"] = validation
    if top_candidates is not None:
        payload["top_candidates"] = top_candidates
    write_json(path, payload)
