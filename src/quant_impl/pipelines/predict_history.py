from __future__ import annotations

import logging

from quant_impl.data.market import load_market_bundle, resolve_bundle_date_window
from quant_impl.pipelines.model_contract import contract_config_from_artifact, load_model_artifact
from quant_impl.pipelines.predict import _load_model, predict_bundle_day, predict_pipeline
from quant_impl.pipelines.validate import validate_pipeline
from quant_impl.settings import AppConfig
import torch


LOGGER = logging.getLogger(__name__)


def predict_history_pipeline(
    config: AppConfig,
    *,
    device: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    anchor_date: str | None = None,
    lookback_months: int | None = None,
    validate: bool = False,
    limit_stocks: int | None = None,
) -> dict[str, object]:
    uses_explicit = start_date is not None or end_date is not None
    uses_lookback = anchor_date is not None or lookback_months is not None
    if uses_explicit == uses_lookback:
        raise ValueError("Provide either start/end dates or anchor_date with lookback_months")
    if uses_lookback and (anchor_date is None or lookback_months is None):
        raise ValueError("anchor_date and lookback_months are required together")

    device_obj = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    artifact = load_model_artifact(config.model_path)
    contract_config = contract_config_from_artifact(config, artifact)
    bundle = load_market_bundle(contract_config, force=False)
    window = resolve_bundle_date_window(
        bundle,
        start_date=start_date,
        end_date=end_date,
        anchor_date=anchor_date,
        lookback_months=lookback_months,
    )
    dates = list(window["dates"])
    LOGGER.info(
        "Predict history pipeline start dates=%s range=%s..%s validate=%s device=%s",
        len(dates),
        window["resolved_start_date"],
        window["resolved_end_date"],
        validate,
        device_obj,
    )

    model, loaded_artifact = _load_model(config, device_obj)
    predictions = []
    for day_index in window["day_indices"]:
        predictions.append(
            predict_bundle_day(
                config,
                model,
                loaded_artifact,
                contract_config,
                bundle=bundle,
                device=device_obj,
                day_index=day_index,
            )
        )

    validation_result = validate_pipeline(config) if validate else None
    return {
        "predicted": len(predictions),
        "dates": [prediction["as_of_date"] for prediction in predictions],
        "window": {
            key: value
            for key, value in window.items()
            if key not in {"day_indices", "dates"}
        },
        "validation": validation_result,
    }
