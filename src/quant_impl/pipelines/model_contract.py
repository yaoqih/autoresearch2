from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from quant_impl.settings import AppConfig, DataSettings, ModelSettings


def _merge_dataclass_payload(instance: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if isinstance(current, tuple):
            setattr(instance, key, tuple(value))
        elif isinstance(current, Path):
            setattr(instance, key, Path(value))
        else:
            setattr(instance, key, value)
    return instance


def load_model_artifact(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def model_settings_from_artifact(artifact: dict[str, Any]) -> ModelSettings:
    return _merge_dataclass_payload(ModelSettings(), artifact.get("model_config", {}))


def data_settings_from_artifact(artifact: dict[str, Any]) -> DataSettings:
    return _merge_dataclass_payload(DataSettings(), artifact.get("data_config", {}))


def contract_config_from_artifact(base_config: AppConfig, artifact: dict[str, Any]) -> AppConfig:
    cloned = deepcopy(base_config)
    cloned.data = data_settings_from_artifact(artifact)
    return cloned


def load_prediction_artifact(
    payload: dict[str, Any],
    *,
    default_model_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    model_path = payload.get("model_path") or default_model_path
    if not model_path:
        raise FileNotFoundError("Prediction payload does not contain model_path")
    path = Path(model_path)
    return path, load_model_artifact(path)
