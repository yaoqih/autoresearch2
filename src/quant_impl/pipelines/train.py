from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import logging
import random

import numpy as np
import torch
from torch import nn

from quant_impl.data.market import (
    build_market_cache,
    build_walk_forward_splits,
    compute_linear_ic_weights,
    evaluate_ranker,
    feature_columns,
    load_market_bundle,
    make_day_batches,
)
from quant_impl.modeling.ranker import CrossSectionalRanker, compute_training_loss
from quant_impl.settings import AppConfig
from quant_impl.utils.io import utc_timestamp, write_json


LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def config_for_profile(config: AppConfig, profile: str) -> tuple[AppConfig, int | None]:
    cloned = deepcopy(config)
    max_folds = None
    if profile == "screen":
        cloned.training.epochs = min(cloned.training.epochs, 4)
        max_folds = 3
    elif profile == "probe":
        cloned.training.epochs = min(cloned.training.epochs, 8)
        max_folds = 6
    return cloned, max_folds


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_one_window(
    bundle,
    config: AppConfig,
    device: torch.device,
    train_day_indices: list[int],
    valid_day_indices: list[int],
    *,
    window_name: str,
) -> tuple[CrossSectionalRanker, dict[str, object]]:
    LOGGER.info(
        "Training window start name=%s train_days=%s valid_days=%s device=%s",
        window_name,
        len(train_day_indices),
        len(valid_day_indices),
        device,
    )
    linear_weights = compute_linear_ic_weights(bundle, config.data, train_day_indices)
    model = CrossSectionalRanker(bundle["features"].shape[1], config.model, linear_weights).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_state = _clone_state_dict(model)
    best_valid = float("-inf")
    best_epoch = 0
    best_metrics = None
    patience = 0
    history = []
    autocast_enabled = device.type == "cuda"

    for epoch in range(config.training.epochs):
        model.train()
        epoch_losses = []
        for batch in make_day_batches(
            bundle,
            day_indices=train_day_indices,
            batch_days=config.training.batch_days,
            clip_value=config.data.normalized_clip,
            shuffle=True,
            seed=config.training.seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model.day_outputs(features, batch["group_sizes"])
                loss, loss_parts = compute_training_loss(outputs, targets, batch["group_sizes"], config.training)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append({"loss": float(loss.detach().cpu().item()), **loss_parts})

        valid_eval = evaluate_ranker(
            model,
            bundle,
            config.data,
            valid_day_indices,
            device,
            top_k=config.inference.top_k,
            batch_days=config.training.batch_days,
        )
        valid_score = valid_eval["metrics"]["selection_score"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean([item["loss"] for item in epoch_losses])) if epoch_losses else 0.0,
                "valid_selection_score": valid_score,
                "valid_mean_return": valid_eval["metrics"]["mean_return"],
                "valid_mean_alpha": valid_eval["metrics"]["mean_alpha"],
            }
        )
        LOGGER.info(
            "Window epoch name=%s epoch=%s/%s train_loss=%.6f valid_selection=%.6f valid_alpha=%.6f",
            window_name,
            epoch + 1,
            config.training.epochs,
            history[-1]["train_loss"],
            valid_score,
            valid_eval["metrics"]["mean_alpha"],
        )
        if valid_score > best_valid:
            best_valid = valid_score
            best_epoch = epoch
            best_state = _clone_state_dict(model)
            best_metrics = valid_eval
            patience = 0
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience:
                LOGGER.info(
                    "Window early stop name=%s epoch=%s patience=%s",
                    window_name,
                    epoch + 1,
                    config.training.early_stopping_patience,
                )
                break

    model.load_state_dict(best_state)
    LOGGER.info(
        "Training window finished name=%s best_epoch=%s best_selection=%.6f",
        window_name,
        best_epoch + 1,
        best_valid,
    )
    return model, {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_metrics,
        "history": history,
    }


def _aggregate_metric(fold_summaries: list[dict[str, object]], section: str, metric_name: str) -> float:
    values = [fold[section]["metrics"][metric_name] for fold in fold_summaries]
    return float(np.mean(values)) if values else 0.0


def _build_training_summary(config: AppConfig, fold_summaries: list[dict[str, object]]) -> dict[str, object]:
    recent = fold_summaries[-config.training.recent_holdout_folds :] if fold_summaries else []
    valid_selection = _aggregate_metric(fold_summaries, "valid", "selection_score")
    holdout_selection = _aggregate_metric(fold_summaries, "holdout", "selection_score")
    recent_selection = _aggregate_metric(recent, "holdout", "selection_score") if recent else holdout_selection
    research_score = config.training.research_holdout_weight * holdout_selection
    research_score += config.training.research_recent_weight * recent_selection
    research_score += config.training.research_valid_weight * valid_selection
    return {
        "research_score": float(research_score),
        "cv_valid_selection_score": valid_selection,
        "cv_valid_mean_return": _aggregate_metric(fold_summaries, "valid", "mean_return"),
        "cv_valid_mean_alpha": _aggregate_metric(fold_summaries, "valid", "mean_alpha"),
        "cv_holdout_selection_score": holdout_selection,
        "cv_holdout_mean_return": _aggregate_metric(fold_summaries, "holdout", "mean_return"),
        "cv_holdout_mean_alpha": _aggregate_metric(fold_summaries, "holdout", "mean_alpha"),
        "cv_holdout_hit_rate": _aggregate_metric(fold_summaries, "holdout", "hit_rate"),
        "cv_holdout_max_drawdown": _aggregate_metric(fold_summaries, "holdout", "max_drawdown"),
        "recent_holdout_selection_score": recent_selection,
        "recent_holdout_mean_return": _aggregate_metric(recent, "holdout", "mean_return") if recent else 0.0,
        "recent_holdout_mean_alpha": _aggregate_metric(recent, "holdout", "mean_alpha") if recent else 0.0,
    }


def train_pipeline(
    config: AppConfig,
    *,
    device: str | None = None,
    profile: str = "full",
    force_prepare: bool = False,
    limit_stocks: int | None = None,
) -> dict[str, object]:
    runtime_config, max_folds = config_for_profile(config, profile)
    seed_everything(runtime_config.training.seed)
    LOGGER.info(
        "Train pipeline start profile=%s device=%s force_prepare=%s limit_stocks=%s seed=%s",
        profile,
        device,
        force_prepare,
        limit_stocks,
        runtime_config.training.seed,
    )
    build_market_cache(runtime_config, force=force_prepare, limit_stocks=limit_stocks)
    bundle = load_market_bundle(runtime_config, force=False, limit_stocks=limit_stocks)
    splits = build_walk_forward_splits(bundle, runtime_config.data)
    if max_folds is not None:
        splits = splits[-max_folds:]
    device_obj = resolve_device(device)
    LOGGER.info(
        "Training bundle loaded assets=%s days=%s features=%s folds=%s resolved_device=%s",
        len(bundle["assets"]),
        len(bundle["dates"]),
        len(bundle["feature_names"]),
        len(splits),
        device_obj,
    )

    fold_summaries = []
    for split in splits:
        LOGGER.info(
            "Fold start fold_id=%s train=%s..%s valid=%s..%s holdout=%s..%s",
            split.fold_id,
            split.train_start_date,
            split.train_end_date,
            split.valid_start_date,
            split.valid_end_date,
            split.holdout_start_date,
            split.holdout_end_date,
        )
        train_day_indices = list(range(split.train_start, split.train_end))
        valid_day_indices = list(range(split.valid_start, split.valid_end))
        holdout_day_indices = list(range(split.holdout_start, split.holdout_end))
        model, fit_summary = fit_one_window(
            bundle,
            runtime_config,
            device_obj,
            train_day_indices,
            valid_day_indices,
            window_name=f"fold-{split.fold_id}",
        )
        holdout_eval = evaluate_ranker(
            model,
            bundle,
            runtime_config.data,
            holdout_day_indices,
            device_obj,
            top_k=runtime_config.inference.top_k,
            batch_days=runtime_config.training.batch_days,
        )
        fold_summaries.append(
            {
                "split": asdict(split),
                "fit": fit_summary,
                "valid": fit_summary["best_valid_metrics"],
                "holdout": holdout_eval,
            }
        )
        LOGGER.info(
            "Fold finished fold_id=%s holdout_selection=%.6f holdout_alpha=%.6f",
            split.fold_id,
            holdout_eval["metrics"]["selection_score"],
            holdout_eval["metrics"]["mean_alpha"],
        )

    summary = _build_training_summary(runtime_config, fold_summaries)
    LOGGER.info(
        "Cross-validation summary research_score=%.6f holdout_selection=%.6f",
        summary["research_score"],
        summary["cv_holdout_selection_score"],
    )

    deployment_valid_days = min(runtime_config.data.valid_days, len(bundle["dates"]) // 5)
    deployment_train_days = list(range(0, max(1, len(bundle["dates"]) - deployment_valid_days)))
    deployment_valid_indices = list(range(max(1, len(bundle["dates"]) - deployment_valid_days), len(bundle["dates"])))
    deployment_model, deployment_fit = fit_one_window(
        bundle,
        runtime_config,
        device_obj,
        deployment_train_days,
        deployment_valid_indices,
        window_name="deployment",
    )
    artifact = {
        "created_at": utc_timestamp(),
        "profile": profile,
        "feature_names": feature_columns(runtime_config.data),
        "model_config": asdict(runtime_config.model),
        "data_config": asdict(runtime_config.data),
        "training_config": asdict(runtime_config.training),
        "state_dict": deployment_model.state_dict(),
        "summary": summary,
        "deployment_fit": deployment_fit,
    }
    torch.save(artifact, runtime_config.model_path)
    LOGGER.info("Deployment model saved path=%s", runtime_config.model_path)

    result = {
        "model_path": str(runtime_config.model_path),
        "profile": profile,
        "summary": summary,
        "folds": fold_summaries,
        "deployment_fit": deployment_fit,
    }
    write_json(runtime_config.training_metrics_path, result)
    LOGGER.info("Training metrics saved path=%s", runtime_config.training_metrics_path)
    return result
