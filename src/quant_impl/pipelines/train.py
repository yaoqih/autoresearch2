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
    simulate_sleeve_equity,
    summarize_period,
    transform_training_targets,
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


class NoOpGradScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    def build_grad_scaler(device_type: str):
        if device_type != "cuda":
            return NoOpGradScaler()
        return torch.amp.GradScaler(device_type, enabled=True)
else:
    def build_grad_scaler(device_type: str):
        if device_type != "cuda":
            return NoOpGradScaler()
        return torch.cuda.amp.GradScaler(enabled=True)


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


def _merge_reports(reports: list[dict[str, object]], config: AppConfig) -> dict[str, object]:
    rows: list[tuple[str, float, float, float]] = []
    for report in reports:
        daily = report["daily"]
        rows.extend(
            zip(
                daily["dates"],
                daily["selected_returns"],
                daily["universe_returns"],
                daily["oracle_returns"],
            )
        )
    rows.sort(key=lambda item: item[0])
    if not rows:
        return summarize_period(config.data, [], [], [], [])
    return summarize_period(
        config.data,
        [row[0] for row in rows],
        [float(row[1]) for row in rows],
        [float(row[2]) for row in rows],
        [float(row[3]) for row in rows],
    )


def _weighted_merge_reports(reports: list[dict[str, object]], weights: list[float], config: AppConfig) -> dict[str, object]:
    rows: list[tuple[str, float, float, float]] = []
    for report, weight in zip(reports, weights):
        daily = report["daily"]
        repeats = max(1, int(round(weight * 4)))
        for _ in range(repeats):
            rows.extend(
                zip(
                    daily["dates"],
                    daily["selected_returns"],
                    daily["universe_returns"],
                    daily["oracle_returns"],
                )
            )
    rows.sort(key=lambda item: item[0])
    if not rows:
        return summarize_period(config.data, [], [], [], [])
    return summarize_period(
        config.data,
        [row[0] for row in rows],
        [float(row[1]) for row in rows],
        [float(row[2]) for row in rows],
        [float(row[3]) for row in rows],
    )


def _report_log_equity(report: dict[str, object], config: AppConfig) -> float:
    returns = report["daily"]["selected_returns"]
    if not returns:
        return 0.0
    equity = simulate_sleeve_equity(returns, config.data.holding_days)
    return float(np.log(np.clip(equity[-1], 1e-12, None)))


def _report_ret_dd_ratio(report: dict[str, object]) -> float:
    metrics = report["metrics"]
    drawdown = abs(float(metrics["max_drawdown"]))
    if drawdown <= 1e-12:
        return 0.0
    return float(metrics["mean_return"]) / drawdown


def fit_one_window(
    bundle,
    config: AppConfig,
    device: torch.device,
    train_day_indices: list[int],
    valid_day_indices: list[int],
    *,
    seed_base: int,
    window_name: str,
) -> tuple[CrossSectionalRanker, dict[str, object]]:
    LOGGER.info(
        "Training window start name=%s train_days=%s valid_days=%s device=%s",
        window_name,
        len(train_day_indices),
        len(valid_day_indices),
        device,
    )
    linear_weights = compute_linear_ic_weights(bundle, config.data, config.training, train_day_indices)
    model = CrossSectionalRanker(bundle["features"].shape[1], config.model, linear_weights).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = build_grad_scaler(device.type)

    best_state = _clone_state_dict(model)
    best_valid = float("-inf")
    best_epoch = 0
    best_metrics = None
    patience = 0
    history = []
    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in make_day_batches(
            bundle,
            day_indices=train_day_indices,
            batch_days=config.training.batch_days,
            clip_value=config.data.normalized_clip,
            shuffle=True,
            seed=seed_base + epoch,
            target_abs_cap=config.training.train_target_abs_cap,
        ):
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device)
            raw_targets = batch["targets"].to(device)
            targets = transform_training_targets(raw_targets, batch["group_sizes"], config.training)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model.forward_components(features)
                loss, loss_parts = compute_training_loss(
                    model,
                    outputs,
                    targets,
                    batch["group_sizes"],
                    config.model,
                    config.data,
                    config.training,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "listwise": float(loss_parts["listwise"].detach().cpu().item()),
                    "pairwise": float(loss_parts["pairwise"].detach().cpu().item()),
                    "huber": float(loss_parts["huber"].detach().cpu().item()),
                    "binary": float(loss_parts["binary"].detach().cpu().item()),
                    "winner": float(loss_parts["winner"].detach().cpu().item()),
                    "rerank_listwise": float(loss_parts["rerank_listwise"].detach().cpu().item()),
                }
            )

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
                "train_listwise_loss": float(np.mean([item["listwise"] for item in epoch_losses])) if epoch_losses else 0.0,
                "train_pairwise_loss": float(np.mean([item["pairwise"] for item in epoch_losses])) if epoch_losses else 0.0,
                "train_huber_loss": float(np.mean([item["huber"] for item in epoch_losses])) if epoch_losses else 0.0,
                "train_binary_loss": float(np.mean([item["binary"] for item in epoch_losses])) if epoch_losses else 0.0,
                "train_winner_loss": float(np.mean([item["winner"] for item in epoch_losses])) if epoch_losses else 0.0,
                "train_rerank_listwise_loss": float(np.mean([item["rerank_listwise"] for item in epoch_losses])) if epoch_losses else 0.0,
                "valid_selection_score": valid_score,
                "valid_mean_return": valid_eval["metrics"]["mean_return"],
                "valid_mean_alpha": valid_eval["metrics"]["mean_alpha"],
            }
        )
        LOGGER.info(
            "Window epoch name=%s epoch=%s/%s train_loss=%.6f valid_selection=%.6f valid_alpha=%.6f",
            window_name,
            epoch,
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
                    epoch,
                    config.training.early_stopping_patience,
                )
                break

    model.load_state_dict(best_state)
    LOGGER.info(
        "Training window finished name=%s best_epoch=%s best_selection=%.6f",
        window_name,
        best_epoch,
        best_valid,
    )
    return model, {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_metrics,
        "history": history,
    }


def _build_training_summary(config: AppConfig, fold_summaries: list[dict[str, object]]) -> dict[str, object]:
    valid_reports = [fold["valid"] for fold in fold_summaries]
    holdout_reports = [fold["holdout"] for fold in fold_summaries]
    aggregate_valid = _merge_reports(valid_reports, config)
    aggregate_holdout = _merge_reports(holdout_reports, config)

    holdout_weights = [1.0] * len(holdout_reports)
    recent_window = min(config.training.recent_holdout_folds, len(holdout_weights))
    for index in range(max(0, len(holdout_weights) - recent_window), len(holdout_weights)):
        holdout_weights[index] = config.training.recent_holdout_weight
    recent_holdout = _weighted_merge_reports(holdout_reports, holdout_weights, config)

    valid_metrics = aggregate_valid["metrics"]
    holdout_metrics = aggregate_holdout["metrics"]
    recent_metrics = recent_holdout["metrics"]
    research_score = float(
        np.average(
            [
                float(holdout_metrics["selection_score"]),
                float(recent_metrics["selection_score"]),
                float(valid_metrics["selection_score"]),
            ],
            weights=[
                config.training.research_holdout_weight,
                config.training.research_recent_weight,
                config.training.research_valid_weight,
            ],
        )
    )
    return {
        "research_score": research_score,
        "cv_valid_selection_score": float(valid_metrics["selection_score"]),
        "cv_valid_mean_return": float(valid_metrics["mean_return"]),
        "cv_valid_mean_alpha": float(valid_metrics["mean_alpha"]),
        "cv_valid_hit_rate": float(valid_metrics["hit_rate"]),
        "cv_valid_max_drawdown": float(valid_metrics["max_drawdown"]),
        "cv_valid_ret_dd": _report_ret_dd_ratio(aggregate_valid),
        "cv_valid_log_equity": _report_log_equity(aggregate_valid, config),
        "cv_holdout_selection_score": float(holdout_metrics["selection_score"]),
        "cv_holdout_mean_return": float(holdout_metrics["mean_return"]),
        "cv_holdout_mean_alpha": float(holdout_metrics["mean_alpha"]),
        "cv_holdout_hit_rate": float(holdout_metrics["hit_rate"]),
        "cv_holdout_max_drawdown": float(holdout_metrics["max_drawdown"]),
        "cv_holdout_ret_dd": _report_ret_dd_ratio(aggregate_holdout),
        "cv_holdout_log_equity": _report_log_equity(aggregate_holdout, config),
        "cv_holdout_cumulative_return": float(
            np.exp(_report_log_equity(aggregate_holdout, config)) if holdout_reports else 1.0
        ),
        "recent_holdout_selection_score": float(recent_metrics["selection_score"]),
        "recent_holdout_mean_return": float(recent_metrics["mean_return"]),
        "recent_holdout_mean_alpha": float(recent_metrics["mean_alpha"]),
        "recent_holdout_hit_rate": float(recent_metrics["hit_rate"]),
        "recent_holdout_max_drawdown": float(recent_metrics["max_drawdown"]),
        "recent_holdout_ret_dd": _report_ret_dd_ratio(recent_holdout),
        "recent_holdout_log_equity": _report_log_equity(recent_holdout, config),
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
    LOGGER.info(
        "Champion spec member=%s temporal_mode=%s target_transform=%s train_target_abs_cap=%.4f",
        runtime_config.training.member_config,
        runtime_config.training.temporal_mode,
        runtime_config.training.target_transform,
        runtime_config.training.train_target_abs_cap,
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
        fold_seed = runtime_config.training.seed + split.fold_id
        seed_everything(fold_seed)
        LOGGER.info(
            "Fold start fold_id=%s seed=%s train=%s..%s valid=%s..%s holdout=%s..%s",
            split.fold_id,
            fold_seed,
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
            seed_base=fold_seed,
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
    deployment_seed = runtime_config.training.seed + len(splits)
    seed_everything(deployment_seed)
    deployment_model, deployment_fit = fit_one_window(
        bundle,
        runtime_config,
        device_obj,
        deployment_train_days,
        deployment_valid_indices,
        seed_base=deployment_seed,
        window_name="deployment",
    )
    artifact = {
        "created_at": utc_timestamp(),
        "profile": profile,
        "champion_spec": {
            "member_config": runtime_config.training.member_config,
            "temporal_mode": runtime_config.training.temporal_mode,
            "target_transform": runtime_config.training.target_transform,
            "train_target_abs_cap": runtime_config.training.train_target_abs_cap,
            "train_target_cap_applies_to_linear_head": runtime_config.training.train_target_cap_applies_to_linear_head,
        },
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
        "champion_spec": artifact["champion_spec"],
    }
    write_json(runtime_config.training_metrics_path, result)
    LOGGER.info("Training metrics saved path=%s", runtime_config.training_metrics_path)
    return result
