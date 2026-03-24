from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_impl.data.market import build_rank_weights, portfolio_size
from quant_impl.settings import DataSettings, ModelSettings, TrainingSettings


def build_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, *, activation: str, dropout: float, use_layernorm: bool, gated: bool):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate = nn.Linear(dim, dim) if gated else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = self.fc1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        if self.gate is not None:
            y = torch.sigmoid(self.gate(residual)) * y
        return residual + y


class Stage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, model_cfg: ModelSettings):
        super().__init__()
        self.transition = nn.Linear(in_dim, out_dim)
        self.transition_norm = nn.LayerNorm(out_dim) if model_cfg.use_layernorm else nn.Identity()
        self.activation = build_activation(model_cfg.activation)
        self.dropout = nn.Dropout(model_cfg.dropout) if model_cfg.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    out_dim,
                    activation=model_cfg.activation,
                    dropout=model_cfg.dropout,
                    use_layernorm=model_cfg.use_layernorm,
                    gated=True,
                )
                for _ in range(model_cfg.num_residual_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transition(x)
        x = self.transition_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return x


class ShortlistContextBlock(nn.Module):
    def __init__(self, dim: int, *, heads: int, dropout: float, activation: str):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=max(1, heads),
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            build_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_output)
        ffn_input = self.ffn_norm(x)
        x = x + self.dropout(self.ffn(ffn_input))
        return x


@dataclass
class DayScores:
    scores: torch.Tensor
    base_scores: torch.Tensor
    shortlist_idx: torch.Tensor


class CrossSectionalRanker(nn.Module):
    def __init__(self, input_dim: int, model_cfg: ModelSettings, linear_weights: torch.Tensor):
        super().__init__()
        self.model_cfg = model_cfg
        dims = [input_dim, *model_cfg.hidden_dims]
        self.input_norm = nn.LayerNorm(input_dim) if model_cfg.use_layernorm else nn.Identity()
        self.feature_gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.stages = nn.ModuleList(
            [Stage(in_dim, out_dim, model_cfg) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
        )
        self.head_norm = nn.LayerNorm(dims[-1]) if model_cfg.use_layernorm else nn.Identity()
        self.broad_head = nn.Linear(dims[-1], 1)
        self.linear_head = nn.Linear(input_dim, 1, bias=False)
        self.linear_head.weight.data.copy_(linear_weights.view(1, -1))
        self.linear_head.weight.requires_grad_(not model_cfg.freeze_linear_head)
        self.linear_head_scale = nn.Parameter(torch.tensor(float(model_cfg.linear_head_mix)))

        rerank_dim = max(16, int(model_cfg.rerank_dim))
        self.rerank_input = nn.Linear(dims[-1] + 2, rerank_dim)
        self.rerank_blocks = nn.ModuleList(
            [
                ShortlistContextBlock(
                    rerank_dim,
                    heads=model_cfg.rerank_heads,
                    dropout=model_cfg.dropout,
                    activation=model_cfg.activation,
                )
                for _ in range(max(1, model_cfg.rerank_blocks))
            ]
        )
        self.rerank_norm = nn.LayerNorm(rerank_dim)
        self.rerank_head = nn.Sequential(
            nn.Linear(rerank_dim, rerank_dim),
            build_activation(model_cfg.activation),
            nn.Dropout(model_cfg.dropout) if model_cfg.dropout > 0 else nn.Identity(),
            nn.Linear(rerank_dim, 1),
        )
        nn.init.zeros_(self.rerank_head[-1].weight)
        nn.init.zeros_(self.rerank_head[-1].bias)
        self.rerank_scale = nn.Parameter(torch.tensor(float(model_cfg.rerank_mix)))

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(features)
        linear_score = self.linear_head(x).squeeze(-1)
        x = x * (0.75 + 0.5 * self.feature_gate(x))
        for stage in self.stages:
            x = stage(x)
        x = self.head_norm(x)
        return x, linear_score

    def forward_components(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, linear_score = self.encode(features)
        linear_scale = torch.clamp(self.linear_head_scale, min=0.0, max=2.0)
        broad_residual = self.broad_head(latent).squeeze(-1)
        broad_score = broad_residual + linear_scale * linear_score
        rerank_input = torch.cat(
            [latent, broad_score.unsqueeze(-1), linear_score.unsqueeze(-1)],
            dim=-1,
        )
        rerank_latent = self.rerank_input(rerank_input)
        return {
            "linear_score": linear_score,
            "broad_residual": broad_residual,
            "broad_score": broad_score,
            "rerank_latent": rerank_latent,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_components(features)["broad_score"]

    def score_day(self, features: torch.Tensor) -> DayScores:
        predictions = self.forward_components(features)
        final_scores, shortlist_indices = rerank_shortlist_scores(
            self,
            predictions["broad_score"],
            predictions["linear_score"],
            predictions["rerank_latent"],
            [features.shape[0]],
        )
        shortlist_idx = (
            shortlist_indices[0]
            if shortlist_indices
            else torch.arange(features.shape[0], device=features.device)
        )
        return DayScores(
            scores=final_scores,
            base_scores=predictions["broad_score"],
            shortlist_idx=shortlist_idx,
        )

    def score_batch(self, features: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        predictions = self.forward_components(features)
        final_scores, _ = rerank_shortlist_scores(
            self,
            predictions["broad_score"],
            predictions["linear_score"],
            predictions["rerank_latent"],
            group_sizes,
        )
        return final_scores


def smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    mean_target = targets.mean()
    return targets * (1.0 - smoothing) + mean_target * smoothing


def weighted_mean_tensors(
    losses: list[torch.Tensor],
    weights: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if not losses:
        return torch.tensor(0.0, device=device)
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(device=device, dtype=loss_tensor.dtype)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-12)


def expanded_bucket_size(
    group_size: int,
    data_cfg: DataSettings,
    expansion_fraction: float,
    scale: float,
) -> int:
    base_k = portfolio_size(group_size, data_cfg)
    expanded_k = int(round(base_k * (1.0 + scale * max(0.0, expansion_fraction))))
    expanded_k = max(base_k, expanded_k)
    return max(1, min(group_size, expanded_k))


def listwise_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for group_size in group_sizes:
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        centered_targets = day_targets - day_targets.mean()
        scale = centered_targets.std(unbiased=False).clamp_min(1e-6)
        soft_probs = torch.softmax(centered_targets / scale, dim=0)
        portfolio_k = portfolio_size(group_size, data_cfg)
        top_indices = torch.topk(day_targets, k=portfolio_k, largest=True).indices
        top_weights = build_rank_weights(
            portfolio_k,
            data_cfg.portfolio_rank_decay,
            device=day_predictions.device,
            dtype=day_predictions.dtype,
        ).to(device=day_predictions.device, dtype=day_predictions.dtype)
        portfolio_probs = torch.zeros_like(day_predictions)
        portfolio_probs[top_indices] = top_weights
        target_probs = (
            training_cfg.listwise_target_blend * portfolio_probs
            + (1.0 - training_cfg.listwise_target_blend) * soft_probs
        )
        target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
        pred_log_probs = F.log_softmax(day_predictions, dim=0)
        losses.append(-(target_probs * pred_log_probs).sum())

    if not losses:
        return predictions.new_zeros(())
    return torch.stack(losses).mean()


def weighted_listwise_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    day_weights: torch.Tensor,
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    offset = 0
    for day_idx, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        centered_targets = day_targets - day_targets.mean()
        scale = centered_targets.std(unbiased=False).clamp_min(1e-6)
        soft_probs = torch.softmax(centered_targets / scale, dim=0)
        portfolio_k = portfolio_size(group_size, data_cfg)
        top_indices = torch.topk(day_targets, k=portfolio_k, largest=True).indices
        top_weights = build_rank_weights(
            portfolio_k,
            data_cfg.portfolio_rank_decay,
            device=day_predictions.device,
            dtype=day_predictions.dtype,
        ).to(device=day_predictions.device, dtype=day_predictions.dtype)
        portfolio_probs = torch.zeros_like(day_predictions)
        portfolio_probs[top_indices] = top_weights
        target_probs = (
            training_cfg.listwise_target_blend * portfolio_probs
            + (1.0 - training_cfg.listwise_target_blend) * soft_probs
        )
        target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
        pred_log_probs = F.log_softmax(day_predictions, dim=0)
        losses.append(-(target_probs * pred_log_probs).sum())
        weights.append(day_weights[day_idx])
    return weighted_mean_tensors(losses, weights, predictions.device)


def pairwise_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for group_size in group_sizes:
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        order = torch.argsort(day_targets, descending=True)
        focus_count = expanded_bucket_size(
            group_size,
            data_cfg,
            training_cfg.pair_focus_fraction,
            training_cfg.top_bucket_expansion_scale,
        )
        focus_index = order[:focus_count]
        rest_index = order[focus_count:]
        if rest_index.numel() == 0:
            continue

        pair_count = min(training_cfg.pair_samples_per_day, focus_index.numel() * rest_index.numel())
        left_index = focus_index[torch.randint(0, focus_index.numel(), (pair_count,), device=predictions.device)]
        right_index = rest_index[torch.randint(0, rest_index.numel(), (pair_count,), device=predictions.device)]
        pred_diff = day_predictions[left_index] - day_predictions[right_index]
        losses.append(F.softplus(-pred_diff).mean())

    if not losses:
        return predictions.new_zeros(())
    return torch.stack(losses).mean()


def weighted_pairwise_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    day_weights: torch.Tensor,
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    offset = 0
    for day_idx, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        order = torch.argsort(day_targets, descending=True)
        focus_count = expanded_bucket_size(
            group_size,
            data_cfg,
            training_cfg.pair_focus_fraction,
            training_cfg.top_bucket_expansion_scale,
        )
        focus_index = order[:focus_count]
        rest_index = order[focus_count:]
        if rest_index.numel() == 0:
            continue

        pair_count = min(training_cfg.pair_samples_per_day, focus_index.numel() * rest_index.numel())
        left_index = focus_index[torch.randint(0, focus_index.numel(), (pair_count,), device=predictions.device)]
        right_index = rest_index[torch.randint(0, rest_index.numel(), (pair_count,), device=predictions.device)]
        pred_diff = day_predictions[left_index] - day_predictions[right_index]
        losses.append(F.softplus(-pred_diff).mean())
        weights.append(day_weights[day_idx])
    return weighted_mean_tensors(losses, weights, predictions.device)


def weighted_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for group_size in group_sizes:
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset

        centered_targets = day_targets - day_targets.mean()
        scale = centered_targets.std(unbiased=False).clamp_min(1e-6)
        weights = 1.0 + centered_targets.abs() / scale
        loss = F.smooth_l1_loss(day_predictions, day_targets, reduction="none")
        losses.append((loss * weights).mean())

    if not losses:
        return predictions.new_zeros(())
    return torch.stack(losses).mean()


def weighted_huber_day_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    day_weights: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    offset = 0
    for day_idx, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset

        centered_targets = day_targets - day_targets.mean()
        scale = centered_targets.std(unbiased=False).clamp_min(1e-6)
        element_weights = 1.0 + centered_targets.abs() / scale
        loss = F.smooth_l1_loss(day_predictions, day_targets, reduction="none")
        losses.append((loss * element_weights).mean())
        weights.append(day_weights[day_idx])
    return weighted_mean_tensors(losses, weights, predictions.device)


def binary_top_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for group_size in group_sizes:
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        positive_count = expanded_bucket_size(
            group_size,
            data_cfg,
            training_cfg.positive_fraction,
            training_cfg.top_bucket_expansion_scale,
        )
        top_index = torch.topk(day_targets, k=positive_count, largest=True).indices
        labels = torch.zeros(group_size, device=predictions.device)
        labels[top_index] = build_rank_weights(
            positive_count,
            data_cfg.portfolio_rank_decay,
            device=predictions.device,
            dtype=labels.dtype,
        ).to(device=labels.device, dtype=labels.dtype)
        if training_cfg.label_smoothing > 0:
            labels = labels * (1.0 - training_cfg.label_smoothing) + 0.5 * training_cfg.label_smoothing
        pos_weight = torch.tensor(
            [(group_size - positive_count) / max(positive_count, 1)],
            device=predictions.device,
        )
        losses.append(
            F.binary_cross_entropy_with_logits(
                day_predictions,
                labels,
                pos_weight=pos_weight,
            )
        )

    if not losses:
        return predictions.new_zeros(())
    return torch.stack(losses).mean()


def weighted_binary_top_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    day_weights: torch.Tensor,
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    offset = 0
    for day_idx, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        positive_count = expanded_bucket_size(
            group_size,
            data_cfg,
            training_cfg.positive_fraction,
            training_cfg.top_bucket_expansion_scale,
        )
        top_index = torch.topk(day_targets, k=positive_count, largest=True).indices
        labels = torch.zeros(group_size, device=predictions.device)
        labels[top_index] = build_rank_weights(
            positive_count,
            data_cfg.portfolio_rank_decay,
            device=predictions.device,
            dtype=labels.dtype,
        ).to(device=labels.device, dtype=labels.dtype)
        if training_cfg.label_smoothing > 0:
            labels = labels * (1.0 - training_cfg.label_smoothing) + 0.5 * training_cfg.label_smoothing
        pos_weight = torch.tensor(
            [(group_size - positive_count) / max(positive_count, 1)],
            device=predictions.device,
        )
        losses.append(F.binary_cross_entropy_with_logits(day_predictions, labels, pos_weight=pos_weight))
        weights.append(day_weights[day_idx])
    return weighted_mean_tensors(losses, weights, predictions.device)


def select_focus_indices(
    day_scores: torch.Tensor,
    group_size: int,
    focus_size: int,
    include_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if group_size <= 1:
        return torch.arange(group_size, device=day_scores.device)

    focus_count = min(group_size, max(2, int(focus_size)))
    focus_index = torch.topk(day_scores, k=focus_count, largest=True).indices
    if include_indices is None:
        return focus_index

    include_indices = include_indices.to(device=day_scores.device, dtype=focus_index.dtype).flatten()
    if include_indices.numel() == 0:
        return focus_index
    if focus_count >= group_size:
        return torch.arange(group_size, device=day_scores.device)

    include_indices = torch.unique(include_indices, sorted=False)
    existing_mask = (focus_index[:, None] == include_indices[None, :]).any(dim=1)
    missing_mask = ~(include_indices[:, None] == focus_index[None, :]).any(dim=1)
    missing_indices = include_indices[missing_mask]
    if missing_indices.numel() == 0:
        return focus_index
    if missing_indices.numel() >= focus_count:
        return missing_indices[:focus_count]

    keep_budget = focus_count - missing_indices.numel()
    kept_existing = focus_index[existing_mask][:keep_budget]
    kept_remaining = focus_index[~existing_mask][: max(0, keep_budget - kept_existing.numel())]
    return torch.cat([kept_existing, kept_remaining, missing_indices], dim=0)


def top_target_indices_per_day(
    targets: torch.Tensor,
    group_sizes: list[int],
    data_cfg: DataSettings,
) -> list[torch.Tensor]:
    top_indices_per_day: list[torch.Tensor] = []
    offset = 0
    for group_size in group_sizes:
        next_offset = offset + group_size
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            top_indices_per_day.append(torch.arange(group_size, device=targets.device))
            continue

        oracle_count = portfolio_size(group_size, data_cfg)
        oracle_indices = torch.topk(day_targets, k=oracle_count, largest=True).indices
        top_indices_per_day.append(oracle_indices)
    return top_indices_per_day


def shortlist_target_distribution(
    day_targets: torch.Tensor,
    shortlist_size: int,
    data_cfg: DataSettings,
    model_cfg: ModelSettings,
) -> torch.Tensor:
    group_size = int(day_targets.numel())
    shortlist_count = min(group_size, max(2, int(shortlist_size)))
    shortlist_top_index = torch.topk(day_targets, k=shortlist_count, largest=True).indices
    shortlist_weights = build_rank_weights(
        shortlist_count,
        data_cfg.portfolio_rank_decay,
        device=day_targets.device,
        dtype=day_targets.dtype,
    ).to(device=day_targets.device, dtype=day_targets.dtype)
    shortlist_probs = torch.zeros_like(day_targets)
    shortlist_probs[shortlist_top_index] = shortlist_weights

    centered_targets = day_targets - day_targets.mean()
    scale = centered_targets.std(unbiased=False).clamp_min(1e-6)
    soft_probs = torch.softmax(centered_targets / scale, dim=0)
    target_probs = (
        model_cfg.shortlist_target_blend * shortlist_probs
        + (1.0 - model_cfg.shortlist_target_blend) * soft_probs
    )
    return target_probs / target_probs.sum().clamp_min(1e-12)


def rerank_shortlist_scores(
    model: CrossSectionalRanker,
    broad_scores: torch.Tensor,
    linear_scores: torch.Tensor | None,
    rerank_latent: torch.Tensor,
    group_sizes: list[int],
    include_targets: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    chunks: list[torch.Tensor] = []
    shortlist_indices: list[torch.Tensor] = []
    offset = 0
    rerank_scale = torch.clamp(model.rerank_scale, min=0.0, max=2.0)
    for group_size in group_sizes:
        next_offset = offset + group_size
        day_broad = broad_scores[offset:next_offset]
        day_rerank_latent = rerank_latent[offset:next_offset]
        if group_size < 2:
            chunks.append(day_broad)
            shortlist_indices.append(torch.arange(group_size, device=day_broad.device))
            offset = next_offset
            continue

        include_index = None
        if include_targets is not None:
            include_index = include_targets[len(shortlist_indices)]
        focus_index = select_focus_indices(
            day_broad.detach(),
            group_size,
            model.model_cfg.shortlist_size,
            include_indices=include_index,
        )
        secondary_size = max(0, int(model.model_cfg.secondary_shortlist_size))
        if (
            secondary_size > 0
            and model.model_cfg.secondary_shortlist_source == "linear"
            and linear_scores is not None
        ):
            day_linear = linear_scores[offset:next_offset]
            secondary_index = select_focus_indices(
                day_linear.detach(),
                group_size,
                secondary_size,
                include_indices=include_index,
            )
            focus_index = torch.unique(torch.cat([focus_index, secondary_index], dim=0), sorted=False)
        shortlist_indices.append(focus_index)

        shortlist_broad = day_broad[focus_index]
        shortlist_context = day_rerank_latent[focus_index].unsqueeze(0)
        for block in model.rerank_blocks:
            shortlist_context = block(shortlist_context)
        shortlist_context = model.rerank_norm(shortlist_context.squeeze(0))
        rerank_delta = rerank_scale * torch.tanh(model.rerank_head(shortlist_context).squeeze(-1))

        day_final = day_broad.clone()
        day_final[focus_index] = shortlist_broad + rerank_delta
        chunks.append(day_final)
        offset = next_offset

    if not chunks:
        return broad_scores.new_empty(0), shortlist_indices
    return torch.cat(chunks, dim=0), shortlist_indices


def winner_ce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    shortlist_indices: list[torch.Tensor],
    data_cfg: DataSettings,
    model_cfg: ModelSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for day_index, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        focus_index = shortlist_indices[day_index]
        focused_predictions = day_predictions[focus_index]
        focused_targets = shortlist_target_distribution(
            day_targets,
            shortlist_size=focus_index.numel(),
            data_cfg=data_cfg,
            model_cfg=model_cfg,
        )[focus_index]
        focused_targets = focused_targets / focused_targets.sum().clamp_min(1e-12)
        losses.append(-(focused_targets * F.log_softmax(focused_predictions, dim=0)).sum())

    if not losses:
        return predictions.new_zeros(())
    return torch.stack(losses).mean()


def weighted_winner_ce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_sizes: list[int],
    shortlist_indices: list[torch.Tensor],
    day_weights: torch.Tensor,
    data_cfg: DataSettings,
    model_cfg: ModelSettings,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    offset = 0
    for day_idx, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        day_predictions = predictions[offset:next_offset]
        day_targets = targets[offset:next_offset]
        offset = next_offset
        if group_size < 2:
            continue

        focus_index = shortlist_indices[day_idx]
        focused_predictions = day_predictions[focus_index]
        focused_targets = shortlist_target_distribution(
            day_targets,
            shortlist_size=focus_index.numel(),
            data_cfg=data_cfg,
            model_cfg=model_cfg,
        )[focus_index]
        focused_targets = focused_targets / focused_targets.sum().clamp_min(1e-12)
        losses.append(-(focused_targets * F.log_softmax(focused_predictions, dim=0)).sum())
        weights.append(day_weights[day_idx])
    return weighted_mean_tensors(losses, weights, predictions.device)


def compute_training_loss(
    model: CrossSectionalRanker,
    predictions: dict[str, torch.Tensor],
    targets: torch.Tensor,
    group_sizes: list[int],
    model_cfg: ModelSettings,
    data_cfg: DataSettings,
    training_cfg: TrainingSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    smoothed_targets = smooth_targets(targets, training_cfg.label_smoothing)
    broad_score = predictions["broad_score"]
    day_weights = torch.ones(len(group_sizes), device=targets.device, dtype=torch.float32)
    oracle_shortlist_targets = top_target_indices_per_day(targets, group_sizes, data_cfg)
    final_score, shortlist_indices = rerank_shortlist_scores(
        model,
        broad_score,
        predictions["linear_score"],
        predictions["rerank_latent"],
        group_sizes=group_sizes,
        include_targets=oracle_shortlist_targets,
    )
    listwise = weighted_listwise_rank_loss(
        broad_score,
        smoothed_targets,
        group_sizes,
        day_weights,
        data_cfg,
        training_cfg,
    )
    pairwise = weighted_pairwise_rank_loss(
        broad_score,
        smoothed_targets,
        group_sizes,
        day_weights,
        data_cfg,
        training_cfg,
    )
    huber = weighted_huber_day_loss(broad_score, smoothed_targets, group_sizes, day_weights)
    binary = weighted_binary_top_loss(
        broad_score,
        smoothed_targets,
        group_sizes,
        day_weights,
        data_cfg,
        training_cfg,
    )
    winner = weighted_winner_ce_loss(
        final_score,
        smoothed_targets,
        group_sizes,
        shortlist_indices,
        day_weights,
        data_cfg,
        model_cfg,
    )
    rerank_listwise = weighted_listwise_rank_loss(
        final_score,
        smoothed_targets,
        group_sizes,
        day_weights,
        data_cfg,
        training_cfg,
    )
    total = (
        training_cfg.listwise_loss_weight * listwise
        + training_cfg.pairwise_loss_weight * pairwise
        + training_cfg.huber_loss_weight * huber
        + training_cfg.binary_loss_weight * binary
        + training_cfg.winner_loss_weight * winner
        + training_cfg.rerank_listwise_loss_weight * rerank_listwise
    )
    return total, {
        "listwise": listwise.detach(),
        "pairwise": pairwise.detach(),
        "huber": huber.detach(),
        "binary": binary.detach(),
        "winner": winner.detach(),
        "rerank_listwise": rerank_listwise.detach(),
    }
