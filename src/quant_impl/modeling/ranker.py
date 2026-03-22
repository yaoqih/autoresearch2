from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_impl.settings import ModelSettings, TrainingSettings


def build_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation {name}")


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, *, activation: str, dropout: float, use_layernorm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim, dim)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.gate(residual)) * x
        return residual + x


class FeedForwardBackbone(nn.Module):
    def __init__(self, input_dim: int, model_cfg: ModelSettings):
        super().__init__()
        dims = [input_dim, *model_cfg.hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if model_cfg.use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(build_activation(model_cfg.activation))
            if model_cfg.dropout > 0:
                layers.append(nn.Dropout(model_cfg.dropout))
        self.stem = nn.Sequential(*layers)
        last_dim = model_cfg.hidden_dims[-1]
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    last_dim,
                    activation=model_cfg.activation,
                    dropout=model_cfg.dropout,
                    use_layernorm=model_cfg.use_layernorm,
                )
                for _ in range(model_cfg.num_residual_blocks)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        for block in self.blocks:
            x = block(x)
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
        self.backbone = FeedForwardBackbone(input_dim, model_cfg)
        hidden_dim = model_cfg.hidden_dims[-1]
        self.base_head = nn.Linear(hidden_dim, 1)
        self.linear_head = nn.Linear(input_dim, 1, bias=False)
        self.linear_head.weight.data.copy_(linear_weights.view(1, -1))
        self.linear_head.weight.requires_grad_(not model_cfg.freeze_linear_head)
        self.shortlist_proj = nn.Linear(hidden_dim, model_cfg.rerank_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.rerank_dim,
            nhead=model_cfg.rerank_heads,
            dim_feedforward=model_cfg.rerank_dim * 4,
            dropout=model_cfg.dropout,
            activation=model_cfg.activation,
            batch_first=True,
            norm_first=False,
        )
        self.reranker = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg.rerank_blocks)
        self.rerank_head = nn.Linear(model_cfg.rerank_dim, 1)

    def score_day(self, features: torch.Tensor) -> DayScores:
        encoded = self.backbone(features)
        linear_score = self.linear_head(features).squeeze(-1)
        residual_score = self.base_head(encoded).squeeze(-1)
        base_score = self.model_cfg.linear_head_mix * linear_score
        base_score += (1.0 - self.model_cfg.linear_head_mix) * residual_score

        shortlist_size = max(1, min(features.shape[0], self.model_cfg.shortlist_size))
        shortlist_idx = torch.topk(base_score, k=shortlist_size, largest=True).indices
        if shortlist_size >= 2:
            rerank_input = self.shortlist_proj(encoded[shortlist_idx]).unsqueeze(0)
            rerank_tokens = self.reranker(rerank_input).squeeze(0)
            rerank_score = self.rerank_head(rerank_tokens).squeeze(-1)
            final_score = base_score.clone()
            final_score[shortlist_idx] = (1.0 - self.model_cfg.rerank_mix) * base_score[shortlist_idx]
            final_score[shortlist_idx] += self.model_cfg.rerank_mix * rerank_score
        else:
            final_score = base_score
        return DayScores(scores=final_score, base_scores=base_score, shortlist_idx=shortlist_idx)

    def day_outputs(self, features: torch.Tensor, group_sizes: list[int]) -> list[DayScores]:
        outputs: list[DayScores] = []
        offset = 0
        for group_size in group_sizes:
            next_offset = offset + group_size
            outputs.append(self.score_day(features[offset:next_offset]))
            offset = next_offset
        return outputs

    def score_batch(self, features: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        outputs = self.day_outputs(features, group_sizes)
        return torch.cat([output.scores for output in outputs], dim=0)


def supervision_bucket_size(group_size: int, training_cfg: TrainingSettings) -> tuple[int, int]:
    positive = max(2, int(round(group_size * training_cfg.positive_fraction)))
    positive = min(group_size, positive)
    expanded = min(group_size, max(positive, positive + int(round(training_cfg.top_bucket_expansion_scale))))
    return positive, expanded


def build_target_distribution(
    targets: torch.Tensor,
    training_cfg: TrainingSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_size = targets.numel()
    positive_k, expanded_k = supervision_bucket_size(group_size, training_cfg)
    order = torch.argsort(targets, descending=True)
    target_dist = torch.zeros_like(targets)

    focus_idx = order[:positive_k]
    focus_weights = torch.arange(1, positive_k + 1, device=targets.device, dtype=targets.dtype).pow(-1.0)
    focus_weights = focus_weights / focus_weights.sum().clamp_min(1e-12)
    target_dist[focus_idx] = focus_weights

    expanded_idx = order[:expanded_k]
    soft_target = torch.softmax(targets[expanded_idx] / max(training_cfg.label_smoothing, 1e-3), dim=0)
    expanded_dist = torch.zeros_like(targets)
    expanded_dist[expanded_idx] = soft_target
    blended = training_cfg.listwise_target_blend * target_dist + (1.0 - training_cfg.listwise_target_blend) * expanded_dist
    blended = blended / blended.sum().clamp_min(1e-12)
    binary_labels = torch.zeros_like(targets)
    binary_labels[expanded_idx] = 1.0
    return blended, binary_labels


def pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    training_cfg: TrainingSettings,
) -> torch.Tensor:
    group_size = targets.numel()
    if group_size < 4:
        return scores.new_zeros(())
    focus_k = max(2, min(group_size // 2, int(round(group_size * training_cfg.pair_focus_fraction))))
    order = torch.argsort(targets, descending=True)
    pos_idx = order[:focus_k]
    neg_idx = order[-focus_k:]
    sample_count = min(training_cfg.pair_samples_per_day, pos_idx.numel() * neg_idx.numel())
    if sample_count <= 0:
        return scores.new_zeros(())
    pos_choice = torch.randint(0, pos_idx.numel(), (sample_count,), device=scores.device)
    neg_choice = torch.randint(0, neg_idx.numel(), (sample_count,), device=scores.device)
    margin = scores[pos_idx[pos_choice]] - scores[neg_idx[neg_choice]]
    return F.softplus(-margin).mean()


def compute_training_loss(
    outputs: list[DayScores],
    targets: torch.Tensor,
    group_sizes: list[int],
    training_cfg: TrainingSettings,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {
        "listwise": [],
        "pairwise": [],
        "huber": [],
        "binary": [],
        "winner": [],
        "rerank": [],
    }
    offset = 0
    for output, group_size in zip(outputs, group_sizes):
        next_offset = offset + group_size
        day_targets = targets[offset:next_offset]
        offset = next_offset
        target_dist, binary_labels = build_target_distribution(day_targets, training_cfg)
        log_probs = F.log_softmax(output.scores, dim=0)
        losses["listwise"].append(F.kl_div(log_probs, target_dist, reduction="batchmean"))
        losses["pairwise"].append(pairwise_ranking_loss(output.scores, day_targets, training_cfg))
        losses["huber"].append(F.smooth_l1_loss(output.scores, day_targets))

        positive_count = float(binary_labels.sum().item())
        negative_count = max(1.0, float(binary_labels.numel()) - positive_count)
        pos_weight = output.scores.new_tensor(negative_count / max(positive_count, 1.0))
        losses["binary"].append(
            F.binary_cross_entropy_with_logits(output.scores, binary_labels, pos_weight=pos_weight)
        )

        winner_index = torch.argmax(day_targets).view(1)
        losses["winner"].append(F.cross_entropy(output.scores.unsqueeze(0), winner_index))

        shortlist_targets = day_targets[output.shortlist_idx]
        if shortlist_targets.numel() >= 2:
            shortlist_probs = torch.softmax(
                shortlist_targets / max(training_cfg.label_smoothing, 1e-3),
                dim=0,
            )
            shortlist_scores = output.scores[output.shortlist_idx]
            losses["rerank"].append(
                F.kl_div(F.log_softmax(shortlist_scores, dim=0), shortlist_probs, reduction="batchmean")
            )
        else:
            losses["rerank"].append(output.scores.new_zeros(()))

    def mean_loss(name: str) -> torch.Tensor:
        return torch.stack(losses[name]).mean() if losses[name] else targets.new_zeros(())

    aggregated = {
        "listwise": mean_loss("listwise"),
        "pairwise": mean_loss("pairwise"),
        "huber": mean_loss("huber"),
        "binary": mean_loss("binary"),
        "winner": mean_loss("winner"),
        "rerank": mean_loss("rerank"),
    }
    total = training_cfg.listwise_loss_weight * aggregated["listwise"]
    total += training_cfg.pairwise_loss_weight * aggregated["pairwise"]
    total += training_cfg.huber_loss_weight * aggregated["huber"]
    total += training_cfg.binary_loss_weight * aggregated["binary"]
    total += training_cfg.winner_loss_weight * aggregated["winner"]
    total += training_cfg.rerank_listwise_loss_weight * aggregated["rerank"]
    return total, {name: float(value.detach().cpu().item()) for name, value in aggregated.items()}
