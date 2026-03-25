from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathSettings:
    raw_daily_dir: Path = Path("data/daily")
    merged_parquet: Path = Path("data/market_daily.parquet")
    cache_dir: Path = Path("artifacts/cache")
    models_dir: Path = Path("artifacts/models")
    predictions_dir: Path = Path("artifacts/predictions")
    validation_dir: Path = Path("artifacts/validation")
    logs_dir: Path = Path("artifacts/logs")
    metrics_dir: Path = Path("artifacts/metrics")
    reference_merged_parquet: Path | None = None

    def resolve(self, root: Path) -> "PathSettings":
        for field_name in (
            "raw_daily_dir",
            "merged_parquet",
            "cache_dir",
            "models_dir",
            "predictions_dir",
            "validation_dir",
            "logs_dir",
            "metrics_dir",
        ):
            value = getattr(self, field_name)
            setattr(self, field_name, (root / value).resolve() if not value.is_absolute() else value)
        if self.reference_merged_parquet is not None and not self.reference_merged_parquet.is_absolute():
            self.reference_merged_parquet = (root / self.reference_merged_parquet).resolve()
        return self

    def ensure(self) -> None:
        for path in (
            self.raw_daily_dir,
            self.cache_dir,
            self.models_dir,
            self.predictions_dir,
            self.validation_dir,
            self.logs_dir,
            self.metrics_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.merged_parquet.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingSettings:
    level: str = "INFO"
    console: bool = True
    write_file: bool = True
    log_file: str | None = None


@dataclass
class DownloadSettings:
    start_date: str = "1990-01-01"
    end_date: str | None = None
    adjust: str = "hfq"
    use_env_proxy: bool = False
    eastmoney_cookie_warmup: bool = False
    eastmoney_cookie_cache_file: str | None = "artifacts/cache/eastmoney_cookie.json"
    eastmoney_cookie_max_age_seconds: int = 21600
    eastmoney_cookie_node_binary: str = "node"
    eastmoney_cookie_script: str = "tools/eastmoney_cookie_warmer.mjs"
    eastmoney_browser_path: str | None = None
    eastmoney_browser_proxy: str | None = None
    eastmoney_cookie_timeout_ms: int = 15000
    juliang_enabled: bool = False
    juliang_trade_no: str | None = None
    juliang_api_key: str | None = None
    juliang_proxy_username: str | None = None
    juliang_proxy_password: str | None = None
    juliang_api_base: str = "http://v2.api.juliangip.com"
    juliang_proxy_type: int = 1
    juliang_lease_refresh_margin_seconds: int = 5
    juliang_default_lease_seconds: int = 30
    max_workers: int = 0
    host_max_workers: int = 8
    max_retries: int = 2
    timeout: float = 20.0
    request_interval: float = 0.0
    request_jitter: float = 0.0
    retry_sleep: float = 1.5
    refresh_lookback_days: int = 5
    shuffle_symbols: bool = True
    limit: int | None = None
    symbols_file: str | None = None
    extra_symbols: list[str] = field(default_factory=list)
    report_file: str | None = None


@dataclass
class DataSettings:
    cache_version: str = "champion_20260323_lc96_rank_center_full5y"
    min_listed_days: int = 120
    min_daily_universe: int = 100
    target_clip: tuple[float, float] = (-0.25, 0.25)
    train_days: int = 252 * 5
    valid_days: int = 252
    holdout_days: int = 252
    step_days: int = 252
    rolling_train: bool = True
    feature_clip: float = 20.0
    normalized_clip: float = 5.0
    ret_windows: tuple[int, ...] = (2, 3, 5, 10, 20, 60)
    ma_windows: tuple[int, ...] = (5, 10, 20, 60)
    vol_windows: tuple[int, ...] = (5, 10, 20, 60)
    flow_windows: tuple[int, ...] = (5, 20, 60)
    portfolio_top_fraction: float = 0.005
    portfolio_min_k: int = 4
    portfolio_max_k: int = 12
    portfolio_rank_decay: float = 1.0
    eval_top_k: int = 1
    robust_drawdown_weight: float = 0.01
    selection_alpha_weight: float = 0.25
    selection_hit_weight: float = 0.05
    entry_offset_days: int = 1
    exit_offset_days: int = 2

    @property
    def holding_days(self) -> int:
        return max(1, self.exit_offset_days - self.entry_offset_days)


@dataclass
class ModelSettings:
    hidden_dims: tuple[int, ...] = (384, 192, 96)
    dropout: float = 0.2
    activation: str = "gelu"
    use_layernorm: bool = False
    num_residual_blocks: int = 2
    linear_head_mix: float = 0.6
    freeze_linear_head: bool = True
    shortlist_size: int = 96
    secondary_shortlist_size: int = 0
    secondary_shortlist_source: str = "off"
    rerank_dim: int = 96
    rerank_blocks: int = 2
    rerank_heads: int = 8
    rerank_mix: float = 0.25
    shortlist_target_blend: float = 0.75


@dataclass
class TrainingSettings:
    member_config: str = "lc96"
    temporal_mode: str = "full5y"
    target_transform: str = "rank_center"
    train_target_abs_cap: float = 0.10
    train_target_cap_applies_to_linear_head: bool = True
    seed: int = 42
    epochs: int = 18
    deployment_epochs: int = 5
    batch_days: int = 20
    learning_rate: float = 0.0005
    weight_decay: float = 0.0005
    label_smoothing: float = 0.02
    grad_clip: float = 1.0
    early_stopping_patience: int = 4
    positive_fraction: float = 0.08
    pair_samples_per_day: int = 96
    pair_focus_fraction: float = 0.33
    listwise_target_blend: float = 0.9
    top_bucket_expansion_scale: float = 10.0
    listwise_loss_weight: float = 0.51
    pairwise_loss_weight: float = 0.19
    huber_loss_weight: float = 0.14
    binary_loss_weight: float = 0.05
    winner_loss_weight: float = 0.11
    rerank_listwise_loss_weight: float = 0.06
    recent_holdout_folds: int = 5
    recent_holdout_weight: float = 1.75
    research_holdout_weight: float = 0.35
    research_recent_weight: float = 0.45
    research_valid_weight: float = 0.2


@dataclass
class InferenceSettings:
    top_k: int = 1
    archive_top_n: int = 10
    prediction_name: str = "champion_20260323_lc96_rank_center_full5y"


@dataclass
class AppConfig:
    paths: PathSettings = field(default_factory=PathSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    download: DownloadSettings = field(default_factory=DownloadSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)

    def resolve(self, root: Path = REPO_ROOT) -> "AppConfig":
        self.paths.resolve(root)
        self.paths.ensure()
        return self

    @property
    def bundle_path(self) -> Path:
        return self.paths.cache_dir / f"{self.data.cache_version}_bundle.pt"

    @property
    def summary_path(self) -> Path:
        return self.paths.cache_dir / f"{self.data.cache_version}_summary.json"

    @property
    def model_path(self) -> Path:
        return self.paths.models_dir / f"{self.data.cache_version}.pt"

    @property
    def training_metrics_path(self) -> Path:
        return self.paths.metrics_dir / f"{self.data.cache_version}_training.json"

    @property
    def validation_history_path(self) -> Path:
        return self.paths.validation_dir / "history.csv"

    @property
    def prediction_daily_dir(self) -> Path:
        return self.paths.predictions_dir / "daily"

    def prediction_daily_path(self, as_of_date: str) -> Path:
        return self.prediction_daily_dir / f"{as_of_date}.json"

    @property
    def prediction_index_path(self) -> Path:
        return self.paths.predictions_dir / "index.json"

    @property
    def prediction_latest_path(self) -> Path:
        return self.paths.predictions_dir / "latest.json"


def _merge_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if isinstance(current, Path):
            setattr(instance, key, Path(value))
        elif key == "reference_merged_parquet" and value is not None:
            setattr(instance, key, Path(value))
        elif isinstance(current, tuple):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else REPO_ROOT / "configs" / "default.yaml"
    config = AppConfig()
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for section in ("paths", "logging", "download", "data", "model", "training", "inference"):
            if section in payload:
                _merge_dataclass(getattr(config, section), payload[section])
    return config.resolve()
