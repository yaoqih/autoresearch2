from __future__ import annotations

import logging
from pathlib import Path

from quant_impl.settings import AppConfig, REPO_ROOT


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def resolve_runtime_path(path_value: str | Path | None, *, root: Path = REPO_ROOT) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_log_file_path(
    config: AppConfig,
    command_name: str,
    *,
    log_file: str | Path | None = None,
) -> Path | None:
    resolved = resolve_runtime_path(log_file if log_file is not None else config.logging.log_file)
    if resolved is not None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    if not config.logging.write_file:
        return None
    path = (config.paths.logs_dir / f"{command_name}.log").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(
    config: AppConfig,
    command_name: str,
    *,
    log_level: str | None = None,
    log_file: str | Path | None = None,
) -> Path | None:
    level_name = str(log_level or config.logging.level).upper()
    level = getattr(logging, level_name, logging.INFO)
    resolved_log_file = resolve_log_file_path(config, command_name, log_file=log_file)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    handlers: list[logging.Handler] = []
    if config.logging.console:
        handlers.append(logging.StreamHandler())
    if resolved_log_file is not None:
        handlers.append(logging.FileHandler(resolved_log_file, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())

    formatter = logging.Formatter(LOG_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    logging.captureWarnings(True)
    return resolved_log_file
