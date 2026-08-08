"""Configuration loading.

Every module in `src/` gets its parameters from `configs/config.yaml` through
this loader. Keeping a single typed entry point means a run is fully described
by (git commit, config file) which is what makes the pipeline reproducible.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the directory containing this file.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


class Config(dict):
    """A dict that also supports attribute access and nested path lookup."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted_key: str) -> Path:
        """Resolve a `paths.*` entry to an absolute Path, creating parents."""
        node: Any = self
        for part in dotted_key.split("."):
            node = node[part]
        path = REPO_ROOT / str(node)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Read the YAML config and return it as a `Config`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def ensure_dirs(cfg: Config) -> None:
    """Create every directory referenced by the config (idempotent)."""
    for key, value in cfg["paths"].items():
        target = REPO_ROOT / str(value)
        directory = target if key.endswith("_dir") else target.parent
        directory.mkdir(parents=True, exist_ok=True)


CFG = load_config()


def rel(path) -> str:
    """Render a path relative to the repository root for logging.

    Absolute paths in logs are noise: they are machine-specific, they make output
    non-reproducible across environments, and they leak local directory layout
    into artefacts that get shared. Every log line therefore reports a
    repository-relative path.
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
