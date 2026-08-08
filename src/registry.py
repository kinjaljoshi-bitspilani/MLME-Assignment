"""A minimal file-backed model registry.

MLflow or SageMaker Model Registry would do this in production. The point of
re-implementing the *interface* in ~150 lines is that it makes the contract
explicit and testable:

* a model artefact is immutable once written and addressed by ``version``;
* exactly one version holds the ``production`` stage at a time;
* the serving layer resolves ``production`` at load time and never hard-codes a
  filename, so promotion and rollback are metadata operations that require no
  code change and no redeploy;
* every version carries the metrics it was promoted on, which is what the
  retraining trigger later compares live performance against.

Registry state lives in ``models/registry.json``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib

from .config import CFG, Config

Stage = Literal["staging", "production", "archived"]


class ModelRegistry:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or CFG
        self.registry_path: Path = self.cfg.get_path("paths.registry")
        self.models_dir: Path = self.cfg.get_path("paths.models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {"models": [], "created_at": datetime.now(timezone.utc).isoformat()}

    def _save(self) -> None:
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.registry_path.write_text(json.dumps(self._state, indent=2, default=str))

    # -------------------------------------------------------------- writes
    def next_version(self, name: str) -> str:
        existing = [m for m in self._state["models"] if m["name"] == name]
        return f"v{len(existing) + 1}"

    def register(
        self,
        name: str,
        estimator: Any,
        metrics: dict[str, float],
        params: dict[str, Any],
        feature_columns: list[str],
        stage: Stage = "staging",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist an artefact and its metadata. Returns the registry entry."""
        version = self.next_version(name)
        artefact = self.models_dir / f"{name}_{version}.joblib"
        joblib.dump(estimator, artefact)

        entry = {
            "name": name,
            "version": version,
            "model_id": f"{name}:{version}",
            "stage": stage,
            "artefact_path": str(artefact.relative_to(self.cfg.get_path("paths.models_dir").parent)),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {k: (None if v is None else round(float(v), 6))
                        for k, v in metrics.items()},
            "params": params,
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            **(extra or {}),
        }
        self._state["models"].append(entry)
        self._save()
        print(f"[registry] registered {entry['model_id']} (stage={stage})")
        return entry

    def promote(self, model_id: str) -> dict[str, Any]:
        """Move ``model_id`` to production and archive the incumbent."""
        target = self.get(model_id=model_id)
        if target is None:
            raise KeyError(f"unknown model_id {model_id}")
        for m in self._state["models"]:
            if m["name"] == target["name"] and m["stage"] == "production":
                m["stage"] = "archived"
                m["archived_at"] = datetime.now(timezone.utc).isoformat()
                print(f"[registry] archived previous production {m['model_id']}")
        target["stage"] = "production"
        target["promoted_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        print(f"[registry] PROMOTED {model_id} -> production")
        return target

    def rollback(self, name: str) -> dict[str, Any]:
        """Re-promote the most recently archived version. The incident lever."""
        archived = [m for m in self._state["models"]
                    if m["name"] == name and m["stage"] == "archived"
                    and m.get("archived_at")]
        if not archived:
            raise RuntimeError(f"no archived version of {name} to roll back to")
        latest = max(archived, key=lambda m: m["archived_at"])
        return self.promote(latest["model_id"])

    # --------------------------------------------------------------- reads
    def get(self, model_id: str | None = None, name: str | None = None,
            version: str | None = None) -> dict[str, Any] | None:
        for m in self._state["models"]:
            if model_id and m["model_id"] == model_id:
                return m
            if name and version and m["name"] == name and m["version"] == version:
                return m
        return None

    def get_stage(self, stage: Stage = "production", name: str | None = None):
        cands = [m for m in self._state["models"] if m["stage"] == stage
                 and (name is None or m["name"] == name)]
        if not cands:
            return None
        return max(cands, key=lambda m: m["registered_at"])

    def load(self, model_id: str | None = None, stage: Stage | None = "production"):
        """Return ``(estimator, entry)`` resolving by id or by stage."""
        entry = self.get(model_id=model_id) if model_id else self.get_stage(stage)
        if entry is None:
            raise RuntimeError(f"no model found (model_id={model_id}, stage={stage})")
        root = self.cfg.get_path("paths.models_dir").parent
        estimator = joblib.load(root / entry["artefact_path"])
        return estimator, entry

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._state["models"])

    def summary(self):
        import pandas as pd

        rows = []
        for m in self._state["models"]:
            row = {"model_id": m["model_id"], "stage": m["stage"],
                   "registered_at": m["registered_at"][:19],
                   "n_features": m["n_features"]}
            row.update({f"metric_{k}": v for k, v in m["metrics"].items()})
            rows.append(row)
        return pd.DataFrame(rows)
