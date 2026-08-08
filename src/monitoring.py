"""Data quality and drift monitoring.

The pipeline writes a frozen *reference profile* at training time; every later
batch is compared against it. Three families of check, in the order they should
run:

1. **Schema / contract** - do the expected columns exist with usable dtypes?
   A schema break invalidates everything downstream, so it short-circuits.
2. **Data quality** - null rates, out-of-range values, all-constant columns,
   duplicate entities. These catch broken *upstream data* rather than a
   changed world.
3. **Distribution drift** - PSI plus a two-sample Kolmogorov-Smirnov test per
   numeric feature, and PSI over category shares for categoricals. These catch
   a changed world rather than broken data.

Why PSI *and* KS. PSI is bucketed, bounded and has industry-standard action
thresholds (0.10 / 0.25) which makes it good for alerting, but it is
insensitive to small shifts. KS is sensitive and gives a p-value, but on large
batches it flags shifts far too small to matter. Requiring both to agree, and
alerting on the *share* of drifting features rather than on any single one,
keeps the alert rate survivable - a monitor that pages every day gets muted,
and a muted monitor is worse than no monitor.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import CFG, Config, rel
from .features import (
    CATEGORICAL_FEATURES,
    ENTITY_COL,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    NUMERIC_FEATURES,
)

REFERENCE_FILENAME = "training_reference_profile.json"


# ==========================================================================
# Reference profile
# ==========================================================================
def build_reference_profile(
    train_df: pd.DataFrame, n_bins: int = 10, model_id: str | None = None
) -> dict[str, Any]:
    """Freeze the training distribution: quantile edges, moments, category shares.

    Bin edges are stored explicitly. Re-deriving edges from each new batch would
    make PSI compare a distribution against itself and always return ~0 - a
    silent, very common monitoring bug.
    """
    profile: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "n_rows": int(len(train_df)),
        "n_bins": n_bins,
        "numeric": {},
        "categorical": {},
    }

    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(train_df[col], errors="coerce").dropna()
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(s, qs))
        if len(edges) < 3:  # near-constant column
            edges = np.array([s.min() - 1e-6, s.max() + 1e-6])
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(s, bins=edges)
        profile["numeric"][col] = {
            "edges": edges.tolist(),
            "expected_pct": (counts / max(counts.sum(), 1)).tolist(),
            "mean": float(s.mean()), "std": float(s.std(ddof=0)),
            "p01": float(s.quantile(0.01)), "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)), "p75": float(s.quantile(0.75)),
            "p99": float(s.quantile(0.99)),
            "min": float(s.min()), "max": float(s.max()),
            "null_rate": float(train_df[col].isna().mean()),
            "sample": s.sample(min(len(s), 5000), random_state=0).tolist(),
        }

    for col in CATEGORICAL_FEATURES:
        shares = train_df[col].astype(str).value_counts(normalize=True)
        profile["categorical"][col] = {
            "levels": shares.index.tolist(),
            "expected_pct": shares.to_numpy().tolist(),
            "null_rate": float(train_df[col].isna().mean()),
        }

    if "churn_90d" in train_df.columns:
        profile["target"] = {"base_rate": float(train_df["churn_90d"].mean())}
    return profile


def write_reference_stats(
    train_df: pd.DataFrame, cfg: Config | None = None, model_id: str | None = None
) -> Path:
    cfg = cfg or CFG
    profile = build_reference_profile(train_df, model_id=model_id)
    out_dir = cfg.get_path("paths.reference_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REFERENCE_FILENAME
    path.write_text(json.dumps(profile, indent=2))
    print(f"[monitor] reference profile -> {rel(path)}")
    return path


def load_reference_profile(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or CFG
    path = cfg.get_path("paths.reference_dir") / REFERENCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No reference profile at {path}. Run `python -m src.train` first.")
    return json.loads(path.read_text())


# ==========================================================================
# Drift statistics
# ==========================================================================
def population_stability_index(
    expected_pct: np.ndarray, actual_pct: np.ndarray, eps: float = 1e-6
) -> float:
    """PSI = sum (actual - expected) * ln(actual / expected).

    ``eps`` floors empty buckets; without it a bucket that is empty in the new
    batch sends the term to infinity and one sparse tail bucket dominates.
    """
    e = np.clip(np.asarray(expected_pct, dtype=float), eps, None)
    a = np.clip(np.asarray(actual_pct, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(ref: dict[str, Any], values: pd.Series) -> float:
    edges = np.asarray(ref["edges"], dtype=float)
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    counts, _ = np.histogram(s, bins=edges)
    return population_stability_index(
        np.asarray(ref["expected_pct"]), counts / max(counts.sum(), 1))


def psi_categorical(ref: dict[str, Any], values: pd.Series) -> tuple[float, list[str]]:
    levels = list(ref["levels"])
    actual = values.astype(str).value_counts(normalize=True)
    unseen = [lv for lv in actual.index if lv not in levels]
    exp = np.asarray(ref["expected_pct"], dtype=float)
    act = np.asarray([actual.get(lv, 0.0) for lv in levels], dtype=float)
    if unseen:  # fold every unseen level into one extra bucket
        exp = np.append(exp, 1e-6)
        act = np.append(act, float(sum(actual[lv] for lv in unseen)))
    return population_stability_index(exp, act), unseen


def _severity(psi: float, warn: float, alert: float) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi >= alert:
        return "alert"
    if psi >= warn:
        return "warn"
    return "ok"


# ==========================================================================
# Checks
# ==========================================================================
def check_schema(batch: pd.DataFrame) -> dict[str, Any]:
    """Contract check. Runs first because nothing else is valid without it."""
    missing = [c for c in FEATURE_COLUMNS if c not in batch.columns]
    extra = [c for c in batch.columns
             if c not in FEATURE_COLUMNS + [ENTITY_COL, "snapshot_date",
                                            "churn_90d", "feature_timestamp"]]
    wrong_dtype = [c for c in NUMERIC_FEATURES if c in batch.columns
                   and not pd.api.types.is_numeric_dtype(batch[c])]
    passed = not missing and not wrong_dtype
    return {
        "check": "schema", "passed": passed,
        "severity": "ok" if passed else "alert",
        "missing_columns": missing, "unexpected_columns": extra,
        "non_numeric_numeric_columns": wrong_dtype,
        "message": "Schema matches the training contract." if passed
        else f"SCHEMA BREAK missing={missing} wrong_dtype={wrong_dtype}",
    }


def check_data_quality(
    batch: pd.DataFrame, cfg: Config | None = None
) -> dict[str, Any]:
    """Null rates, range violations, constant columns, duplicate entities."""
    cfg = cfg or CFG
    m = cfg["monitoring"]
    findings: list[dict[str, Any]] = []

    for col in FEATURE_COLUMNS:
        if col not in batch.columns:
            continue
        null_rate = float(batch[col].isna().mean())
        if null_rate > m["null_rate_alert"]:
            findings.append({
                "feature": col, "issue": "high_null_rate",
                "value": round(null_rate, 4),
                "threshold": m["null_rate_alert"], "severity": "alert"})

    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in batch.columns:
            continue
        s = pd.to_numeric(batch[col], errors="coerce")
        n_bad = int(((s < lo) | (s > hi)).sum())
        if n_bad:
            findings.append({
                "feature": col, "issue": "out_of_range",
                "value": n_bad, "threshold": f"[{lo}, {hi}]",
                "severity": "alert" if n_bad / max(len(batch), 1) > 0.01 else "warn"})

    for col in NUMERIC_FEATURES:
        if col in batch.columns and batch[col].nunique(dropna=True) <= 1:
            findings.append({
                "feature": col, "issue": "constant_column",
                "value": int(batch[col].nunique(dropna=True)),
                "threshold": "> 1", "severity": "alert"})

    if ENTITY_COL in batch.columns:
        dupes = int(batch[ENTITY_COL].duplicated().sum())
        if dupes:
            findings.append({
                "feature": ENTITY_COL, "issue": "duplicate_entities",
                "value": dupes, "threshold": 0, "severity": "warn"})

    severities = [f["severity"] for f in findings]
    return {
        "check": "data_quality",
        "passed": "alert" not in severities,
        "severity": "alert" if "alert" in severities
        else ("warn" if "warn" in severities else "ok"),
        "n_rows": int(len(batch)),
        "n_findings": len(findings),
        "findings": findings,
        "message": "No data-quality violations." if not findings
        else f"{len(findings)} data-quality finding(s): "
             + ", ".join(f"{f['feature']}:{f['issue']}" for f in findings[:6]),
    }


def check_feature_drift(
    batch: pd.DataFrame,
    reference: dict[str, Any] | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Per-feature PSI + KS against the frozen training reference."""
    cfg = cfg or CFG
    m = cfg["monitoring"]
    reference = reference or load_reference_profile(cfg)
    rows: list[dict[str, Any]] = []

    for col, ref in reference["numeric"].items():
        if col not in batch.columns:
            continue
        s = pd.to_numeric(batch[col], errors="coerce").dropna()
        psi = psi_numeric(ref, s)
        ks_stat, ks_p = (float("nan"), float("nan"))
        if len(s) >= 20 and ref.get("sample"):
            ks = stats.ks_2samp(np.asarray(ref["sample"], dtype=float), s.to_numpy())
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        ref_mean, ref_std = ref["mean"], max(ref["std"], 1e-9)
        rows.append({
            "feature": col, "kind": "numeric",
            "psi": round(psi, 5),
            "severity": _severity(psi, m["psi_warn"], m["psi_alert"]),
            "ks_stat": round(ks_stat, 5), "ks_pvalue": ks_p,
            "ks_flag": bool(ks_p < m["ks_pvalue_alert"]) if ks_p == ks_p else False,
            "ref_mean": round(ref_mean, 4),
            "batch_mean": round(float(s.mean()), 4) if len(s) else None,
            # Standardised mean shift: interpretable effect size, unlike PSI.
            "mean_shift_in_ref_sd": round(float((s.mean() - ref_mean) / ref_std), 4)
            if len(s) else None,
            "ref_p50": round(ref["p50"], 4),
            "batch_p50": round(float(s.median()), 4) if len(s) else None,
        })

    for col, ref in reference["categorical"].items():
        if col not in batch.columns:
            continue
        psi, unseen = psi_categorical(ref, batch[col])
        rows.append({
            "feature": col, "kind": "categorical",
            "psi": round(psi, 5),
            "severity": _severity(psi, m["psi_warn"], m["psi_alert"]),
            "ks_stat": None, "ks_pvalue": None, "ks_flag": False,
            "unseen_levels": unseen,
            "ref_mean": None, "batch_mean": None,
            "mean_shift_in_ref_sd": None, "ref_p50": None, "batch_p50": None,
        })

    drift = pd.DataFrame(rows).sort_values("psi", ascending=False)
    n = len(drift)
    n_alert = int((drift["severity"] == "alert").sum())
    n_warn = int((drift["severity"] == "warn").sum())
    drift_share = (n_alert + n_warn) / n if n else 0.0
    # Both statistics must agree before a single feature is called "drifted".
    n_confirmed = int(((drift["severity"] != "ok") & (drift["ks_flag"])).sum())

    if drift_share >= m["drift_share_alert"] or n_alert >= 3:
        overall = "alert"
    elif n_alert or n_warn:
        overall = "warn"
    else:
        overall = "ok"

    return {
        "check": "feature_drift", "passed": overall == "ok", "severity": overall,
        "n_rows": int(len(batch)), "n_features": n,
        "n_alert": n_alert, "n_warn": n_warn,
        "n_confirmed_by_psi_and_ks": n_confirmed,
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": m["drift_share_alert"],
        "top_drifting": drift.head(8).to_dict("records"),
        "table": drift,
        "reference_model_id": reference.get("model_id"),
        "message": (f"{n_alert} feature(s) at alert PSI and {n_warn} at warn; "
                    f"drift share {drift_share:.1%} vs threshold "
                    f"{m['drift_share_alert']:.0%}."),
    }


def check_prediction_drift(
    scores: np.ndarray, reference_scores: np.ndarray, cfg: Config | None = None
) -> dict[str, Any]:
    """Drift in the score distribution itself.

    Cheap, needs no labels, and available immediately - unlike accuracy, which
    has to wait a full 90-day label horizon. In practice this is the earliest
    warning the system gets.
    """
    cfg = cfg or CFG
    m = cfg["monitoring"]
    edges = np.unique(np.quantile(reference_scores, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        edges = np.array([0.0, 0.5, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(reference_scores, bins=edges)
    a_counts, _ = np.histogram(scores, bins=edges)
    psi = population_stability_index(
        e_counts / max(e_counts.sum(), 1), a_counts / max(a_counts.sum(), 1))
    sev = _severity(psi, m["psi_warn"], m["psi_alert"])
    return {
        "check": "prediction_drift", "passed": sev == "ok", "severity": sev,
        "psi": round(psi, 5),
        "ref_mean_score": round(float(np.mean(reference_scores)), 4),
        "batch_mean_score": round(float(np.mean(scores)), 4),
        "ref_positive_rate_at_0.5": round(float(np.mean(reference_scores >= 0.5)), 4),
        "batch_positive_rate_at_0.5": round(float(np.mean(scores >= 0.5)), 4),
        "message": f"Score PSI={psi:.4f} ({sev}); mean predicted risk moved "
                   f"{np.mean(reference_scores):.3f} -> {np.mean(scores):.3f}.",
    }


# ==========================================================================
# Orchestration
# ==========================================================================
def run_monitoring_suite(
    batch: pd.DataFrame,
    batch_name: str,
    scores: np.ndarray | None = None,
    reference_scores: np.ndarray | None = None,
    cfg: Config | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run every check, aggregate to one status, persist the report."""
    cfg = cfg or CFG
    checks: list[dict[str, Any]] = []

    schema = check_schema(batch)
    checks.append(schema)

    if schema["passed"]:
        checks.append(check_data_quality(batch, cfg))
        if len(batch) >= cfg["monitoring"]["min_rows_for_check"]:
            checks.append(check_feature_drift(batch, cfg=cfg))
        else:
            checks.append({
                "check": "feature_drift", "passed": True, "severity": "skipped",
                "message": f"Batch has {len(batch)} rows, below "
                           f"min_rows_for_check="
                           f"{cfg['monitoring']['min_rows_for_check']}; "
                           f"PSI on a small sample is too noisy to action."})
        if scores is not None and reference_scores is not None:
            checks.append(check_prediction_drift(scores, reference_scores, cfg))
    else:
        checks.append({"check": "data_quality", "passed": False,
                       "severity": "skipped",
                       "message": "Skipped: schema check failed."})

    order = {"alert": 3, "warn": 2, "ok": 1, "skipped": 0, "unknown": 0}
    overall = max((c["severity"] for c in checks), key=lambda s: order.get(s, 0))

    report = {
        "batch_name": batch_name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(batch)),
        "overall_status": overall,
        "action": {
            "alert": "PAGE on-call DS. Block retraining on this batch until triaged.",
            "warn": "Create a ticket; review at the next weekly model review.",
            "ok": "No action.",
        }.get(overall, "No action."),
        "checks": [{k: v for k, v in c.items() if k != "table"} for c in checks],
    }
    if write:
        out = cfg.get_path("paths.monitoring_dir")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"monitoring_{batch_name}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"[monitor] {batch_name}: {overall.upper()} -> {rel(path)}")
    report["_raw_checks"] = checks  # keeps the drift DataFrame for the notebook
    return report
