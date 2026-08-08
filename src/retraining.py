"""Retraining trigger logic.

The question a scheduler asks every night is a single boolean: *retrain now?*
Answering it from one signal is fragile, so four independent signals are
evaluated and combined by policy:

    S1  staleness      - model age exceeds `max_model_age_days`
    S2  fresh labels   - at least N new snapshots have a fully observed horizon
    S3  performance    - AUC on recently matured labels dropped by > X
    S4  drift          - share of drifting features exceeds threshold

Policy (deliberately asymmetric, because retraining is not free and a bad
retrain is worse than a stale model):

* **S3 alone is sufficient** - measured degradation is ground truth.
* **S1 requires S2** - never retrain on no new information just because the
  calendar moved.
* **S4 requires S2** - drift without fresh labels means we would fit new inputs
  to old targets, which encodes the drift rather than correcting it.
* Any **schema break blocks retraining entirely** - fix the pipeline first,
  otherwise the bad batch is baked into the model.

The function returns a full reason trace so a human can audit why the system
did or did not act on any given night.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import CFG, Config


@dataclass
class RetrainingSignals:
    """Everything the trigger needs, gathered by the orchestrator."""
    model_age_days: int
    new_labelled_snapshots: int
    recent_auc: float | None = None
    auc_at_promotion: float | None = None
    drift_share: float = 0.0
    drift_status: str = "ok"          # ok | warn | alert
    schema_ok: bool = True
    data_quality_ok: bool = True
    rows_ingested_since_training: int = 0


@dataclass
class TriggerResult:
    should_retrain: bool
    urgency: str                      # none | scheduled | high | blocked
    fired: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    evaluated_at_utc: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_retrain(
    signals: RetrainingSignals, cfg: Config | None = None
) -> TriggerResult:
    """Evaluate the retraining policy. Pure function: easy to unit-test."""
    cfg = cfg or CFG
    r = cfg["retraining"]
    fired: list[str] = []
    blocked: list[str] = []
    reasons: list[str] = []

    # ---- hard blocks first -------------------------------------------
    if not signals.schema_ok:
        blocked.append("schema_break")
        reasons.append(
            "Upstream schema break detected. Retraining is blocked: fitting on a "
            "malformed batch would bake the defect into the model. Fix the "
            "ingestion contract, backfill, then re-evaluate.")
    if not signals.data_quality_ok:
        blocked.append("data_quality_failure")
        reasons.append(
            "Data-quality checks are failing at alert severity. Retraining is "
            "blocked until the batch is triaged.")

    # ---- S3 performance degradation (sufficient on its own) ----------
    perf_fired = False
    if signals.recent_auc is not None and signals.auc_at_promotion is not None:
        drop = signals.auc_at_promotion - signals.recent_auc
        if drop >= r["auc_drop_alert"]:
            perf_fired = True
            fired.append("performance_degradation")
            reasons.append(
                f"AUC on recently matured labels is {signals.recent_auc:.4f} vs "
                f"{signals.auc_at_promotion:.4f} at promotion (drop "
                f"{drop:.4f} >= {r['auc_drop_alert']}). Measured degradation is "
                f"ground truth, so this alone justifies a retrain.")
        else:
            reasons.append(
                f"Performance holding: AUC {signals.recent_auc:.4f} vs "
                f"{signals.auc_at_promotion:.4f} at promotion "
                f"(drop {drop:.4f} < {r['auc_drop_alert']}).")
    else:
        reasons.append(
            "No matured labels yet for the live model, so performance cannot be "
            "measured. Falling back to leading indicators (drift, staleness).")

    # ---- S2 fresh labels: the enabling condition ---------------------
    has_fresh = signals.new_labelled_snapshots >= r["min_new_labelled_snapshots"]
    if has_fresh:
        reasons.append(
            f"{signals.new_labelled_snapshots} new snapshot(s) have a fully "
            f"observed label horizon (need >= "
            f"{r['min_new_labelled_snapshots']}).")
    else:
        reasons.append(
            f"Only {signals.new_labelled_snapshots} new labelled snapshot(s); "
            f"nothing new to learn from.")

    # ---- S1 staleness -------------------------------------------------
    if signals.model_age_days >= r["max_model_age_days"]:
        if has_fresh:
            fired.append("model_staleness")
            reasons.append(
                f"Model is {signals.model_age_days} days old (limit "
                f"{r['max_model_age_days']}) and fresh labels exist -> "
                f"scheduled refresh.")
        else:
            reasons.append(
                f"Model is {signals.model_age_days} days old but there are no "
                f"fresh labels; a calendar-only retrain would reproduce the same "
                f"model at full cost.")

    # ---- S4 drift ------------------------------------------------------
    if signals.drift_status == "alert" or signals.drift_share >= r["drift_share_trigger"]:
        if has_fresh:
            fired.append("feature_drift")
            reasons.append(
                f"Drift share {signals.drift_share:.1%} >= "
                f"{r['drift_share_trigger']:.0%} (status={signals.drift_status}) "
                f"with fresh labels available -> retrain on the new regime.")
        else:
            reasons.append(
                f"Drift share {signals.drift_share:.1%} is elevated but no fresh "
                f"labels are available. Retraining now would fit new inputs to "
                f"stale targets. Action: investigate the cause and, if the shift "
                f"is genuine, wait for the label horizon to mature.")

    # ---- combine -------------------------------------------------------
    if blocked:
        return TriggerResult(False, "blocked", fired, blocked, reasons,
                             datetime.now(timezone.utc).isoformat(),
                             asdict(signals))
    should = bool(fired)
    if perf_fired:
        urgency = "high"
    elif should:
        urgency = "scheduled"
    else:
        urgency = "none"
    if not should:
        reasons.append("No trigger fired. Continue serving the current model.")
    return TriggerResult(should, urgency, fired, blocked, reasons,
                         datetime.now(timezone.utc).isoformat(), asdict(signals))


# --------------------------------------------------------------------------
def collect_signals(
    cfg: Config | None = None,
    events: pd.DataFrame | None = None,
    recent_auc: float | None = None,
    drift_report: dict[str, Any] | None = None,
) -> RetrainingSignals:
    """Assemble the signals from the artefacts the pipeline already writes.

    This is the glue a scheduler (Airflow / cron) would call; it reads the
    registry, the ingestion log and the latest monitoring report rather than
    recomputing anything.
    """
    from .ingest import ingestion_history
    from .registry import ModelRegistry

    cfg = cfg or CFG
    now = pd.Timestamp(cfg["project"]["system_asof"])

    reg = ModelRegistry(cfg)
    prod = reg.get_stage("production")
    if prod is None:
        return RetrainingSignals(model_age_days=10**6, new_labelled_snapshots=1,
                                 rows_ingested_since_training=0)

    promoted_at = pd.Timestamp(prod.get("promoted_at") or prod["registered_at"]).tz_localize(None)
    # Model age is measured on the *simulated* clock so the demo is deterministic.
    trained_through = max(pd.Timestamp(d) for d in prod["train_snapshots"])
    model_age_days = int((now - trained_through).days)

    horizon = cfg["label"]["horizon_days"]
    known = {pd.Timestamp(d) for d in prod["train_snapshots"]}
    candidates = [pd.Timestamp(d) for d in cfg["label"]["snapshot_dates"]]
    new_labelled = sum(
        1 for d in candidates
        if d not in known and d + pd.Timedelta(days=horizon) <= now)

    hist = ingestion_history(cfg)
    rows_since = 0
    if not hist.empty and "rows_appended" in hist.columns:
        ts = pd.to_datetime(hist["ingested_at_utc"], errors="coerce", utc=True)
        rows_since = int(hist.loc[ts >= promoted_at.tz_localize("UTC"),
                                  "rows_appended"].fillna(0).sum())

    drift_share, drift_status, schema_ok, dq_ok = 0.0, "ok", True, True
    if drift_report:
        for c in drift_report.get("checks", []):
            if c["check"] == "feature_drift":
                drift_share = float(c.get("drift_share", 0.0) or 0.0)
                drift_status = c.get("severity", "ok")
            if c["check"] == "schema":
                schema_ok = bool(c.get("passed", True))
            if c["check"] == "data_quality":
                dq_ok = c.get("severity") != "alert"

    return RetrainingSignals(
        model_age_days=model_age_days,
        new_labelled_snapshots=new_labelled,
        recent_auc=recent_auc,
        auc_at_promotion=prod["metrics"].get("valid_roc_auc"),
        drift_share=drift_share, drift_status=drift_status,
        schema_ok=schema_ok, data_quality_ok=dq_ok,
        rows_ingested_since_training=rows_since,
    )


def format_decision(result: TriggerResult) -> str:
    lines = [
        f"Retrain: {'YES' if result.should_retrain else 'NO'}  "
        f"(urgency={result.urgency})",
        f"Triggers fired : {result.fired or 'none'}",
        f"Blocked by     : {result.blocked_by or 'none'}",
        "Reasoning:",
    ]
    lines += [f"  - {r}" for r in result.reasons]
    return "\n".join(lines)
