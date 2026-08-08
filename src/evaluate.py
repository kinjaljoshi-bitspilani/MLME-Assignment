"""Offline evaluation harness.

Three responsibilities, deliberately separated from ``train.py`` so that a
model can be re-scored without being re-fitted:

1. ``evaluate_model`` - the metric panel. Ranking quality (ROC AUC, PR AUC),
   thresholded quality (precision/recall/F1), probability quality (Brier,
   log loss) and the operating-point metrics the retention team actually cares
   about (recall and precision in the contactable top decile/quintile).
2. ``expected_campaign_value`` - translates probabilities into pounds, because
   "AUC went up 0.01" is not a decision and "we save GBP 4.2k per campaign" is.
3. ``promotion_decision`` - the guardrail. A candidate is promoted only if it
   clears absolute quality floors *and* does not regress against the incumbent
   by more than a tolerance. Encoding this as data rather than as a human
   judgement is what stops a quietly worse model reaching production.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CFG, Config, rel


# --------------------------------------------------------------------------
def _top_k_mask(scores: np.ndarray, k_pct: float) -> np.ndarray:
    """Boolean mask selecting the highest-scoring ``k_pct`` share of rows."""
    n = len(scores)
    k = max(int(np.ceil(n * k_pct)), 1)
    cutoff_idx = np.argsort(-scores)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[cutoff_idx] = True
    return mask


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k_pct: float) -> float:
    """Precision in the top-k slice divided by the base rate."""
    base = float(np.mean(y_true))
    if base == 0:
        return float("nan")
    mask = _top_k_mask(scores, k_pct)
    return float(np.mean(y_true[mask]) / base)


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "model",
) -> dict[str, Any]:
    """The full metric panel for one model on one fold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "model": label,
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        # --- ranking quality: threshold-free, the promotion metric ---------
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        # --- decision quality at the chosen operating point ---------------
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # --- probability quality: needed because we rank by expected value -
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
    }

    # --- campaign operating points: we can only contact a fixed share -----
    for k_pct, name in [(0.10, "decile"), (0.20, "quintile")]:
        mask = _top_k_mask(y_prob, k_pct)
        captured = int(y_true[mask].sum())
        total_pos = int(y_true.sum())
        metrics[f"precision_at_top_{name}"] = float(np.mean(y_true[mask]))
        metrics[f"recall_at_top_{name}"] = float(captured / total_pos) if total_pos else 0.0
        metrics[f"lift_at_top_{name}"] = lift_at_k(y_true, y_prob, k_pct)
    return metrics


def expected_campaign_value(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    cfg: Config | None = None,
    capacity_pct: float | None = None,
) -> dict[str, float]:
    """Monetise a ranking under a fixed contact budget.

    Value = (churners correctly contacted x uplift x margin) - (contacts x cost)

    Contacting a non-churner is pure cost, which is why precision in the
    contactable slice - not global accuracy - is the metric that maps to money.
    """
    cfg = cfg or CFG
    biz = cfg["business"]
    capacity = capacity_pct if capacity_pct is not None else biz["campaign_capacity_pct"]

    y_true = np.asarray(y_true).astype(int)
    mask = _top_k_mask(np.asarray(y_prob), capacity)
    n_contacted = int(mask.sum())
    true_churners_contacted = int(y_true[mask].sum())

    saved = true_churners_contacted * biz["campaign_uplift"]
    gross = saved * biz["gross_margin_per_retained_customer"]
    cost = n_contacted * biz["retention_offer_cost"]

    # A random-targeting baseline, to show the model is doing the work.
    random_churners = n_contacted * float(np.mean(y_true))
    random_net = (random_churners * biz["campaign_uplift"]
                  * biz["gross_margin_per_retained_customer"]) - cost

    return {
        "capacity_pct": float(capacity),
        "n_contacted": n_contacted,
        "true_churners_contacted": true_churners_contacted,
        "expected_customers_saved": round(float(saved), 2),
        "gross_margin_gbp": round(float(gross), 2),
        "campaign_cost_gbp": round(float(cost), 2),
        "net_value_gbp": round(float(gross - cost), 2),
        "net_value_random_targeting_gbp": round(float(random_net), 2),
        "uplift_vs_random_gbp": round(float((gross - cost) - random_net), 2),
        "roi": round(float((gross - cost) / cost), 4) if cost else float("nan"),
    }


# --------------------------------------------------------------------------
def promotion_decision(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Apply the configured guardrails and return an auditable verdict.

    Every gate is reported with its threshold, observed value and pass/fail, so
    the decision can be reviewed months later without re-running anything.
    """
    cfg = cfg or CFG
    g = cfg["promotion"]
    gates: list[dict[str, Any]] = []

    def gate(name: str, observed: float | None, threshold: float,
             op: str, rationale: str) -> None:
        if observed is None:
            passed = False
        elif op == ">=":
            passed = observed >= threshold
        elif op == "<=":
            passed = observed <= threshold
        else:  # pragma: no cover
            raise ValueError(op)
        gates.append({
            "gate": name, "observed": None if observed is None else round(float(observed), 6),
            "operator": op, "threshold": threshold, "passed": bool(passed),
            "rationale": rationale,
        })

    gate("roc_auc_floor", candidate.get("roc_auc"), g["min_roc_auc"], ">=",
         "Absolute ranking-quality floor; below this the campaign list is not "
         "reliably better than intuition.")
    gate("pr_auc_floor", candidate.get("pr_auc"), g["min_pr_auc"], ">=",
         "Guards precision on the positive class, which drives campaign waste.")
    gate("calibration_brier_ceiling", candidate.get("brier"), g["max_brier"], "<=",
         "Probabilities are multiplied by margin to rank; poor calibration "
         "corrupts the expected-value ordering.")
    gate("recall_at_top_decile_floor", candidate.get("recall_at_top_decile"),
         g["min_recall_at_top_decile"], ">=",
         "The contactable slice must capture a meaningful share of churners.")
    gate("campaign_breakeven_precision", candidate.get("precision_at_top_quintile"),
         g["min_precision_at_top_quintile"], ">=",
         "THE decisive gate. Break-even precision in the contacted slice is "
         "offer_cost / (uplift * margin) = 8 / 13.50 = 0.593. Below this the "
         "retention campaign destroys value, so no AUC improvement can rescue it.")

    if incumbent is None:
        gates.append({
            "gate": "no_regression_vs_production", "observed": None, "operator": "n/a",
            "threshold": g["max_auc_regression_vs_prod"], "passed": True,
            "rationale": "No incumbent in production; first model is promoted if "
                         "the absolute floors pass.",
        })
        delta = None
    else:
        delta = float(candidate["roc_auc"]) - float(incumbent["roc_auc"])
        gates.append({
            "gate": "no_regression_vs_production", "observed": round(delta, 6),
            "operator": ">=", "threshold": -g["max_auc_regression_vs_prod"],
            "passed": bool(delta >= -g["max_auc_regression_vs_prod"]),
            "rationale": "A candidate may not be worse than the live model by "
                         "more than the tolerance, even if the floors pass.",
        })

    passed_all = all(x["passed"] for x in gates)
    failed = [x["gate"] for x in gates if not x["passed"]]
    return {
        "decision": "PROMOTE" if passed_all else "DO_NOT_PROMOTE",
        "promote": passed_all,
        "candidate": candidate.get("model"),
        "incumbent": incumbent.get("model") if incumbent else None,
        "roc_auc_delta_vs_incumbent": delta,
        "failed_gates": failed,
        "gates": gates,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
def comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Tidy side-by-side comparison of several models on the same fold."""
    cols = ["model", "n", "base_rate", "roc_auc", "pr_auc", "accuracy",
            "precision", "recall", "f1", "brier", "log_loss",
            "precision_at_top_decile", "recall_at_top_decile", "lift_at_top_decile",
            "precision_at_top_quintile", "recall_at_top_quintile",
            "lift_at_top_quintile"]
    df = pd.DataFrame(results)
    return df.loc[:, [c for c in cols if c in df.columns]]


def save_eval_report(payload: dict[str, Any], filename: str,
                     cfg: Config | None = None) -> Path:
    cfg = cfg or CFG
    out_dir = cfg.get_path("paths.eval_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[eval] wrote {rel(path)}")
    return path
