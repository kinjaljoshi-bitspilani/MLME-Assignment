"""Repeatable training pipeline.

    ingest -> event table -> snapshots(features+labels) -> temporal split
        -> fit baseline + candidate -> calibrate -> evaluate -> guardrail
        -> register + (conditionally) promote -> write reference stats

Design notes that matter for correctness:

* **The artefact is a Pipeline, not a bare estimator.** Imputation, scaling and
  one-hot encoding are fitted *inside* the same object that is serialised. The
  serving process therefore cannot apply a different transform to what training
  applied - the transform travels with the model.
* **Nothing is fitted on validation or test.** The encoder, the scaler and the
  calibrator each see only the fold they are entitled to see.
* **Calibration is fitted on validation**, then the model is evaluated on the
  untouched test fold, because the probabilities are multiplied by margin
  downstream and a mis-calibrated ranking loses money even at equal AUC.
* **Reference statistics are written at training time.** Drift is meaningless
  without a frozen baseline; the baseline must be the training distribution of
  the model that is actually live.

Usage
-----
    python -m src.train                    # full run
    python -m src.train --rebuild-events   # re-clean raw CSV first
    python -m src.train --no-promote       # evaluate and register only
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from inspect import signature
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CFG, REPO_ROOT, Config, rel
from .data_prep import build_event_table, load_event_table
from .evaluate import (
    comparison_table,
    evaluate_model,
    expected_campaign_value,
    promotion_decision,
    save_eval_report,
)
from .features import (
    ASOF_COL,
    CATEGORICAL_FEATURES,
    ENTITY_COL,
    FEATURE_COLUMNS,
    LABEL_COL,
    NUMERIC_FEATURES,
)
from .labels import build_training_table, split_report, temporal_split
from .monitoring import write_reference_stats
from .registry import ModelRegistry


# --------------------------------------------------------------------------
# Preprocessing - shared by every model so comparisons are apples-to-apples
# --------------------------------------------------------------------------
def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Numeric imputation (+ optional scaling) and bounded one-hot encoding.

    ``scale=True`` for the linear baseline, which needs it; ``False`` for the
    tree candidate, which does not and for which scaling only adds a step that
    could go wrong.

    ``handle_unknown="infrequent_if_exist"`` matters at serving time: a country
    never seen in training must not raise, it must fall into the learned
    infrequent bucket.
    """
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                          min_frequency=20, sparse_output=False)),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class RecencyHeuristic(BaseEstimator, ClassifierMixin):
    """The rule the business already uses: "dormant for a while => at risk".

    Scores by ``recency_days`` alone, min-max scaled into [0, 1] using the
    training range. This is the reference any ML model must beat to justify its
    own existence - if a 22-feature gradient-booster cannot outperform one
    column and a sort, the modelling effort is not paying for itself.
    """

    def fit(self, X: pd.DataFrame, y=None):
        r = pd.to_numeric(X["recency_days"], errors="coerce")
        self.lo_ = float(np.nanpercentile(r, 1))
        self.hi_ = float(np.nanpercentile(r, 99))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        r = pd.to_numeric(X["recency_days"], errors="coerce").fillna(self.lo_)
        p = ((r - self.lo_) / max(self.hi_ - self.lo_, 1e-9)).clip(0.001, 0.999)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_heuristic() -> Pipeline:
    """No preprocessing: the heuristic reads one raw column by design."""
    return Pipeline([("clf", RecencyHeuristic())])


def build_baseline(cfg: Config) -> Pipeline:
    """Regularised logistic regression - the honest, cheap reference."""
    p = cfg["training"]["baseline"]
    return Pipeline([
        ("prep", build_preprocessor(scale=True)),
        ("clf", LogisticRegression(
            C=p["C"], max_iter=p["max_iter"], class_weight=p["class_weight"],
            solver="lbfgs", random_state=cfg["project"]["seed"])),
    ])


def build_candidate(cfg: Config) -> tuple[Pipeline, str]:
    """Gradient-boosted trees. Falls back to sklearn if LightGBM is absent."""
    p = cfg["training"]["candidate"]
    seed = cfg["project"]["seed"]
    try:
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
            subsample=p["subsample"], subsample_freq=p["subsample_freq"],
            colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        kind = "lightgbm"
    except ImportError:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
            max_leaf_nodes=p["num_leaves"], min_samples_leaf=p["min_child_samples"],
            l2_regularization=p["reg_lambda"], early_stopping=True,
            validation_fraction=0.15, random_state=seed,
        )
        kind = "hist_gradient_boosting"
    return Pipeline([("prep", build_preprocessor(scale=False)), ("clf", clf)]), kind


# --------------------------------------------------------------------------
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def run_environment() -> dict[str, Any]:
    """Captured with every run: reproducibility is (code + config + env)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "git_sha": _git_sha(),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def tune_n_estimators_by_early_stopping(
    pipeline: Pipeline, train: pd.DataFrame, cfg: Config
) -> dict[str, Any]:
    """Find the boosting round count on an INNER temporal fold of the training data.

    Why not early-stop on the validation fold: validation is already used for
    model selection, threshold choice and probability calibration. Early
    stopping there too would let the candidate peek at the same rows three
    different ways and make the validation metric optimistic.

    So the training snapshots are themselves split by time - all but the last
    snapshot fit the model, the last snapshot decides when to stop - and the
    model is then refitted on the *full* training set for the number of rounds
    that was found. Validation stays genuinely held out.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "n_estimators"):
        return {"tuned": False, "reason": "estimator has no n_estimators"}

    dates = sorted(pd.to_datetime(train[ASOF_COL]).unique())
    if len(dates) < 2:
        return {"tuned": False, "reason": "need >= 2 training snapshots"}

    inner_cut = dates[-1]
    d = pd.to_datetime(train[ASOF_COL])
    inner_tr, inner_va = train.loc[d < inner_cut], train.loc[d == inner_cut]

    prep = build_preprocessor(scale=False)
    Xtr = prep.fit_transform(inner_tr.loc[:, FEATURE_COLUMNS])
    Xva = prep.transform(inner_va.loc[:, FEATURE_COLUMNS])
    ytr = inner_tr[LABEL_COL].to_numpy()
    yva = inner_va[LABEL_COL].to_numpy()

    rounds = cfg["training"]["candidate"]["early_stopping_rounds"]
    try:
        import lightgbm as lgb

        probe = clf.__class__(**{**clf.get_params(), "n_estimators": clf.n_estimators})
        fit_kwargs: dict[str, Any] = {
            "eval_metric": "auc",
            "callbacks": [lgb.early_stopping(rounds, verbose=False),
                          lgb.log_evaluation(0)],
        }
        # LightGBM >= 4.6 prefers eval_X / eval_y; older builds only take eval_set.
        if "eval_X" in signature(probe.fit).parameters:
            fit_kwargs.update(eval_X=Xva, eval_y=yva)
        else:  # pragma: no cover
            fit_kwargs.update(eval_set=[(Xva, yva)])
        probe.fit(Xtr, ytr, **fit_kwargs)
        best = int(probe.best_iteration_ or clf.n_estimators)
    except ImportError:  # pragma: no cover - sklearn fallback self-tunes
        return {"tuned": False, "reason": "lightgbm unavailable"}

    # Refitting on ~33% more data supports slightly more capacity.
    scaled = max(int(best * len(train) / max(len(inner_tr), 1)), 20)
    pipeline.set_params(clf__n_estimators=scaled)
    info = {
        "tuned": True,
        "inner_train_snapshots": [str(pd.Timestamp(x).date()) for x in dates[:-1]],
        "inner_eval_snapshot": str(pd.Timestamp(inner_cut).date()),
        "best_iteration_inner": best,
        "n_estimators_refit": scaled,
        "n_estimators_configured_cap": int(clf.n_estimators),
    }
    print(f"[train] early stopping: best_iteration={best} on "
          f"{info['inner_eval_snapshot']} -> refit with n_estimators={scaled}")
    return info


def fit_and_score(
    model: Pipeline,
    label: str,
    splits: dict[str, pd.DataFrame],
    cfg: Config,
    calibrate: bool = False,
    early_stop: bool = False,
) -> dict[str, Any]:
    """Fit on train, optionally calibrate on validation, score all folds."""
    Xtr = splits["train"].loc[:, FEATURE_COLUMNS]
    ytr = splits["train"][LABEL_COL].to_numpy()

    es_info = (tune_n_estimators_by_early_stopping(model, splits["train"], cfg)
               if early_stop else {"tuned": False, "reason": "not requested"})

    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    scorer: Any = model
    if calibrate:
        # Fit the calibrator on the validation fold only. Wrapping in
        # `FrozenEstimator` guarantees the underlying model is not refitted, so
        # we learn the probability mapping on a fold the model never trained on.
        # (`cv="prefit"` was the pre-1.6 spelling of this and is now removed.)
        Xva = splits["valid"].loc[:, FEATURE_COLUMNS]
        yva = splits["valid"][LABEL_COL].to_numpy()
        scorer = CalibratedClassifierCV(
            FrozenEstimator(model), method=cfg["training"]["calibration"])
        scorer.fit(Xva, yva)

    out: dict[str, Any] = {"label": label, "model": model, "scorer": scorer,
                           "fit_seconds": round(fit_s, 3),
                           "early_stopping": es_info, "folds": {}}
    for fold, part in splits.items():
        prob = scorer.predict_proba(part.loc[:, FEATURE_COLUMNS])[:, 1]
        out["folds"][fold] = {
            "metrics": evaluate_model(part[LABEL_COL], prob, label=label),
            "value": expected_campaign_value(part[LABEL_COL], prob, cfg),
            "probabilities": prob,
        }
    return out


def permutation_importance_report(
    scorer: Any, part: pd.DataFrame, n_repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Model-agnostic importance measured on a held-out fold.

    Preferred over LightGBM's split-count importance because it is measured in
    the metric we care about (AUC) on data the model has not seen, so it
    reflects genuine predictive contribution rather than tree-building
    mechanics.
    """
    from sklearn.inspection import permutation_importance

    X = part.loc[:, FEATURE_COLUMNS]
    y = part[LABEL_COL].to_numpy()
    res = permutation_importance(
        scorer, X, y, scoring="roc_auc", n_repeats=n_repeats,
        random_state=seed, n_jobs=1)
    return (
        pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "auc_drop_mean": res.importances_mean,
            "auc_drop_std": res.importances_std,
        })
        .sort_values("auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
def run_training(
    cfg: Config | None = None,
    rebuild_events: bool = False,
    allow_promotion: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """The whole pipeline. Returns everything the notebook and report need."""
    cfg = cfg or CFG
    np.random.seed(cfg["project"]["seed"])

    # ---- 1. data -----------------------------------------------------
    events = build_event_table(cfg) if rebuild_events else load_event_table(cfg)
    if verbose:
        print(f"[train] events={len(events):,} "
              f"customers={events[ENTITY_COL].nunique():,}")

    table = build_training_table(events, cfg, write=True)
    splits = temporal_split(table, cfg)
    report = split_report(splits)
    if verbose:
        print(report.to_string(index=False))

    # ---- 2. models ---------------------------------------------------
    # A prevalence-only classifier: the floor any real model must clear.
    trivial = Pipeline([("prep", build_preprocessor(scale=False)),
                        ("clf", DummyClassifier(strategy="prior"))])
    baseline = build_baseline(cfg)
    candidate, candidate_kind = build_candidate(cfg)

    # Four tiers of increasing sophistication. Each must justify itself against
    # the one below: prevalence -> business rule -> linear model -> boosted trees.
    runs = {
        "trivial_prior": fit_and_score(trivial, "trivial_prior", splits, cfg),
        "heuristic_recency": fit_and_score(build_heuristic(), "heuristic_recency",
                                           splits, cfg),
        "baseline_logreg": fit_and_score(baseline, "baseline_logreg", splits, cfg),
        "candidate_gbdt": fit_and_score(candidate, f"candidate_{candidate_kind}",
                                        splits, cfg, calibrate=True,
                                        early_stop=True),
    }

    # ---- 3. comparison ----------------------------------------------
    valid_rows = [r["folds"]["valid"]["metrics"] for r in runs.values()]
    test_rows = [r["folds"]["test"]["metrics"] for r in runs.values()]
    cmp_valid = comparison_table(valid_rows)
    cmp_test = comparison_table(test_rows)
    if verbose:
        print("\n[train] validation fold\n" + cmp_valid.to_string(index=False))
        print("\n[train] test fold\n" + cmp_test.to_string(index=False))

    # ---- 4. guardrail -----------------------------------------------
    # Selection uses validation; the test fold is reported once, at the end.
    cand_valid = runs["candidate_gbdt"]["folds"]["valid"]["metrics"]
    base_valid = runs["baseline_logreg"]["folds"]["valid"]["metrics"]
    decision = promotion_decision(cand_valid, base_valid, cfg)
    if verbose:
        print(f"\n[train] promotion decision: {decision['decision']}")
        for g in decision["gates"]:
            flag = "PASS" if g["passed"] else "FAIL"
            print(f"   [{flag}] {g['gate']}: observed={g['observed']} "
                  f"{g['operator']} {g['threshold']}")

    # ---- 5. registry -------------------------------------------------
    reg = ModelRegistry(cfg)
    env = run_environment()
    entries = {}
    for key, spec in [
        ("baseline_logreg", cfg["training"]["baseline"]),
        ("candidate_gbdt", cfg["training"]["candidate"]),
    ]:
        run = runs[key]
        entries[key] = reg.register(
            name=cfg["project"]["name"],
            estimator=run["scorer"],
            metrics={
                "valid_roc_auc": run["folds"]["valid"]["metrics"]["roc_auc"],
                "valid_pr_auc": run["folds"]["valid"]["metrics"]["pr_auc"],
                "valid_brier": run["folds"]["valid"]["metrics"]["brier"],
                "test_roc_auc": run["folds"]["test"]["metrics"]["roc_auc"],
                "test_pr_auc": run["folds"]["test"]["metrics"]["pr_auc"],
            },
            params={"family": key, **dict(spec)},
            feature_columns=FEATURE_COLUMNS,
            stage="staging",
            extra={
                "label_definition": dict(cfg["label"]),
                "train_snapshots": cfg["split"]["train_snapshots"],
                "n_train_rows": int(len(splits["train"])),
                "country_whitelist": list(table.attrs.get("country_whitelist", [])),
                "environment": env,
                "fit_seconds": run["fit_seconds"],
            },
        )

    # ---- promotion: the guardrail is binding, not advisory ------------
    # Ordering matters. If the candidate fails its gates we do NOT quietly ship
    # something else; we keep whatever is already live. Only when the registry
    # has no production model at all (cold start, as on a first ever run) do we
    # fall back - and then only to a model that itself clears the gates.
    promoted = None
    promotion_note = "promotion not attempted (allow_promotion=False)"
    if allow_promotion:
        incumbent_entry = reg.get_stage("production")
        if decision["promote"]:
            promoted = reg.promote(entries["candidate_gbdt"]["model_id"])
            promotion_note = "candidate cleared all gates and was promoted"
        elif incumbent_entry is not None:
            promoted = incumbent_entry
            promotion_note = (
                f"candidate blocked by {decision['failed_gates']}; "
                f"incumbent {incumbent_entry['model_id']} retained")
            print(f"[train] {promotion_note}")
        else:
            fallback = promotion_decision(base_valid, None, cfg)
            if fallback["promote"]:
                promoted = reg.promote(entries["baseline_logreg"]["model_id"])
                promotion_note = (
                    "cold start: candidate failed its gates, baseline cleared "
                    "them and was promoted so the service is functional")
            else:
                promotion_note = (
                    "cold start: NEITHER model cleared the gates. Nothing was "
                    "promoted; the service stays degraded and the modelling "
                    "approach needs rework before any deployment.")
            print(f"[train] {promotion_note}")

    # ---- 6. importance + reference stats -----------------------------
    importance = permutation_importance_report(
        runs["candidate_gbdt"]["scorer"], splits["valid"],
        seed=cfg["project"]["seed"])

    ref_path = write_reference_stats(
        splits["train"], cfg,
        model_id=promoted["model_id"] if promoted else None)

    # also persist the online feature table for the serving demo
    fs_path = materialise_feature_store(events, cfg)

    # ---- 7. reports ---------------------------------------------------
    payload = {
        "run": env,
        "config_used": {k: dict(cfg[k]) for k in
                        ["label", "split", "features", "training", "promotion",
                         "business"]},
        "data": {
            "n_events": int(len(events)),
            "n_customers": int(events[ENTITY_COL].nunique()),
            "event_date_min": str(events["event_date"].min().date()),
            "event_date_max": str(events["event_date"].max().date()),
            "snapshot_table_rows": int(len(table)),
        },
        "splits": report.to_dict("records"),
        "metrics_valid": cmp_valid.to_dict("records"),
        "metrics_test": cmp_test.to_dict("records"),
        "business_value_test": {
            k: r["folds"]["test"]["value"] for k, r in runs.items()},
        "promotion": decision,
        "registered": {k: {"model_id": v["model_id"], "stage": v["stage"]}
                       for k, v in entries.items()},
        "production_model": promoted["model_id"] if promoted else None,
        "top_features": importance.head(12).to_dict("records"),
        "reference_stats_path": rel(ref_path),
        "feature_store_path": rel(fs_path),
    }
    save_eval_report(payload, "training_report.json", cfg)
    (cfg.get_path("paths.eval_dir") / "metrics_valid.csv").write_text(
        cmp_valid.to_csv(index=False))
    (cfg.get_path("paths.eval_dir") / "metrics_test.csv").write_text(
        cmp_test.to_csv(index=False))
    (cfg.get_path("paths.eval_dir") / "feature_importance.csv").write_text(
        importance.to_csv(index=False))
    _write_markdown_summary(payload, cmp_valid, cmp_test, importance, cfg)

    return {"events": events, "table": table, "splits": splits, "runs": runs,
            "cmp_valid": cmp_valid, "cmp_test": cmp_test, "decision": decision,
            "registry": reg, "importance": importance, "report": payload,
            "split_report": report}


def materialise_feature_store(events: pd.DataFrame, cfg: Config | None = None) -> Path:
    """Write the online feature table: latest feature row per customer.

    This is the "precompute nightly, look up online" half of the hybrid serving
    pattern. It is produced by ``features.compute_features`` - the same function
    the training table uses - which is what makes the online values consistent
    with training by construction.
    """
    from .features import compute_features

    cfg = cfg or CFG
    asof = pd.Timestamp(cfg["project"]["system_asof"])
    hist = events.loc[events["event_date"] <= asof]
    whitelist = (hist.loc[~hist["is_return"], "country"].value_counts()
                 .head(cfg["features"]["top_n_countries"]).index.tolist())
    feats = compute_features(
        events, asof=asof,
        windows=tuple(cfg["features"]["windows_days"]),
        top_n_countries=cfg["features"]["top_n_countries"],
        country_whitelist=whitelist)
    feats["feature_timestamp"] = asof
    path = cfg.get_path("paths.feature_store")
    feats.to_parquet(path, index=False)
    print(f"[train] feature store: {len(feats):,} rows -> {rel(path)}")
    return path


def _write_markdown_summary(payload, cmp_valid, cmp_test, importance, cfg) -> Path:
    lines = [
        "# Offline Evaluation Report",
        "",
        f"- Project: `{cfg['project']['name']}`",
        f"- Run (UTC): {payload['run']['run_at_utc']}",
        f"- Label: `{cfg['label']['name']}` = no purchase within "
        f"{cfg['label']['horizon_days']} days, scoped to customers active in the "
        f"prior {cfg['label']['activity_window_days']} days",
        f"- Events: {payload['data']['n_events']:,} "
        f"({payload['data']['event_date_min']} to {payload['data']['event_date_max']})",
        f"- Supervised rows: {payload['data']['snapshot_table_rows']:,}",
        "",
        "## Temporal split", "",
        pd.DataFrame(payload["splits"]).to_markdown(index=False),
        "", "## Validation fold", "", cmp_valid.round(4).to_markdown(index=False),
        "", "## Test fold (untouched until now)", "",
        cmp_test.round(4).to_markdown(index=False),
        "", "## Promotion guardrail", "",
        f"**Decision: {payload['promotion']['decision']}**", "",
        pd.DataFrame(payload["promotion"]["gates"]).to_markdown(index=False),
        "", "## Top features (permutation importance, validation AUC drop)", "",
        importance.head(12).round(5).to_markdown(index=False),
        "", "## Expected campaign value on the test fold", "",
        pd.DataFrame(payload["business_value_test"]).T.to_markdown(),
        "",
        f"Production model: `{payload['production_model']}`", "",
    ]
    path = cfg.get_path("paths.eval_dir") / "evaluation_report.md"
    path.write_text("\n".join(lines))
    print(f"[eval] wrote {rel(path)}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the churn model")
    ap.add_argument("--rebuild-events", action="store_true")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    from .config import load_config

    cfg = load_config(args.config) if args.config else CFG
    run_training(cfg, rebuild_events=args.rebuild_events,
                 allow_promotion=not args.no_promote)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
