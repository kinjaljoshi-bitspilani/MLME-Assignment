"""Tests for label construction, ingestion, monitoring and the retrain trigger."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data_prep import clean_events
from src.features import FEATURE_COLUMNS, FEATURE_RANGES, NUMERIC_FEATURES
from src.ingest import SchemaError, merge_into_warehouse, read_batch, validate_schema
from src.labels import customers_in_scope, make_labels, temporal_split
from src.monitoring import (
    build_reference_profile,
    check_data_quality,
    check_feature_drift,
    check_schema,
    population_stability_index,
    psi_numeric,
)
from src.retraining import RetrainingSignals, should_retrain

CFG = load_config()


# ==========================================================================
# Labels
# ==========================================================================
@pytest.fixture
def events() -> pd.DataFrame:
    rows = [
        ("A", "2024-01-10"), ("A", "2024-02-10"),   # active, buys again later
        ("B", "2024-01-15"),                        # active, never returns
        ("C", "2023-06-01"),                        # too old to be in scope
        ("D", "2024-02-20"), ("D", "2024-04-15"),   # active, buys in horizon
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "event_date"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["event_ts"] = df["event_date"]
    df["is_return"] = False
    df["invoice_no"] = ["I" + str(i) for i in range(len(df))]
    df["stock_code"] = "P1"
    df["quantity"] = 1
    df["unit_price"] = 10.0
    df["line_revenue"] = 10.0
    df["country"] = "United Kingdom"
    return df


ASOF = pd.Timestamp("2024-03-01")


def test_activity_window_scopes_population(events):
    ids = set(customers_in_scope(events, ASOF, activity_window_days=180))
    assert ids == {"A", "B", "D"}, "C bought >180 days ago and must be out of scope"


def test_label_is_one_when_no_purchase_in_horizon(events):
    ids = np.array(["A", "B", "D"])
    y = make_labels(events, ASOF, ids, horizon_days=90)
    # A's last purchase is 2024-02-10 (before as-of) and there is nothing after
    # as-of, so A churns. B churns. D buys on 2024-04-15, inside the horizon.
    assert y["A"] == 1 and y["B"] == 1 and y["D"] == 0


def test_label_respects_horizon_boundary(events):
    """A purchase beyond the horizon must not rescue the customer."""
    ids = np.array(["D"])
    assert make_labels(events, ASOF, ids, horizon_days=30)["D"] == 1
    assert make_labels(events, ASOF, ids, horizon_days=90)["D"] == 0


def test_returns_do_not_count_as_activity(events):
    df = events.copy()
    df.loc[df["customer_id"] == "B", "is_return"] = True
    assert "B" not in set(customers_in_scope(df, ASOF, 180))


def test_temporal_split_is_disjoint_in_time():
    table = pd.DataFrame({
        "customer_id": [str(i) for i in range(6)],
        "snapshot_date": pd.to_datetime(
            ["2011-03-31", "2011-04-30", "2011-05-31",
             "2011-06-30", "2011-07-31", "2011-08-31"]),
        "churn_90d": [0, 1, 0, 1, 0, 1],
    })
    splits = temporal_split(table, CFG)
    tr = set(pd.to_datetime(splits["train"]["snapshot_date"]))
    va = set(pd.to_datetime(splits["valid"]["snapshot_date"]))
    te = set(pd.to_datetime(splits["test"]["snapshot_date"]))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert max(tr) < min(va) < min(te), "folds must be strictly time-ordered"
    # The purge month is in none of the folds.
    assert pd.Timestamp("2011-07-31") not in (tr | va | te)


# ==========================================================================
# Ingestion
# ==========================================================================
def _source_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "InvoiceNo": ["1001", "1002", "C1003"],
        "StockCode": ["P1", "POST", "P1"],
        "Description": ["thing", "postage", "thing"],
        "Quantity": [5, 1, -2],
        "InvoiceDate": ["12/1/2011 10:00", "12/1/2011 10:05", "12/1/2011 11:00"],
        "UnitPrice": [3.0, 15.0, 3.0],
        "CustomerID": ["17850", "17850", "17850"],
        "Country": ["United Kingdom"] * 3,
    })


def test_validate_schema_accepts_good_frame():
    validate_schema(_source_frame())  # must not raise


def test_validate_schema_rejects_missing_column():
    bad = _source_frame().drop(columns=["UnitPrice"])
    with pytest.raises(SchemaError, match="UnitPrice"):
        validate_schema(bad)


def test_validate_schema_rejects_renamed_column():
    bad = _source_frame().rename(columns={"CustomerID": "customer_key"})
    with pytest.raises(SchemaError, match="CustomerID"):
        validate_schema(bad)


def test_validate_schema_tolerates_additive_column(capsys):
    """An extra upstream column is a warning, not a failure."""
    extra = _source_frame()
    extra["NewUpstreamField"] = 1
    validate_schema(extra)
    assert "new upstream column" in capsys.readouterr().out.lower()


def test_cleaning_drops_admin_codes_and_keeps_returns():
    events, audit = clean_events(_source_frame().rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity",
        "InvoiceDate": "event_ts", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country"}), CFG)
    assert "POST" not in set(events["stock_code"]), "admin code not removed"
    assert events["is_return"].sum() == 1, "cancellation should be flagged, not dropped"
    assert set(audit["rule"]) >= {"drop_non_product_stock_codes",
                                  "drop_exact_duplicate_lines"}


def test_cleaning_drops_missing_customer_id():
    src = _source_frame().rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity",
        "InvoiceDate": "event_ts", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country"})
    src.loc[0, "customer_id"] = None
    events, _ = clean_events(src, CFG)
    assert events["customer_id"].notna().all()


def test_merge_is_idempotent(tmp_path):
    """Re-ingesting the same batch must append zero rows."""
    src = _source_frame().rename(columns={
        "InvoiceNo": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity",
        "InvoiceDate": "event_ts", "UnitPrice": "unit_price",
        "CustomerID": "customer_id", "Country": "country"})
    events, _ = clean_events(src, CFG)
    wh = tmp_path / "events.parquet"

    first, total1 = merge_into_warehouse(events, wh)
    second, total2 = merge_into_warehouse(events, wh)
    assert first > 0, "first ingestion should append rows"
    assert second == 0, "re-ingesting the same file must be a no-op"
    assert total1 == total2


def test_read_batch_rejects_broken_file(tmp_path):
    bad = _source_frame().drop(columns=["Quantity"])
    path = tmp_path / "transactions_2011-12-01.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(SchemaError):
        read_batch(path)


# ==========================================================================
# Monitoring
# ==========================================================================
def _synthetic_batch(n: int = 800, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A clean synthetic batch that respects every declared feature range.

    Generating N(10, 3) for *every* column would put ratio features bounded to
    [0, 1] far out of range, and the data-quality check would correctly flag it -
    so the fixture has to honour the contract it is testing against.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for c in NUMERIC_FEATURES:
        lo, hi = FEATURE_RANGES[c]
        if hi <= 1.0:  # bounded ratio: keep inside [0, 1]
            centre = float(np.clip(0.5 + shift / 20.0, 0.05, 0.95))
            data[c] = np.clip(rng.normal(centre, 0.1, n), lo, hi)
        else:
            data[c] = np.clip(np.abs(rng.normal(10 + shift, 3, n)), lo, hi)
    data["country_grp"] = rng.choice(["United Kingdom", "France", "Other"],
                                     n, p=[0.8, 0.1, 0.1])
    data["customer_id"] = np.array([f"c{i}" for i in range(n)])
    return pd.DataFrame(data)


def test_psi_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert population_stability_index(p, p) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_divergence():
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    small = population_stability_index(ref, np.array([0.30, 0.25, 0.25, 0.20]))
    large = population_stability_index(ref, np.array([0.70, 0.10, 0.10, 0.10]))
    assert 0 < small < large


def test_psi_is_finite_with_empty_bucket():
    """An empty bucket must not send PSI to infinity."""
    psi = population_stability_index(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert np.isfinite(psi)


def test_reference_profile_round_trips():
    batch = _synthetic_batch()
    prof = build_reference_profile(batch)
    assert set(prof["numeric"]) == set(NUMERIC_FEATURES)
    # Scoring the reference against itself must show no drift.
    for col, ref in prof["numeric"].items():
        assert psi_numeric(ref, batch[col]) == pytest.approx(0.0, abs=1e-6)


def test_drift_detected_on_shifted_batch():
    ref = build_reference_profile(_synthetic_batch(seed=1))
    shifted = _synthetic_batch(shift=6.0, seed=2)
    result = check_feature_drift(shifted, reference=ref, cfg=CFG)
    assert result["severity"] == "alert"
    assert result["n_alert"] > 0
    assert result["drift_share"] > CFG["monitoring"]["drift_share_alert"]


def test_no_drift_on_same_distribution():
    ref = build_reference_profile(_synthetic_batch(seed=1))
    same = _synthetic_batch(seed=99)
    result = check_feature_drift(same, reference=ref, cfg=CFG)
    assert result["severity"] in {"ok", "warn"}
    assert result["n_alert"] == 0


def test_schema_check_flags_missing_feature():
    batch = _synthetic_batch().drop(columns=["recency_days"])
    out = check_schema(batch)
    assert not out["passed"] and "recency_days" in out["missing_columns"]


def test_schema_check_flags_wrong_dtype():
    batch = _synthetic_batch()
    batch["recency_days"] = "not a number"
    out = check_schema(batch)
    assert not out["passed"]
    assert "recency_days" in out["non_numeric_numeric_columns"]


def test_data_quality_flags_nulls_and_ranges():
    batch = _synthetic_batch()
    batch.loc[:200, "revenue_90d"] = np.nan          # ~25% nulls
    batch.loc[:50, "active_days_ratio"] = 7.5        # out of [0, 1]
    out = check_data_quality(batch, CFG)
    issues = {(f["feature"], f["issue"]) for f in out["findings"]}
    assert ("revenue_90d", "high_null_rate") in issues
    assert ("active_days_ratio", "out_of_range") in issues
    assert out["severity"] == "alert"


def test_data_quality_flags_constant_column():
    batch = _synthetic_batch()
    batch["orders_90d"] = 3.0
    out = check_data_quality(batch, CFG)
    assert any(f["issue"] == "constant_column" for f in out["findings"])


def test_clean_batch_passes_quality():
    out = check_data_quality(_synthetic_batch(), CFG)
    assert out["severity"] == "ok" and out["n_findings"] == 0


# ==========================================================================
# Retraining trigger
# ==========================================================================
def test_no_trigger_when_everything_is_healthy():
    r = should_retrain(RetrainingSignals(
        model_age_days=5, new_labelled_snapshots=0,
        recent_auc=0.79, auc_at_promotion=0.79), CFG)
    assert not r.should_retrain and r.urgency == "none"


def test_performance_drop_fires_alone():
    """Measured degradation is ground truth and needs no other signal."""
    r = should_retrain(RetrainingSignals(
        model_age_days=1, new_labelled_snapshots=0,
        recent_auc=0.70, auc_at_promotion=0.79), CFG)
    assert r.should_retrain and r.urgency == "high"
    assert "performance_degradation" in r.fired


def test_staleness_needs_fresh_labels():
    stale_no_labels = should_retrain(RetrainingSignals(
        model_age_days=90, new_labelled_snapshots=0), CFG)
    assert not stale_no_labels.should_retrain, (
        "a calendar-only retrain would refit the same data at full cost")

    stale_with_labels = should_retrain(RetrainingSignals(
        model_age_days=90, new_labelled_snapshots=2), CFG)
    assert stale_with_labels.should_retrain
    assert "model_staleness" in stale_with_labels.fired


def test_drift_needs_fresh_labels():
    drift_only = should_retrain(RetrainingSignals(
        model_age_days=1, new_labelled_snapshots=0,
        drift_share=0.8, drift_status="alert"), CFG)
    assert not drift_only.should_retrain, (
        "retraining on drifted inputs with stale targets encodes the drift")

    drift_with_labels = should_retrain(RetrainingSignals(
        model_age_days=1, new_labelled_snapshots=1,
        drift_share=0.8, drift_status="alert"), CFG)
    assert drift_with_labels.should_retrain
    assert "feature_drift" in drift_with_labels.fired


def test_schema_break_blocks_retraining_entirely():
    r = should_retrain(RetrainingSignals(
        model_age_days=999, new_labelled_snapshots=5,
        recent_auc=0.50, auc_at_promotion=0.79,
        drift_share=1.0, drift_status="alert", schema_ok=False), CFG)
    assert not r.should_retrain
    assert r.urgency == "blocked" and "schema_break" in r.blocked_by


def test_data_quality_failure_blocks_retraining():
    r = should_retrain(RetrainingSignals(
        model_age_days=999, new_labelled_snapshots=5,
        data_quality_ok=False), CFG)
    assert not r.should_retrain and "data_quality_failure" in r.blocked_by


def test_result_is_serialisable_and_traceable():
    r = should_retrain(RetrainingSignals(
        model_age_days=90, new_labelled_snapshots=1), CFG)
    payload = json.dumps(r.to_dict())
    assert "reasons" in payload and len(r.reasons) >= 2, (
        "every decision must carry a human-auditable reason trace")
