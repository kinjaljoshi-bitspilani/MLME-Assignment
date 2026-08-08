"""Tests for the shared feature module.

The highest-value tests in an ML repo are not "does the model score well" but
"can the pipeline produce a wrong number silently". These target exactly that:
point-in-time leakage, the offline/online contract, and the invariants the
serving layer assumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    NUMERIC_FEATURES,
    TIME_DEPENDENT_FEATURES,
    align_features,
    compute_features,
    refresh_time_dependent_features,
)


# --------------------------------------------------------------------------
@pytest.fixture
def events() -> pd.DataFrame:
    """A tiny hand-built event log with known, hand-checkable answers.

    Customer A: 3 purchase days (Jan 1, Jan 11, Jan 21) -> gaps of 10 and 10.
    Customer B: 1 purchase day (Jan 5) -> no gap defined.
    Customer C: purchases on Jan 2 and a return on Jan 3.
    Customer D: purchases only AFTER the as-of date -> must be invisible.
    """
    rows = [
        # A
        ("A", "2024-01-01", "INV1", "P1", 2, 5.0, False, "United Kingdom"),
        ("A", "2024-01-11", "INV2", "P2", 1, 10.0, False, "United Kingdom"),
        ("A", "2024-01-21", "INV3", "P1", 3, 5.0, False, "United Kingdom"),
        # B
        ("B", "2024-01-05", "INV4", "P3", 10, 2.0, False, "France"),
        # C
        ("C", "2024-01-02", "INV5", "P1", 4, 5.0, False, "Germany"),
        ("C", "2024-01-03", "C900", "P1", -1, 5.0, True, "Germany"),
        # D - future only
        ("D", "2024-03-01", "INV6", "P4", 1, 99.0, False, "Spain"),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "event_date", "invoice_no",
                                     "stock_code", "quantity", "unit_price",
                                     "is_return", "country"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["event_ts"] = df["event_date"]
    df["line_revenue"] = df["quantity"] * df["unit_price"]
    return df


ASOF = pd.Timestamp("2024-01-31")


# ---------------------------- point-in-time -------------------------------
def test_no_future_events_leak(events):
    """A customer whose only activity is after as-of must not appear at all."""
    out = compute_features(events, asof=ASOF)
    assert "D" not in set(out["customer_id"]), "future-only customer leaked in"


def test_features_ignore_post_asof_rows(events):
    """Adding a future row must not change any feature value."""
    before = compute_features(events, asof=ASOF).set_index("customer_id")
    extra = events.iloc[[0]].copy()
    extra["customer_id"] = "A"
    extra["event_date"] = pd.Timestamp("2024-02-15")
    extra["event_ts"] = extra["event_date"]
    extra["invoice_no"] = "INV_FUTURE"
    after = compute_features(pd.concat([events, extra]),
                             asof=ASOF).set_index("customer_id")
    pd.testing.assert_frame_equal(
        before.loc[["A"], NUMERIC_FEATURES], after.loc[["A"], NUMERIC_FEATURES])


def test_recency_and_tenure_arithmetic(events):
    out = compute_features(events, asof=ASOF).set_index("customer_id")
    # A last bought 2024-01-21; as-of is 2024-01-31 -> 10 days.
    assert out.loc["A", "recency_days"] == 10
    # A first bought 2024-01-01 -> 30 days of tenure.
    assert out.loc["A", "tenure_days"] == 30
    # Gaps are 10 and 10 -> mean 10, std 0 -> cv 0.
    assert out.loc["A", "avg_interpurchase_gap_days"] == pytest.approx(10.0)
    assert out.loc["A", "interpurchase_gap_cv"] == pytest.approx(0.0, abs=1e-6)
    # 10 days dormant against a 10-day cadence -> exactly one cycle overdue.
    assert out.loc["A", "recency_over_avg_gap"] == pytest.approx(1.0, rel=1e-4)


def test_single_purchase_customer_gets_sentinel_gap(events):
    """One purchase means no gap exists; it must not become NaN."""
    out = compute_features(events, asof=ASOF).set_index("customer_id")
    assert out.loc["B", "avg_interpurchase_gap_days"] == 999.0
    assert not out.loc["B", NUMERIC_FEATURES].isna().any()


def test_returns_excluded_from_revenue_but_counted_as_returns(events):
    out = compute_features(events, asof=ASOF).set_index("customer_id")
    # C bought 4 x 5.00 = 20.00 and returned 1 x 5.00.
    assert out.loc["C", "revenue_180d"] == pytest.approx(20.0)
    assert out.loc["C", "n_returns_180d"] == 1
    assert out.loc["C", "return_value_ratio_180d"] > 0


def test_windowed_aggregates_are_nested(events):
    """Longer windows must dominate shorter ones - a basic monotonicity check."""
    out = compute_features(events, asof=ASOF)
    assert (out["orders_180d"] >= out["orders_90d"]).all()
    assert (out["orders_90d"] >= out["orders_30d"]).all()
    assert (out["revenue_180d"] >= out["revenue_90d"] - 1e-9).all()


# ------------------------------ contract ----------------------------------
def test_column_contract_exact_and_ordered(events):
    out = compute_features(events, asof=ASOF)
    assert list(out.columns) == ["customer_id", "snapshot_date"] + FEATURE_COLUMNS


def test_no_nulls_or_infinities(events):
    out = compute_features(events, asof=ASOF)
    assert not out[NUMERIC_FEATURES].isna().any().any()
    assert np.isfinite(out[NUMERIC_FEATURES].to_numpy()).all()


def test_all_features_within_declared_ranges(events):
    """Winsorisation must hold, or the API's Pydantic bounds would reject our
    own offline values."""
    out = compute_features(events, asof=ASOF)
    for col, (lo, hi) in FEATURE_RANGES.items():
        assert out[col].between(lo, hi).all(), f"{col} outside [{lo}, {hi}]"


def test_country_whitelist_bounds_cardinality(events):
    out = compute_features(events, asof=ASOF,
                           country_whitelist=["United Kingdom"])
    assert set(out["country_grp"]) <= {"United Kingdom", "Other", "Unknown"}


def test_align_features_raises_on_missing_column(events):
    out = compute_features(events, asof=ASOF)
    broken = out.drop(columns=["recency_days"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        align_features(broken, strict=True)


def test_align_features_reorders(events):
    out = compute_features(events, asof=ASOF)
    shuffled = out.loc[:, list(reversed(FEATURE_COLUMNS))]
    assert list(align_features(shuffled).columns) == FEATURE_COLUMNS


# ------------------------- training/serving skew ---------------------------
def test_reaging_matches_recomputation():
    """THE skew test.

    A cached feature row re-aged by N days must equal a full recomputation at
    the later date. If this ever fails, the online service is serving a
    different number than training would have produced for the same customer at
    the same instant - the exact bug the shared module exists to prevent.
    """
    rows = [("A", "2024-01-01", "INV1", "P1", 2, 5.0, False, "United Kingdom"),
            ("A", "2024-01-11", "INV2", "P2", 1, 10.0, False, "United Kingdom")]
    df = pd.DataFrame(rows, columns=["customer_id", "event_date", "invoice_no",
                                     "stock_code", "quantity", "unit_price",
                                     "is_return", "country"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["event_ts"] = df["event_date"]
    df["line_revenue"] = df["quantity"] * df["unit_price"]

    cached = compute_features(df, asof="2024-01-20").iloc[0].to_dict()
    cached["feature_timestamp"] = pd.Timestamp("2024-01-20")
    reaged = refresh_time_dependent_features(cached, "2024-01-27")

    fresh = compute_features(df, asof="2024-01-27").iloc[0].to_dict()
    for feat in TIME_DEPENDENT_FEATURES:
        assert reaged[feat] == pytest.approx(fresh[feat], rel=1e-6), (
            f"offline/online skew on {feat}: "
            f"re-aged={reaged[feat]} vs recomputed={fresh[feat]}")


def test_reaging_is_a_noop_when_fresh():
    row = {"recency_days": 5.0, "tenure_days": 50.0, "active_days_ratio": 0.5,
           "avg_interpurchase_gap_days": 10.0, "recency_over_avg_gap": 0.5,
           "feature_timestamp": pd.Timestamp("2024-01-20")}
    out = refresh_time_dependent_features(row, "2024-01-20")
    assert out["recency_days"] == 5.0
    assert "_feature_staleness_days" not in out


def test_reaging_reports_staleness():
    row = {"recency_days": 5.0, "tenure_days": 50.0,
           "avg_interpurchase_gap_days": 10.0,
           "feature_timestamp": pd.Timestamp("2024-01-20")}
    out = refresh_time_dependent_features(row, "2024-01-23")
    assert out["_feature_staleness_days"] == 3
    assert out["recency_days"] == 8.0


# ---------------------------- determinism ----------------------------------
def test_deterministic_across_row_order(events):
    """Feature values must not depend on input row ordering."""
    a = compute_features(events, asof=ASOF).set_index("customer_id").sort_index()
    shuffled = events.sample(frac=1.0, random_state=7)
    b = compute_features(shuffled, asof=ASOF).set_index("customer_id").sort_index()
    pd.testing.assert_frame_equal(a[NUMERIC_FEATURES], b[NUMERIC_FEATURES])


def test_feature_families_are_declared():
    """Every column belongs to exactly one declared family."""
    assert set(FEATURE_COLUMNS) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert set(TIME_DEPENDENT_FEATURES) <= set(NUMERIC_FEATURES)
