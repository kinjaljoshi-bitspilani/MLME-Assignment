"""Feature engineering - THE single source of truth for feature logic.

This module is the primary defence against training/serving skew. It is
imported by:

  * ``src/train.py``      (offline, builds the training table)
  * ``src/serving/app.py`` (online, materialises / refreshes features per request)
  * ``src/monitoring.py``  (drift checks operate on the same column contract)

If a feature definition changes it changes in exactly one place, so an offline
and an online value for the same customer at the same instant are equal by
construction rather than by convention.

Two guarantees are enforced here:

1. **Point-in-time correctness.** ``compute_features`` never looks at an event
   with ``event_date > asof``. The filter happens once, at the top of the
   function, so no individual feature can accidentally bypass it.
2. **A frozen column contract.** ``FEATURE_COLUMNS`` fixes the order and
   membership of the feature vector. ``align_features`` re-indexes any frame to
   that contract, so a silently reordered or missing column becomes an
   explicit, catchable error instead of a wrong prediction.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column contract
# --------------------------------------------------------------------------
ENTITY_COL = "customer_id"
ASOF_COL = "snapshot_date"
LABEL_COL = "churn_90d"

NUMERIC_FEATURES: list[str] = [
    # --- recency / tenure ------------------------------------------------
    "recency_days",
    "tenure_days",
    "active_days_ratio",
    # --- inter-purchase rhythm -------------------------------------------
    "avg_interpurchase_gap_days",
    "interpurchase_gap_cv",
    "recency_over_avg_gap",          # ratio feature - the strongest signal
    # --- frequency by window ---------------------------------------------
    "orders_30d",
    "orders_90d",
    "orders_180d",
    "order_freq_trend",              # ratio: momentum of ordering
    # --- monetary by window ----------------------------------------------
    "revenue_30d",
    "revenue_90d",
    "revenue_180d",
    "lifetime_revenue",
    "aov_90d",                       # ratio: revenue / orders
    "revenue_trend",                 # ratio: momentum of spend
    # --- basket composition ----------------------------------------------
    "distinct_products_90d",
    "avg_items_per_order_90d",
    "avg_unit_price_90d",
    "product_breadth_ratio",         # ratio: distinct products / items
    # --- return behaviour -------------------------------------------------
    "n_returns_180d",
    "return_value_ratio_180d",       # ratio: |returns| / gross sales
]

CATEGORICAL_FEATURES: list[str] = [
    "country_grp",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features whose value depends on *when* you ask, not only on the event
# history. These must be recomputed at scoring time from the request
# timestamp; serving a stale cached copy is the classic online skew bug.
TIME_DEPENDENT_FEATURES: list[str] = [
    "recency_days",
    "tenure_days",
    "active_days_ratio",
    "recency_over_avg_gap",
]

# Plausible domain ranges, used by the data-quality check in monitoring.py.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "recency_days": (0, 1000),
    "tenure_days": (0, 1000),
    "active_days_ratio": (0, 1),
    "avg_interpurchase_gap_days": (0, 1000),
    "interpurchase_gap_cv": (0, 20),
    "recency_over_avg_gap": (0, 200),
    "orders_30d": (0, 200),
    "orders_90d": (0, 400),
    "orders_180d": (0, 800),
    "order_freq_trend": (0, 50),
    "revenue_30d": (0, 1e6),
    "revenue_90d": (0, 1e6),
    "revenue_180d": (0, 2e6),
    "lifetime_revenue": (0, 5e6),
    "aov_90d": (0, 1e6),
    "revenue_trend": (0, 50),
    "distinct_products_90d": (0, 5000),
    "avg_items_per_order_90d": (0, 1e5),
    "avg_unit_price_90d": (0, 5000),
    "product_breadth_ratio": (0, 1),
    "n_returns_180d": (0, 500),
    "return_value_ratio_180d": (0, 50),
}

_EPS = 1e-9


# --------------------------------------------------------------------------
# Core feature computation
# --------------------------------------------------------------------------
def compute_features(
    events: pd.DataFrame,
    asof: pd.Timestamp | str,
    customer_ids: Iterable[str] | None = None,
    windows: Sequence[int] = (30, 90, 180),
    top_n_countries: int = 6,
    country_whitelist: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the point-in-time feature vector for every customer.

    Parameters
    ----------
    events
        Long event table with columns ``customer_id``, ``event_date``,
        ``invoice_no``, ``stock_code``, ``quantity``, ``unit_price``,
        ``line_revenue``, ``is_return``, ``country``.
    asof
        The observation instant. **No event after this date is used.**
    customer_ids
        Restrict output to these customers. If ``None`` every customer with at
        least one purchase on or before ``asof`` is returned.
    windows
        Look-back windows in days for the aggregation features.
    top_n_countries
        Countries outside the top-N (by row count in the training reference)
        are collapsed into ``"Other"`` to bound one-hot cardinality.
    country_whitelist
        Explicit list of countries to keep as their own level. Passing the list
        learned at training time is what keeps the encoding stable at serving
        time; if it is ``None`` the list is derived from ``events``.

    Returns
    -------
    DataFrame indexed 0..n-1 with columns
    ``[customer_id, snapshot_date] + FEATURE_COLUMNS``.
    """
    asof = pd.Timestamp(asof).normalize()

    # ---- (1) point-in-time filter: applied ONCE, for every feature ------
    hist = events.loc[events["event_date"] <= asof].copy()

    # Purchases and returns are two different behavioural signals.
    sales = hist.loc[~hist["is_return"]]
    returns = hist.loc[hist["is_return"]]

    if customer_ids is None:
        index = pd.Index(sales[ENTITY_COL].unique(), name=ENTITY_COL)
    else:
        index = pd.Index(pd.unique(pd.Series(list(customer_ids))), name=ENTITY_COL)

    out = pd.DataFrame(index=index)

    # ---- (2) recency / tenure -------------------------------------------
    first_last = sales.groupby(ENTITY_COL)["event_date"].agg(["min", "max"])
    out["recency_days"] = (asof - first_last["max"]).dt.days
    out["tenure_days"] = (asof - first_last["min"]).dt.days
    # Fraction of the observed lifetime that is *not* the current dormancy.
    out["active_days_ratio"] = (
        (out["tenure_days"] - out["recency_days"]) / (out["tenure_days"] + 1.0)
    ).clip(0, 1)

    # ---- (3) inter-purchase rhythm --------------------------------------
    # One row per (customer, purchase day) so multi-line invoices count once.
    order_days = (
        sales.loc[:, [ENTITY_COL, "event_date"]]
        .drop_duplicates()
        .sort_values([ENTITY_COL, "event_date"])
    )
    gaps = order_days.groupby(ENTITY_COL)["event_date"].diff().dt.days
    gap_stats = (
        gaps.groupby(order_days[ENTITY_COL]).agg(["mean", "std"]).rename(
            columns={"mean": "gap_mean", "std": "gap_std"}
        )
    )
    out["avg_interpurchase_gap_days"] = gap_stats["gap_mean"]
    # Coefficient of variation: is this customer clockwork-regular or erratic?
    out["interpurchase_gap_cv"] = gap_stats["gap_std"] / (gap_stats["gap_mean"] + _EPS)
    # How overdue is the customer, in units of their own normal cadence?
    # A value of 1 means "exactly on schedule", 5 means "five cycles late".
    out["recency_over_avg_gap"] = out["recency_days"] / (
        out["avg_interpurchase_gap_days"] + _EPS
    )

    # ---- (4) windowed frequency / monetary aggregates -------------------
    for w in windows:
        lo = asof - pd.Timedelta(days=w)
        win = sales.loc[sales["event_date"] > lo]
        grp = win.groupby(ENTITY_COL)
        out[f"orders_{w}d"] = grp["invoice_no"].nunique()
        out[f"revenue_{w}d"] = grp["line_revenue"].sum()

    out["lifetime_revenue"] = sales.groupby(ENTITY_COL)["line_revenue"].sum()

    # ---- (5) basket composition over the 90-day window ------------------
    lo90 = asof - pd.Timedelta(days=90)
    win90 = sales.loc[sales["event_date"] > lo90]
    g90 = win90.groupby(ENTITY_COL)
    out["distinct_products_90d"] = g90["stock_code"].nunique()
    items_90d = g90["quantity"].sum()
    out["avg_items_per_order_90d"] = items_90d / (out["orders_90d"] + _EPS)
    out["avg_unit_price_90d"] = g90["unit_price"].mean()
    # Do they buy many of a few things (wholesaler) or few of many (retail)?
    out["product_breadth_ratio"] = out["distinct_products_90d"] / (items_90d + _EPS)

    # ---- (6) return behaviour over 180 days -----------------------------
    lo180 = asof - pd.Timedelta(days=180)
    ret180 = returns.loc[returns["event_date"] > lo180]
    out["n_returns_180d"] = ret180.groupby(ENTITY_COL)["invoice_no"].nunique()
    ret_value = ret180.groupby(ENTITY_COL)["line_revenue"].sum().abs()
    out["return_value_ratio_180d"] = ret_value / (out["revenue_180d"] + 1.0)

    # ---- (7) ratio / trend features -------------------------------------
    # Recent 30 days versus the 90-day average month. < 1 means slowing down,
    # which is exactly the pre-churn behaviour we want the model to see.
    out["order_freq_trend"] = out["orders_30d"] / ((out["orders_90d"] / 3.0) + _EPS)
    out["revenue_trend"] = out["revenue_30d"] / ((out["revenue_90d"] / 3.0) + _EPS)
    out["aov_90d"] = out["revenue_90d"] / (out["orders_90d"] + _EPS)

    # ---- (8) categorical: country, cardinality-bounded ------------------
    # Most recent known country wins (a customer can ship to more than one).
    last_country = (
        sales.sort_values("event_date").groupby(ENTITY_COL)["country"].last()
    )
    if country_whitelist is None:
        country_whitelist = (
            sales["country"].value_counts().head(top_n_countries).index.tolist()
        )
    country = last_country.reindex(index)
    out["country_grp"] = (
        country.where(country.isin(list(country_whitelist)), "Other")
        .fillna("Unknown")
        .astype(str)
    )

    # ---- (9) fill and finalise ------------------------------------------
    # A customer with a single purchase has no inter-purchase gap. Encoding
    # "undefined" as a large sentinel instead of NaN keeps the semantic
    # ordering (one-time buyers behave like very overdue buyers) and keeps the
    # online path free of imputation logic that could differ from offline.
    out["avg_interpurchase_gap_days"] = out["avg_interpurchase_gap_days"].fillna(999.0)
    out["interpurchase_gap_cv"] = out["interpurchase_gap_cv"].fillna(0.0)
    out["recency_over_avg_gap"] = (
        out["recency_days"] / (out["avg_interpurchase_gap_days"] + _EPS)
    )

    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        # Guard against inf produced by a zero denominator slipping through.
        out[col] = out[col].replace([np.inf, -np.inf], 0.0).astype("float64")
        # Winsorise to the declared domain. Unbounded ratios (a customer whose
        # own cadence is 1 day and who has been dormant for 372 days scores
        # recency_over_avg_gap = 372) otherwise produce extreme values that add
        # no information beyond "hopelessly overdue" while dominating a linear
        # model's coefficients and violating the API's own schema bounds.
        # Applying it HERE, in the shared module, is what keeps the offline
        # table, the online lookup and the Pydantic contract mutually consistent.
        if col in FEATURE_RANGES:
            lo, hi = FEATURE_RANGES[col]
            out[col] = out[col].clip(lo, hi)

    out = out.reset_index()
    out.insert(1, ASOF_COL, asof)
    return out.loc[:, [ENTITY_COL, ASOF_COL] + FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# Contract helpers - used by training, serving and tests alike
# --------------------------------------------------------------------------
def align_features(frame: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Re-index ``frame`` onto ``FEATURE_COLUMNS`` in the canonical order.

    ``strict=True`` raises on a missing column, which is what we want on the
    serving path: a malformed request must fail loudly, not be silently
    imputed into a confident but meaningless prediction.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing and strict:
        raise ValueError(f"Missing required feature columns: {missing}")
    for col in missing:
        frame[col] = 0.0 if col in NUMERIC_FEATURES else "Unknown"
    return frame.loc[:, FEATURE_COLUMNS]


def refresh_time_dependent_features(
    row: dict, scoring_time: pd.Timestamp | str
) -> dict:
    """Re-age a cached feature row to ``scoring_time``.

    The online feature table is materialised by a nightly batch job. If a
    request arrives 14 hours later, ``recency_days`` in the cache is 14 hours
    stale. Training never has that staleness, so serving the cached value
    directly produces a systematic offline/online gap on the single most
    important feature. This function closes it using the cached
    ``feature_timestamp``.

    PRECONDITION -- this is important and easy to get wrong
    -------------------------------------------------------
    Re-aging assumes **no new events arrived for this customer** between
    ``feature_timestamp`` and ``scoring_time``. Ageing is only additive on a
    quiet history: if the customer purchased in the interval, their true
    ``recency_days`` *resets toward zero* while this function would keep adding
    to it, producing an error equal to the days since that purchase.

    That precondition is satisfied in the deployed design because the nightly
    materialisation job rewrites the row of every customer who had activity, so
    any customer with new events carries a fresh ``feature_timestamp`` and the
    re-aging is a no-op for them. The function is therefore correct for exactly
    the population it matters for -- dormant customers, who are the churn risks.

    The residual exposure is intra-day: a customer who purchases at 10:00 and is
    scored at 15:00 is served a row that still reflects the overnight state. The
    error is bounded by one refresh interval and is *conservative* for this use
    case (it over-states churn risk for a customer who just bought, so the cost
    is a wasted GBP 8 contact rather than a missed churner). Closing it properly
    requires event-driven cache invalidation -- publishing a
    ``last_purchase_date`` update to the online store on each order -- which is
    listed as future work rather than implemented here.
    """
    row = dict(row)
    cached_at = pd.Timestamp(row.get("feature_timestamp") or scoring_time).normalize()
    now = pd.Timestamp(scoring_time).normalize()
    age_days = max((now - cached_at).days, 0)
    if age_days == 0:
        return row

    # The same clipping the offline path applies, so a re-aged row cannot drift
    # outside the contract that training and the API schema both assume.
    def _clip(name: str, value: float) -> float:
        lo, hi = FEATURE_RANGES.get(name, (-np.inf, np.inf))
        return float(np.clip(value, lo, hi))

    row["recency_days"] = _clip("recency_days",
                                float(row.get("recency_days", 0.0)) + age_days)
    row["tenure_days"] = _clip("tenure_days",
                               float(row.get("tenure_days", 0.0)) + age_days)
    tenure = row["tenure_days"]
    row["active_days_ratio"] = _clip(
        "active_days_ratio", (tenure - row["recency_days"]) / (tenure + 1.0))
    row["recency_over_avg_gap"] = _clip(
        "recency_over_avg_gap",
        row["recency_days"] / (float(row.get("avg_interpurchase_gap_days", 999.0)) + _EPS))
    row["_feature_staleness_days"] = age_days
    return row


def feature_summary() -> pd.DataFrame:
    """Human-readable feature catalogue (used in the notebook and the report)."""
    catalogue = [
        ("recency_days", "recency", "Days since the customer's last purchase", "aggregation", "online (re-aged)"),
        ("tenure_days", "recency", "Days since the customer's first purchase", "aggregation", "online (re-aged)"),
        ("active_days_ratio", "recency", "Share of observed lifetime not spent dormant", "ratio", "online (re-aged)"),
        ("avg_interpurchase_gap_days", "rhythm", "Mean days between consecutive purchase days", "aggregation", "offline"),
        ("interpurchase_gap_cv", "rhythm", "Std/mean of purchase gaps - regularity", "aggregation", "offline"),
        ("recency_over_avg_gap", "rhythm", "How overdue the customer is in units of own cadence", "ratio", "online (re-aged)"),
        ("orders_30d", "frequency", "Distinct invoices in the last 30 days", "time-window agg", "offline"),
        ("orders_90d", "frequency", "Distinct invoices in the last 90 days", "time-window agg", "offline"),
        ("orders_180d", "frequency", "Distinct invoices in the last 180 days", "time-window agg", "offline"),
        ("order_freq_trend", "frequency", "orders_30d / (orders_90d/3) - ordering momentum", "ratio/trend", "offline"),
        ("revenue_30d", "monetary", "Gross revenue in the last 30 days", "time-window agg", "offline"),
        ("revenue_90d", "monetary", "Gross revenue in the last 90 days", "time-window agg", "offline"),
        ("revenue_180d", "monetary", "Gross revenue in the last 180 days", "time-window agg", "offline"),
        ("lifetime_revenue", "monetary", "Gross revenue over all history to date", "aggregation", "offline"),
        ("aov_90d", "monetary", "revenue_90d / orders_90d - average order value", "ratio", "offline"),
        ("revenue_trend", "monetary", "revenue_30d / (revenue_90d/3) - spend momentum", "ratio/trend", "offline"),
        ("distinct_products_90d", "basket", "Distinct stock codes bought in 90 days", "time-window agg", "offline"),
        ("avg_items_per_order_90d", "basket", "Units per invoice in 90 days", "ratio", "offline"),
        ("avg_unit_price_90d", "basket", "Mean unit price paid in 90 days", "aggregation", "offline"),
        ("product_breadth_ratio", "basket", "Distinct products / total units - retail vs wholesale", "ratio", "offline"),
        ("n_returns_180d", "returns", "Distinct cancellation invoices in 180 days", "time-window agg", "offline"),
        ("return_value_ratio_180d", "returns", "|Return value| / revenue_180d", "ratio", "offline"),
        ("country_grp", "geo", "Country, top-6 kept and the rest collapsed", "encoding", "offline"),
    ]
    return pd.DataFrame(
        catalogue, columns=["feature", "family", "definition", "type", "availability"]
    )
