"""Label definition and construction of the supervised training table.

Churn on a non-contractual retail business has no cancellation event, so the
label must be *defined*. The definition here is a standard fixed-horizon,
activity-scoped one:

    For a customer who purchased at least once in the ``activity_window_days``
    before an observation date ``t``, the label is 1 if they make **no**
    purchase in the interval ``(t, t + horizon_days]``.

Three design consequences are worth stating explicitly because they drive the
rest of the system:

1. **Scoping matters.** Without the activity window, every customer who ever
   bought once in 2010 is a permanent "churner" and the prevalence becomes
   meaningless. Scoping to recent actives makes the population the one the
   retention team would actually target.
2. **The horizon must be fully observed.** A snapshot is only usable if
   ``t + horizon <= system_asof``; otherwise a customer looks like a churner
   purely because their future has not happened yet (right-censoring).
3. **The same customer appears at several snapshots.** A random train/test
   split therefore leaks the *entity*. The split is strictly temporal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG, Config
from .features import ASOF_COL, ENTITY_COL, LABEL_COL, compute_features


def customers_in_scope(
    events: pd.DataFrame, asof: pd.Timestamp, activity_window_days: int
) -> np.ndarray:
    """Customers with >= 1 purchase in the activity window ending at ``asof``."""
    lo = asof - pd.Timedelta(days=activity_window_days)
    mask = (
        (~events["is_return"])
        & (events["event_date"] <= asof)
        & (events["event_date"] > lo)
    )
    return events.loc[mask, ENTITY_COL].unique()


def make_labels(
    events: pd.DataFrame,
    asof: pd.Timestamp,
    customer_ids: np.ndarray,
    horizon_days: int,
) -> pd.Series:
    """1 = no purchase in ``(asof, asof + horizon_days]``."""
    hi = asof + pd.Timedelta(days=horizon_days)
    future = events.loc[
        (~events["is_return"])
        & (events["event_date"] > asof)
        & (events["event_date"] <= hi),
        ENTITY_COL,
    ].unique()
    return pd.Series(
        np.isin(customer_ids, future, invert=True).astype("int8"),
        index=pd.Index(customer_ids, name=ENTITY_COL),
        name=LABEL_COL,
    )


def build_snapshot(
    events: pd.DataFrame,
    asof: pd.Timestamp | str,
    cfg: Config | None = None,
    country_whitelist: list[str] | None = None,
    with_label: bool = True,
) -> pd.DataFrame:
    """One observation date -> features (+ label) for every in-scope customer."""
    cfg = cfg or CFG
    asof = pd.Timestamp(asof).normalize()
    lab_cfg = cfg["label"]

    ids = customers_in_scope(events, asof, lab_cfg["activity_window_days"])
    feats = compute_features(
        events,
        asof=asof,
        customer_ids=ids,
        windows=tuple(cfg["features"]["windows_days"]),
        top_n_countries=cfg["features"]["top_n_countries"],
        country_whitelist=country_whitelist,
    )
    if with_label:
        y = make_labels(events, asof, ids, lab_cfg["horizon_days"])
        feats = feats.merge(y.reset_index(), on=ENTITY_COL, how="left")
    return feats


def build_training_table(
    events: pd.DataFrame, cfg: Config | None = None, write: bool = True
) -> pd.DataFrame:
    """Stack every configured snapshot into the supervised training table.

    The country whitelist is learned **once**, from data available at the first
    (earliest) snapshot, and then reused for every later snapshot and at
    serving time. Re-deriving it per snapshot would make the encoding drift.
    """
    cfg = cfg or CFG
    dates = [pd.Timestamp(d) for d in cfg["label"]["snapshot_dates"]]
    system_asof = pd.Timestamp(cfg["project"]["system_asof"])
    horizon = cfg["label"]["horizon_days"]

    usable, censored = [], []
    for d in dates:
        (usable if d + pd.Timedelta(days=horizon) <= system_asof else censored).append(d)
    if censored:
        print(f"[labels] skipping right-censored snapshots: "
              f"{[str(d.date()) for d in censored]}")

    first = min(usable)
    hist = events.loc[events["event_date"] <= first]
    whitelist = (
        hist.loc[~hist["is_return"], "country"]
        .value_counts()
        .head(cfg["features"]["top_n_countries"])
        .index.tolist()
    )

    frames = []
    for d in usable:
        snap = build_snapshot(events, d, cfg, country_whitelist=whitelist)
        print(f"[labels] {d.date()}  n={len(snap):>5,}  churn_rate={snap[LABEL_COL].mean():.4f}")
        frames.append(snap)

    table = pd.concat(frames, ignore_index=True)
    table.attrs["country_whitelist"] = whitelist
    if write:
        out = cfg.get_path("paths.warehouse_snapshots")
        table.to_parquet(out, index=False)
    return table


def temporal_split(
    table: pd.DataFrame, cfg: Config | None = None
) -> dict[str, pd.DataFrame]:
    """Split by observation date, never at random.

    Rationale, in the order the leakage would bite:

    * **Entity leakage.** The same ``customer_id`` occurs at up to five
      snapshots. Random splitting puts near-duplicate rows for one customer on
      both sides and inflates every metric.
    * **Temporal leakage.** Features at a later snapshot are computed from
      event history that overlaps earlier snapshots. Only ordering by time
      reproduces the production situation of predicting a *future* period.
    * **Label-window bleed.** The training snapshots' label windows extend
      forward in time. ``2011-07-31`` is therefore dropped entirely as an
      embargo (purge) month so that the last training label window
      (ends 2011-08-29) does not overlap the test feature window (ends
      2011-08-31).
    """
    cfg = cfg or CFG
    sp = cfg["split"]
    d = pd.to_datetime(table[ASOF_COL])

    def take(keys: list[str]) -> pd.DataFrame:
        return table.loc[d.isin(pd.to_datetime(keys))].reset_index(drop=True)

    return {
        "train": take(sp["train_snapshots"]),
        "valid": take(sp["valid_snapshots"]),
        "test": take(sp["test_snapshots"]),
    }


def split_report(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row counts, date coverage and prevalence per fold."""
    rows = []
    for name, part in splits.items():
        rows.append(
            {
                "fold": name,
                "snapshots": ", ".join(
                    sorted(pd.to_datetime(part[ASOF_COL]).dt.date.astype(str).unique())
                ),
                "n_rows": len(part),
                "n_customers": part[ENTITY_COL].nunique(),
                "churn_rate": round(float(part[LABEL_COL].mean()), 4),
            }
        )
    return pd.DataFrame(rows)
