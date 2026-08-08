"""Raw -> canonical event table.

The public CSV is a messy transactional extract. This module turns it into the
canonical, append-only *event table* that every downstream stage consumes. It
is deliberately separate from feature engineering: cleaning decisions are
recorded once, auditably, and never re-litigated inside a feature function.

Every cleaning rule below is driven by an observed defect in the source data,
and every rule reports how many rows it removed so the loss is visible rather
than implicit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CFG, REPO_ROOT, Config

CANONICAL_COLUMNS = [
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "unit_price",
    "line_revenue",
    "event_ts",
    "event_date",
    "customer_id",
    "country",
    "is_return",
]

SOURCE_RENAME = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "event_ts",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the source CSV with explicit dtypes (never let pandas guess IDs)."""
    csv_path = Path(path) if path else REPO_ROOT / CFG["paths"]["raw_csv"]
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",  # the file carries a UTF-8 BOM
        dtype={"InvoiceNo": "str", "StockCode": "str", "CustomerID": "str"},
    )
    return df.rename(columns=SOURCE_RENAME)


def clean_events(df: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the documented cleaning rules.

    Returns ``(events, audit)`` where ``audit`` is a per-rule row-loss report.
    """
    cfg = cfg or CFG
    rules = cfg["cleaning"]
    audit: list[dict] = []
    n0 = len(df)

    def record(rule: str, before: int, after: int, rationale: str) -> None:
        audit.append(
            {
                "rule": rule,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": before - after,
                "pct_of_raw_removed": round(100 * (before - after) / n0, 3),
                "rationale": rationale,
            }
        )

    df = df.copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"], format="%m/%d/%Y %H:%M")
    df["event_date"] = df["event_ts"].dt.normalize()

    # Rule 1 - a churn label is defined per customer, so an unattributable
    # transaction cannot be used. ~25% of rows; large but unavoidable. These
    # rows are still real revenue, so they are excluded from *modelling* only,
    # not from any business reporting.
    if rules["drop_missing_customer"]:
        before = len(df)
        df = df.loc[df["customer_id"].notna()]
        record("drop_missing_customer_id", before, len(df),
               "Churn is defined per customer; rows with no CustomerID cannot be labelled.")

    # Rule 2 - drop administrative stock codes (postage, samples, bank fees).
    # They are not products, so counting them as basket breadth is wrong.
    before = len(df)
    non_prod = {c.upper() for c in rules["non_product_stock_codes"]}
    df = df.loc[~df["stock_code"].str.upper().isin(non_prod)]
    record("drop_non_product_stock_codes", before, len(df),
           "POST/DOT/M/BANK CHARGES etc. are adjustments, not purchases.")

    # Rule 3 - flag cancellations. NOT dropped: a customer who returns goods is
    # behaviourally distinct and this becomes the return-rate feature family.
    df["is_return"] = (
        df["invoice_no"].str.upper().str.startswith("C") | (df["quantity"] < 0)
    )

    # Rule 4 - a genuine sale must have a positive price. Zero-price lines are
    # promotional/data-entry noise and would corrupt AOV and revenue features.
    before = len(df)
    bad_price = (~df["is_return"]) & (df["unit_price"] <= rules["min_unit_price"])
    df = df.loc[~bad_price]
    record("drop_non_positive_price_sales", before, len(df),
           "A sale with price <= 0 is promotional noise; it would distort AOV/revenue.")

    # Rule 5 - exact duplicate lines are re-transmission artefacts.
    before = len(df)
    df = df.drop_duplicates(
        subset=["invoice_no", "stock_code", "quantity", "unit_price", "event_ts"]
    )
    record("drop_exact_duplicate_lines", before, len(df),
           "Identical (invoice, product, qty, price, timestamp) rows are re-sends.")

    df["line_revenue"] = df["quantity"] * df["unit_price"]
    df["customer_id"] = df["customer_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["country"] = df["country"].fillna("Unknown").astype(str)
    df["description"] = df["description"].fillna("UNKNOWN").astype(str)

    events = (
        df.loc[:, CANONICAL_COLUMNS]
        .sort_values("event_ts")
        .reset_index(drop=True)
    )
    return events, pd.DataFrame(audit)


def build_event_table(
    cfg: Config | None = None, write: bool = True, until: str | None = None
) -> pd.DataFrame:
    """End-to-end raw -> canonical events, optionally persisted to parquet.

    ``until`` truncates history at a date. This exists so the ingestion demo is
    realistic: the warehouse is backfilled to a cutoff, and the daily files then
    arrive as genuinely new data that must be appended, exactly as a scheduled
    job would experience it.
    """
    cfg = cfg or CFG
    raw = load_raw()
    events, audit = clean_events(raw, cfg)
    if until is not None:
        cutoff = pd.Timestamp(until).normalize()
        n_before = len(events)
        events = events.loc[events["event_date"] <= cutoff].reset_index(drop=True)
        print(f"[data_prep] truncated warehouse at {cutoff.date()}: "
              f"{len(events):,} of {n_before:,} events retained "
              f"({n_before - len(events):,} held back for the ingestion demo)")
    if write:
        out = cfg.get_path("paths.warehouse_events")
        events.to_parquet(out, index=False)
        audit_path = REPO_ROOT / cfg["paths"]["monitoring_dir"] / "cleaning_audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit.to_dict("records"), indent=2))
    return events


def load_event_table(cfg: Config | None = None) -> pd.DataFrame:
    """Read the canonical event table, building it on first use."""
    cfg = cfg or CFG
    path = cfg.get_path("paths.warehouse_events")
    if not path.exists():
        return build_event_table(cfg, write=True)
    df = pd.read_parquet(path)
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Build the canonical event table")
    ap.add_argument("--until", help="truncate history at this date (YYYY-MM-DD)")
    args = ap.parse_args()
    ev = build_event_table(until=args.until)
    print(f"events={len(ev):,} customers={ev['customer_id'].nunique():,} "
          f"range={ev['event_date'].min().date()}..{ev['event_date'].max().date()}")
