"""Micro-batch ingestion: daily CSV drop -> append-only event warehouse.

Contract
--------
* Input  : ``data/incoming/transactions_YYYY-MM-DD.csv`` (raw source schema).
* Output : rows merged into ``data/warehouse/events.parquet``.
* Side effect : one JSON line per file appended to
  ``artifacts/logs/ingestion_log.jsonl``.

The three properties that make this safe to run on a scheduler:

1. **Schema validation before anything else.** A missing or renamed upstream
   column aborts the batch with a non-zero exit code instead of writing a
   half-broken partition. This is the exact failure mode covered by the
   incident scenario in the design document.
2. **Idempotency.** The merge de-duplicates on the natural key
   ``(invoice_no, stock_code, event_ts, quantity, unit_price)``, so re-running
   yesterday's file is a no-op rather than a double-count. Schedulers retry;
   pipelines must tolerate it.
3. **Observability.** Every run records rows read, rows rejected by each
   quality rule, rows actually appended, and the resulting warehouse size.
   "How many rows did we ingest and when" is answerable from the log alone.

Usage
-----
    python -m src.ingest --date 2011-12-01
    python -m src.ingest --all
    python -m src.ingest --date 2011-12-01 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CFG, REPO_ROOT, Config
from .data_prep import CANONICAL_COLUMNS, SOURCE_RENAME, clean_events

REQUIRED_SOURCE_COLUMNS = list(SOURCE_RENAME.keys())
NATURAL_KEY = ["invoice_no", "stock_code", "event_ts", "quantity", "unit_price"]


class SchemaError(RuntimeError):
    """Raised when an incoming file does not match the agreed source schema."""


# --------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast and loudly on an upstream contract break."""
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in REQUIRED_SOURCE_COLUMNS]
    if missing:
        raise SchemaError(
            f"Incoming batch is missing required column(s) {missing}. "
            f"Received columns: {list(df.columns)}. Batch rejected."
        )
    if unexpected:
        # Additive change: warn, do not fail. Extra columns are dropped.
        print(f"[ingest] WARNING new upstream column(s) ignored: {unexpected}")


def read_batch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"InvoiceNo": "str", "StockCode": "str", "CustomerID": "str"},
    )
    validate_schema(df)
    return df.rename(columns=SOURCE_RENAME)


def merge_into_warehouse(
    new_events: pd.DataFrame, warehouse_path: Path
) -> tuple[int, int]:
    """Append with de-duplication. Returns ``(n_appended, n_total)``."""
    if warehouse_path.exists():
        existing = pd.read_parquet(warehouse_path)
        existing["event_ts"] = pd.to_datetime(existing["event_ts"])
        existing["event_date"] = pd.to_datetime(existing["event_date"])
        before = len(existing)
        combined = pd.concat([existing, new_events], ignore_index=True)
    else:
        before = 0
        combined = new_events.copy()

    combined = (
        combined.drop_duplicates(subset=NATURAL_KEY, keep="first")
        .sort_values("event_ts")
        .reset_index(drop=True)
    )
    combined.to_parquet(warehouse_path, index=False)
    return len(combined) - before, len(combined)


def log_run(record: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------
def ingest_file(
    path: Path, cfg: Config | None = None, dry_run: bool = False
) -> dict:
    """Ingest a single daily file and return the run record."""
    cfg = cfg or CFG
    started = datetime.now(timezone.utc)
    warehouse = cfg.get_path("paths.warehouse_events")

    record: dict = {
        "run_id": started.strftime("%Y%m%dT%H%M%S%f"),
        "ingested_at_utc": started.isoformat(),
        "source_file": path.name,
        "status": "ok",
        "dry_run": dry_run,
    }

    try:
        raw = read_batch(path)
    except SchemaError as exc:
        record.update(status="rejected_schema", error=str(exc), rows_read=0,
                      rows_appended=0)
        log_run(record, cfg.get_path("paths.ingestion_log"))
        raise

    record["rows_read"] = len(raw)
    events, audit = clean_events(raw, cfg)
    record["rows_after_cleaning"] = len(events)
    record["rows_rejected_by_rule"] = {
        r["rule"]: int(r["rows_removed"]) for _, r in audit.iterrows()
    }

    if len(events):
        record["event_date_min"] = str(events["event_date"].min().date())
        record["event_date_max"] = str(events["event_date"].max().date())
        record["n_customers"] = int(events["customer_id"].nunique())
        record["n_invoices"] = int(events["invoice_no"].nunique())
        record["gross_revenue"] = round(float(events["line_revenue"].sum()), 2)
    else:
        record.update(event_date_min=None, event_date_max=None, n_customers=0,
                      n_invoices=0, gross_revenue=0.0)

    if dry_run:
        record.update(rows_appended=0, warehouse_rows=None, status="dry_run")
    else:
        appended, total = merge_into_warehouse(
            events.loc[:, CANONICAL_COLUMNS], warehouse
        )
        record["rows_appended"] = appended
        record["warehouse_rows"] = total
        record["rows_deduplicated"] = len(events) - appended

    record["duration_s"] = round(
        (datetime.now(timezone.utc) - started).total_seconds(), 3
    )
    log_run(record, cfg.get_path("paths.ingestion_log"))

    print(
        f"[ingest] {path.name}: read={record['rows_read']:,} "
        f"clean={record['rows_after_cleaning']:,} "
        f"appended={record.get('rows_appended', 0):,} "
        f"warehouse={record.get('warehouse_rows')} "
        f"dates={record['event_date_min']}..{record['event_date_max']} "
        f"({record['duration_s']}s)"
    )
    return record


def ingest_all(cfg: Config | None = None, pattern: str = "transactions_*.csv") -> list[dict]:
    """Ingest every pending file in the drop zone, oldest first."""
    cfg = cfg or CFG
    incoming = REPO_ROOT / cfg["paths"]["incoming_dir"]
    files = sorted(incoming.glob(pattern))
    if not files:
        print(f"[ingest] no files matching {pattern} in {incoming}")
        return []
    return [ingest_file(f, cfg) for f in files]


def ingestion_history(cfg: Config | None = None) -> pd.DataFrame:
    """The ingestion log as a DataFrame (used by the notebook and monitoring)."""
    cfg = cfg or CFG
    log = cfg.get_path("paths.ingestion_log")
    if not log.exists():
        return pd.DataFrame()
    with open(log, "r", encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(line) for line in fh if line.strip()])


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Micro-batch transaction ingestion")
    ap.add_argument("--date", help="ingest data/incoming/transactions_<date>.csv")
    ap.add_argument("--file", help="ingest an explicit path")
    ap.add_argument("--all", action="store_true", help="ingest every pending file")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report, write nothing")
    args = ap.parse_args(argv)
    cfg = CFG

    try:
        if args.all:
            ingest_all(cfg)
        else:
            if args.file:
                path = Path(args.file)
            elif args.date:
                path = (REPO_ROOT / cfg["paths"]["incoming_dir"]
                        / f"transactions_{args.date}.csv")
            else:
                ap.error("one of --date, --file or --all is required")
            if not path.exists():
                print(f"[ingest] ERROR file not found: {path}", file=sys.stderr)
                return 2
            ingest_file(path, cfg, dry_run=args.dry_run)
    except SchemaError as exc:
        print(f"[ingest] ABORTED {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
