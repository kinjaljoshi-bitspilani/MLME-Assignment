"""Split the historical CSV into daily files to simulate a live upstream feed.

The public dataset is a single static export. A production ingestion job,
however, receives one file per day. This script replays history: it partitions
the source CSV by ``InvoiceDate`` so that ``src/ingest.py`` can be exercised
exactly as it would be on a scheduler, and so that the "recent batch" used by
the drift check is a genuine slice of real data rather than a fabrication.

Only the *arrival* of the data is simulated; every value in every file is real.

Usage
-----
    python -m scripts.make_daily_batches --start 2011-11-25 --end 2011-12-09
    python -m scripts.make_daily_batches --last-n-days 14
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.config import CFG, REPO_ROOT
from src.data_prep import SOURCE_RENAME

SOURCE_COLUMNS = list(SOURCE_RENAME.keys())


def make_daily_batches(
    start: str | None = None,
    end: str | None = None,
    last_n_days: int | None = None,
    clean_dir: bool = True,
) -> list[Path]:
    raw_path = REPO_ROOT / CFG["paths"]["raw_csv"]
    out_dir = REPO_ROOT / CFG["paths"]["incoming_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean_dir:
        for old in out_dir.glob("transactions_*.csv"):
            old.unlink()

    df = pd.read_csv(
        raw_path,
        encoding="utf-8-sig",
        dtype={"InvoiceNo": "str", "StockCode": "str", "CustomerID": "str"},
    )
    ts = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    day = ts.dt.normalize()

    if last_n_days is not None:
        end_ts = day.max()
        start_ts = end_ts - pd.Timedelta(days=last_n_days - 1)
    else:
        start_ts = pd.Timestamp(start) if start else day.min()
        end_ts = pd.Timestamp(end) if end else day.max()

    sel = df.loc[(day >= start_ts) & (day <= end_ts)].copy()
    sel_day = day.loc[sel.index]

    written: list[Path] = []
    for d, chunk in sel.groupby(sel_day):
        path = out_dir / f"transactions_{pd.Timestamp(d).date()}.csv"
        chunk.loc[:, SOURCE_COLUMNS].to_csv(path, index=False)
        written.append(path)

    print(f"[batches] wrote {len(written)} daily files to {out_dir} "
          f"({start_ts.date()}..{end_ts.date()}, {len(sel):,} rows)")
    return written


def make_corrupt_batch(reference_day: str, mode: str = "missing_column") -> Path:
    """Produce a deliberately broken file to demonstrate ingestion defences.

    ``mode='missing_column'`` simulates the upstream team renaming/removing a
    column - the incident scenario in the design document.
    """
    out_dir = REPO_ROOT / CFG["paths"]["incoming_dir"]
    src = out_dir / f"transactions_{reference_day}.csv"
    dst = out_dir.parent / f"corrupt_{mode}_{reference_day}.csv"
    df = pd.read_csv(src, dtype=str)
    if mode == "missing_column":
        df = df.drop(columns=["UnitPrice"])
    elif mode == "renamed_column":
        df = df.rename(columns={"CustomerID": "customer_key"})
    df.to_csv(dst, index=False)
    print(f"[batches] wrote corrupt sample ({mode}) -> {dst}")
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--last-n-days", type=int)
    ap.add_argument("--keep-existing", action="store_true")
    args = ap.parse_args(argv)
    make_daily_batches(args.start, args.end, args.last_n_days,
                       clean_dir=not args.keep_existing)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
