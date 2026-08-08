"""Latency and throughput measurement for the inference service.

Measures three things that answer different questions:

* **Online single-request latency** (``/predict``) - what a user waiting on a
  CRM screen experiences. Reported as avg/p50/p95/p99, because a mean hides the
  tail and the tail is what breaches an SLO.
* **Batch throughput** (``/predict/batch``) - rows/sec for the nightly scoring
  job, which is a cost and completion-window question, not a latency one.
* **Concurrent load** - p95 under N parallel clients, to show how the tail
  degrades once requests queue.

A warm-up phase runs first and is discarded. The first request pays lazy
imports, model deserialisation and JIT-style caching costs; including it would
report a p99 that never occurs again in the life of the process.

Usage
-----
    # against a live server
    python -m scripts.load_test --url http://127.0.0.1:8000 --n 500
    # in-process (no network), useful in CI
    python -m scripts.load_test --in-process --n 500
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import CFG, REPO_ROOT
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# --------------------------------------------------------------------------
def sample_payloads(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Draw real customers from the feature store as request bodies.

    Synthetic random payloads would be unrepresentative: the model's runtime is
    data-independent but the *pipeline* branches on category levels, so using
    genuine rows keeps the measurement honest.
    """
    fs = CFG.get_path("paths.feature_store")
    if not fs.exists():
        raise FileNotFoundError(
            f"{fs} not found. Run `python -m src.train` first.")
    df = pd.read_parquet(fs)
    rows = df.sample(n=min(n, len(df)), replace=n > len(df),
                     random_state=seed).reset_index(drop=True)

    payloads = []
    for _, r in rows.iterrows():
        feats = {"customer_id": str(r["customer_id"])}
        for c in NUMERIC_FEATURES:
            feats[c] = float(r[c])
        for c in CATEGORICAL_FEATURES:
            feats[c] = str(r[c])
        payloads.append(feats)
    return payloads


def _summarise(latencies_ms: list[float], label: str, n_errors: int = 0) -> dict[str, Any]:
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "scenario": label,
        "n_requests": int(len(arr)),
        "n_errors": int(n_errors),
        "avg_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "stdev_ms": round(float(arr.std(ddof=1)), 3) if len(arr) > 1 else 0.0,
        "throughput_rps": round(1000.0 / float(arr.mean()), 1),
    }


# --------------------------------------------------------------------------
def measure_online(
    post: Callable[[str, dict], Any],
    payloads: list[dict[str, Any]],
    warmup: int = 20,
) -> dict[str, Any]:
    """Sequential single-row requests. Warm-up excluded."""
    for p in payloads[:warmup]:
        post("/predict", {"features": p})

    latencies, errors = [], 0
    for p in payloads:
        t0 = time.perf_counter()
        try:
            r = post("/predict", {"features": p})
            ok = getattr(r, "status_code", 200) == 200
        except Exception:
            ok = False
        latencies.append((time.perf_counter() - t0) * 1000.0)
        errors += 0 if ok else 1
    return _summarise(latencies, "online_sequential_predict", errors)


def measure_concurrent(
    post: Callable[[str, dict], Any],
    payloads: list[dict[str, Any]],
    workers: int = 8,
) -> dict[str, Any]:
    """Same requests, N in flight at once, to expose queueing in the tail."""
    def one(p: dict[str, Any]) -> float:
        t0 = time.perf_counter()
        post("/predict", {"features": p})
        return (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        latencies = list(ex.map(one, payloads))
    wall = time.perf_counter() - t0

    out = _summarise(latencies, f"online_concurrent_{workers}_workers")
    # Under concurrency the meaningful throughput is completions/wall-clock,
    # not 1/mean-latency, because requests overlap.
    out["throughput_rps"] = round(len(payloads) / wall, 1)
    out["wall_clock_s"] = round(wall, 3)
    out["workers"] = workers
    return out


def measure_batch(
    post: Callable[[str, dict], Any],
    payloads: list[dict[str, Any]],
    batch_sizes: tuple[int, ...] = (1, 10, 100, 500),
) -> list[dict[str, Any]]:
    """Throughput as a function of batch size - shows the amortisation curve."""
    results = []
    for size in batch_sizes:
        if size > len(payloads):
            continue
        items = payloads[:size]
        post("/predict/batch", {"items": items})  # warm
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            post("/predict/batch", {"items": items})
            runs.append((time.perf_counter() - t0) * 1000.0)
        total = stats.median(runs)
        results.append({
            "scenario": "batch_predict",
            "batch_size": size,
            "median_total_ms": round(total, 3),
            "ms_per_row": round(total / size, 4),
            "rows_per_second": round(size / (total / 1000.0), 1),
        })
    return results


# --------------------------------------------------------------------------
def build_poster(url: str | None, in_process: bool):
    """Return a ``post(path, json)`` callable for either transport."""
    if in_process:
        from fastapi.testclient import TestClient

        from src.serving.app import app

        client = TestClient(app)
        client.__enter__()  # trigger lifespan so the model loads
        return (lambda path, body: client.post(path, json=body)), client
    import httpx

    client = httpx.Client(base_url=url, timeout=30.0)
    return (lambda path, body: client.post(path, json=body)), client


def run(url: str | None = None, n: int = 300, workers: int = 8,
        in_process: bool = False, write: bool = True) -> dict[str, Any]:
    post, client = build_poster(url, in_process)
    try:
        payloads = sample_payloads(n)
        transport = "in_process_testclient" if in_process else f"http {url}"
        print(f"[load] transport={transport}  n={n}")

        online = measure_online(post, payloads)
        concurrent = measure_concurrent(post, payloads, workers=workers)
        batch = measure_batch(post, payloads)

        slo = float(CFG["serving"]["latency_slo_ms_p95"])
        report = {
            "measured_at": pd.Timestamp.now().isoformat(),
            "transport": transport,
            "n_payloads": len(payloads),
            "latency_slo_ms_p95": slo,
            "online": online,
            "concurrent": concurrent,
            "batch": batch,
            "slo_met_sequential": bool(online["p95_ms"] <= slo),
            "slo_met_concurrent": bool(concurrent["p95_ms"] <= slo),
        }
        if write:
            out = CFG.get_path("paths.eval_dir") / "latency_report.json"
            out.write_text(json.dumps(report, indent=2))
            print(f"[load] wrote {out}")

        print("\n--- online, sequential ---")
        for k in ["n_requests", "avg_ms", "p50_ms", "p95_ms", "p99_ms",
                  "max_ms", "throughput_rps", "n_errors"]:
            print(f"  {k:>16}: {online[k]}")
        print(f"\n--- online, {workers} concurrent ---")
        for k in ["avg_ms", "p95_ms", "p99_ms", "throughput_rps", "wall_clock_s"]:
            print(f"  {k:>16}: {concurrent[k]}")
        print("\n--- batch ---")
        print(pd.DataFrame(batch).to_string(index=False))
        print(f"\nSLO p95 <= {slo} ms: sequential="
              f"{'MET' if report['slo_met_sequential'] else 'BREACHED'}, "
              f"concurrent={'MET' if report['slo_met_concurrent'] else 'BREACHED'}")
        return report
    finally:
        try:
            client.__exit__(None, None, None)
        except Exception:
            client.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--in-process", action="store_true",
                    help="use FastAPI TestClient instead of HTTP")
    args = ap.parse_args(argv)
    run(args.url, args.n, args.workers, args.in_process)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
