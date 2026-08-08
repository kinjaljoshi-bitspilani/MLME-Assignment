"""FastAPI inference service.

Inference pattern: **hybrid** - nightly batch precomputation of the expensive
window aggregates into a feature table, plus an online request/response endpoint
that looks them up, re-ages the time-dependent ones and scores in-process.

Why hybrid rather than pure online or pure batch:

* Computing ``orders_180d`` from the raw event log at request time would mean
  scanning six months of transactions per call - hundreds of milliseconds, and
  a load pattern the transactional database should not be serving.
* Pure batch scoring cannot answer "score this customer *now*", which is what a
  live retention prompt in the CRM needs when an agent has the customer on the
  phone.
* Precomputing the stable part and re-aging only the time-dependent part gives
  single-digit millisecond responses while keeping the values equal to what
  training saw.

The service never hard-codes a model filename. It asks the registry for whatever
holds the ``production`` stage, so promotion and rollback are metadata changes.

Run:
    uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import CFG, REPO_ROOT
from ..features import (
    FEATURE_COLUMNS,
    align_features,
    refresh_time_dependent_features,
)
from ..registry import ModelRegistry
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    HealthResponse,
    MetadataResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

API_VERSION = "1.0.0"
PREDICTION_LOG = REPO_ROOT / "artifacts" / "logs" / "prediction_log.jsonl"


# --------------------------------------------------------------------------
class ServiceState:
    """Process-local state: model, feature table and rolling counters.

    In a real deployment the counters would be a Prometheus client and the
    feature table a Redis/DynamoDB lookup. The interface is identical; only the
    backing store changes, which is the point of keeping it behind this class.
    """

    def __init__(self) -> None:
        self.cfg = CFG
        self.model: Any = None
        self.entry: dict[str, Any] | None = None
        self.features: pd.DataFrame | None = None
        # Reference stats are cached in memory at startup. Reading the JSON
        # profile per request cost ~45 ms and would have dominated the latency
        # budget for a call that is otherwise sub-millisecond.
        self.reference_numeric: dict[str, Any] = {}
        self.started_at = time.time()

        self.requests_total = 0
        self.errors_total = 0
        self.predictions_total = 0
        self.lookup_misses = 0
        self.stale_servings = 0
        self.latencies_ms: deque[float] = deque(maxlen=5000)
        self.probabilities: deque[float] = deque(maxlen=5000)

    # ---------------------------------------------------------------- load
    def load(self) -> None:
        reg = ModelRegistry(self.cfg)
        stage = self.cfg["serving"]["model_stage"]
        try:
            self.model, self.entry = reg.load(stage=stage)
            print(f"[serve] loaded {self.entry['model_id']} (stage={stage})")
        except Exception as exc:  # model not trained yet
            print(f"[serve] WARNING no model in stage '{stage}': {exc}")
            self.model, self.entry = None, None

        fs = self.cfg.get_path("paths.feature_store")
        if fs.exists():
            df = pd.read_parquet(fs)
            self.features = df.set_index("customer_id", drop=False)
            print(f"[serve] feature store: {len(df):,} customers")
        else:
            self.features = None
            print("[serve] WARNING feature store missing; "
                  "/predict requires explicit features")

        try:
            from ..monitoring import load_reference_profile

            self.reference_numeric = load_reference_profile(self.cfg)["numeric"]
            print(f"[serve] cached reference stats for "
                  f"{len(self.reference_numeric)} features")
        except Exception as exc:
            self.reference_numeric = {}
            print(f"[serve] WARNING reference profile unavailable ({exc}); "
                  f"/predict?explain will return no attribution")

    def reload(self) -> dict[str, Any]:
        """Hot-swap the production model without restarting the process."""
        previous = self.entry["model_id"] if self.entry else None
        self.load()
        return {"previous_model_id": previous,
                "current_model_id": self.entry["model_id"] if self.entry else None}

    # -------------------------------------------------------------- helpers
    def percentile(self, q: float) -> float | None:
        return float(np.percentile(self.latencies_ms, q)) if self.latencies_ms else None

    def threshold(self, override: float | None = None) -> float:
        if override is not None:
            return float(override)
        if self.entry and self.entry.get("decision_threshold") is not None:
            return float(self.entry["decision_threshold"])
        return 0.5


STATE = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.load()
    yield


app = FastAPI(
    title="Retail Churn Inference API",
    description=(
        "Online scoring for 90-day customer churn. Hybrid inference: nightly "
        "batch feature materialisation plus request/response scoring. The model "
        "is resolved from the registry by stage, so promotion and rollback "
        "require no redeploy."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
def _risk_band(p: float) -> str:
    return "high" if p >= 0.70 else ("medium" if p >= 0.40 else "low")


def _expected_value_of_contact(p: float) -> float:
    """Expected GBP from contacting this customer, given the campaign economics."""
    biz = CFG["business"]
    gain = (p * biz["campaign_uplift"] * biz["gross_margin_per_retained_customer"])
    return round(float(gain - biz["retention_offer_cost"]), 3)


def _log_prediction(record: dict[str, Any]) -> None:
    """Append-only prediction log.

    This is the join key for delayed-label evaluation: 90 days from now we join
    these rows to realised purchases to compute the true AUC of what was
    actually served. Without persisting the served features and model id, that
    computation is impossible after the fact.
    """
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # logging must never break serving
        print(f"[serve] prediction log write failed: {exc}")


def _resolve_features(req: PredictRequest) -> tuple[dict[str, Any], str, int]:
    """Return ``(feature_row, source, staleness_days)``.

    Path A - the caller supplied a full vector: use it, source='request'.
    Path B - only a customer_id: look up the nightly table, then re-age the
    time-dependent features to ``scoring_time``. Path B is where skew would
    normally creep in, so the re-aging is mandatory, not optional.
    """
    scoring_time = req.scoring_time or CFG["project"]["system_asof"]

    if req.features is not None:
        row = req.features.model_dump()
        return row, "request", 0

    if req.customer_id is None:
        raise HTTPException(422, "Provide either `features` or `customer_id`.")
    if STATE.features is None:
        raise HTTPException(503, "Feature store unavailable; supply `features`.")
    if req.customer_id not in STATE.features.index:
        STATE.lookup_misses += 1
        raise HTTPException(
            404,
            f"customer_id '{req.customer_id}' not found in the feature store. "
            f"A brand-new customer has no history; route to the cold-start rule "
            f"instead of the model.")

    cached = STATE.features.loc[req.customer_id]
    if isinstance(cached, pd.DataFrame):
        cached = cached.iloc[0]
    row = cached.to_dict()
    refreshed = refresh_time_dependent_features(row, scoring_time)
    staleness = int(refreshed.pop("_feature_staleness_days", 0))
    if staleness > 0:
        STATE.stale_servings += 1
    refreshed["customer_id"] = req.customer_id
    return refreshed, "feature_store", staleness


def _score(rows: list[dict[str, Any]]) -> np.ndarray:
    if STATE.model is None:
        raise HTTPException(503, "No model in the 'production' stage. "
                                 "Run `python -m src.train` first.")
    frame = pd.DataFrame(rows)
    X = align_features(frame, strict=True)
    return STATE.model.predict_proba(X)[:, 1]


def _linear_contributions(row: dict[str, Any], p: float, k: int = 5) -> list[dict]:
    """Cheap per-request attribution: standardised deviation from the training mean.

    Not SHAP - SHAP on the calibrated pipeline is too slow for a p95 of 150 ms.
    This reports which features are most unusual for this customer relative to
    the training distribution, which is what a retention agent actually needs
    ("why is this customer flagged?"), computed in microseconds.
    """
    ref = STATE.reference_numeric
    if not ref:
        return []
    scored = []
    for feat, stats_ in ref.items():
        if feat not in row:
            continue
        sd = max(stats_["std"], 1e-9)
        z = (float(row[feat]) - stats_["mean"]) / sd
        scored.append({"feature": feat, "value": round(float(row[feat]), 4),
                       "training_mean": round(stats_["mean"], 4),
                       "z_score": round(float(z), 3)})
    scored.sort(key=lambda d: abs(d["z_score"]), reverse=True)
    return scored[:k]


# --------------------------------------------------------------------------
@app.middleware("http")
async def track_requests(request: Request, call_next):
    STATE.requests_total += 1
    try:
        response = await call_next(request)
    except Exception:
        STATE.errors_total += 1
        raise
    if response.status_code >= 500:
        STATE.errors_total += 1
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 400:
        STATE.errors_total += 1
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path),
                 "model_id": STATE.entry["model_id"] if STATE.entry else None},
    )


# ------------------------------------ endpoints ----------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness + readiness. A model-less process is 'degraded', not 'ok'."""
    return HealthResponse(
        status="ok" if STATE.model is not None else "degraded",
        model_loaded=STATE.model is not None,
        model_id=STATE.entry["model_id"] if STATE.entry else None,
        model_stage=STATE.entry["stage"] if STATE.entry else None,
        feature_store_rows=0 if STATE.features is None else int(len(STATE.features)),
        uptime_seconds=round(time.time() - STATE.started_at, 2),
        version=API_VERSION,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["ops"])
def metadata() -> MetadataResponse:
    """What exactly is serving? Required for auditing any logged prediction."""
    if STATE.entry is None:
        raise HTTPException(503, "No model loaded.")
    e = STATE.entry
    return MetadataResponse(
        model_id=e["model_id"], model_version=e["version"], model_stage=e["stage"],
        registered_at=e["registered_at"], n_features=e["n_features"],
        feature_columns=e["feature_columns"],
        metrics_at_promotion=e["metrics"],
        label_definition=e.get("label_definition", {}),
        train_snapshots=e.get("train_snapshots", []),
        threshold_default=STATE.threshold(),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
def metrics() -> MetricsResponse:
    """Operational counters consumed by the monitoring plan."""
    p95 = STATE.percentile(95)
    slo = float(CFG["serving"]["latency_slo_ms_p95"])
    probs = list(STATE.probabilities)
    return MetricsResponse(
        requests_total=STATE.requests_total,
        errors_total=STATE.errors_total,
        predictions_total=STATE.predictions_total,
        error_rate=round(STATE.errors_total / max(STATE.requests_total, 1), 5),
        latency_ms_avg=round(float(np.mean(STATE.latencies_ms)), 3) if STATE.latencies_ms else None,
        latency_ms_p50=STATE.percentile(50),
        latency_ms_p95=p95,
        latency_ms_p99=STATE.percentile(99),
        latency_slo_ms_p95=slo,
        slo_breached=bool(p95 is not None and p95 > slo),
        mean_predicted_probability=round(float(np.mean(probs)), 5) if probs else None,
        positive_rate=round(float(np.mean(np.array(probs) >= STATE.threshold())), 5) if probs else None,
        feature_store_lookup_misses=STATE.lookup_misses,
        stale_feature_servings=STATE.stale_servings,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Score one customer.

    Accepts a full feature vector *or* a ``customer_id`` for a feature-store
    lookup. Response carries the model identity so the prediction is auditable.
    """
    t0 = time.perf_counter()
    row, source, staleness = _resolve_features(req)
    customer_id = str(row.pop("customer_id", req.customer_id or "unknown"))
    prob = float(_score([row])[0])
    thr = STATE.threshold(req.threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    STATE.latencies_ms.append(latency_ms)
    STATE.probabilities.append(prob)
    STATE.predictions_total += 1

    request_id = str(uuid.uuid4())
    resp = PredictResponse(
        customer_id=customer_id,
        churn_probability=round(prob, 6),
        prediction=int(prob >= thr),
        risk_band=_risk_band(prob),
        threshold=thr,
        model_id=STATE.entry["model_id"],
        model_version=STATE.entry["version"],
        model_stage=STATE.entry["stage"],
        feature_source=source,
        feature_staleness_days=staleness,
        served_at_utc=datetime.now(timezone.utc).isoformat(),
        latency_ms=round(latency_ms, 3),
        request_id=request_id,
        expected_value_of_contact_gbp=_expected_value_of_contact(prob),
        top_contributions=_linear_contributions(row, prob) if req.explain else None,
    )
    _log_prediction({
        "request_id": request_id, "customer_id": customer_id,
        "model_id": STATE.entry["model_id"], "probability": prob,
        "prediction": resp.prediction, "threshold": thr,
        "feature_source": source, "feature_staleness_days": staleness,
        "latency_ms": round(latency_ms, 3),
        "served_at_utc": resp.served_at_utc,
        "scoring_time": req.scoring_time or CFG["project"]["system_asof"],
        "features": row,
    })
    return resp


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Vectorised scoring for a list of customers.

    One ``predict_proba`` call over the whole frame, because per-row calls pay
    the sklearn dispatch and pandas construction cost N times. This is why the
    measured throughput of the batch path is far higher than 1/latency of the
    online path.
    """
    max_n = CFG["serving"]["max_batch_size"]
    if len(req.items) > max_n:
        raise HTTPException(413, f"Batch of {len(req.items)} exceeds max_batch_size={max_n}.")

    t0 = time.perf_counter()
    rows = [i.model_dump() for i in req.items]
    ids = [str(r.pop("customer_id")) for r in rows]
    probs = _score(rows)
    thr = STATE.threshold(req.threshold)
    total_ms = (time.perf_counter() - t0) * 1000.0
    per_row = total_ms / max(len(ids), 1)

    now = datetime.now(timezone.utc).isoformat()
    out = []
    for cid, p in zip(ids, probs):
        p = float(p)
        STATE.probabilities.append(p)
        out.append(PredictResponse(
            customer_id=cid, churn_probability=round(p, 6),
            prediction=int(p >= thr), risk_band=_risk_band(p), threshold=thr,
            model_id=STATE.entry["model_id"], model_version=STATE.entry["version"],
            model_stage=STATE.entry["stage"], feature_source="request",
            feature_staleness_days=0, served_at_utc=now,
            latency_ms=round(per_row, 4), request_id=str(uuid.uuid4()),
            expected_value_of_contact_gbp=_expected_value_of_contact(p),
        ))
    STATE.predictions_total += len(out)
    STATE.latencies_ms.append(total_ms)

    return BatchPredictResponse(
        n=len(out), predictions=out, model_id=STATE.entry["model_id"],
        total_latency_ms=round(total_ms, 3),
        rows_per_second=round(len(out) / (total_ms / 1000.0), 1),
    )


@app.post("/admin/reload", tags=["ops"])
def reload_model() -> dict[str, Any]:
    """Re-resolve the production model. The rollback lever, no redeploy needed."""
    return {"status": "reloaded", **STATE.reload()}
