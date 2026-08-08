"""Pydantic request/response contracts for the inference API.

The schema *is* the interface documentation: FastAPI derives OpenAPI from it, so
the field constraints below are simultaneously validation, docs and a test
fixture. Two deliberate choices:

* Every numeric feature has a ``ge`` bound matching the domain ranges declared
  in ``features.FEATURE_RANGES``. A negative ``recency_days`` is physically
  impossible, so it is a 422 at the edge rather than a silent prediction.
* The response always carries ``model_version``, ``model_id`` and
  ``feature_source``. Without those a logged prediction cannot be attributed to
  a model, which makes post-hoc evaluation and rollback analysis impossible.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """A fully specified feature vector (the 'client computes features' path)."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., examples=["17850"])

    recency_days: float = Field(..., ge=0, le=1000)
    tenure_days: float = Field(..., ge=0, le=1000)
    active_days_ratio: float = Field(..., ge=0, le=1)
    avg_interpurchase_gap_days: float = Field(..., ge=0, le=1000)
    interpurchase_gap_cv: float = Field(..., ge=0, le=20)
    recency_over_avg_gap: float = Field(..., ge=0, le=200)
    orders_30d: float = Field(..., ge=0, le=200)
    orders_90d: float = Field(..., ge=0, le=400)
    orders_180d: float = Field(..., ge=0, le=800)
    order_freq_trend: float = Field(..., ge=0, le=50)
    revenue_30d: float = Field(..., ge=0)
    revenue_90d: float = Field(..., ge=0)
    revenue_180d: float = Field(..., ge=0)
    lifetime_revenue: float = Field(..., ge=0)
    aov_90d: float = Field(..., ge=0)
    revenue_trend: float = Field(..., ge=0, le=50)
    distinct_products_90d: float = Field(..., ge=0, le=5000)
    avg_items_per_order_90d: float = Field(..., ge=0)
    avg_unit_price_90d: float = Field(..., ge=0)
    product_breadth_ratio: float = Field(..., ge=0, le=1)
    n_returns_180d: float = Field(..., ge=0, le=500)
    return_value_ratio_180d: float = Field(..., ge=0, le=50)
    country_grp: str = Field("United Kingdom", examples=["United Kingdom"])


class PredictRequest(BaseModel):
    """Either supply ``features``, or supply ``customer_id`` for a store lookup."""

    model_config = ConfigDict(extra="forbid")

    features: CustomerFeatures | None = None
    customer_id: str | None = None
    scoring_time: str | None = Field(
        None,
        description="ISO timestamp. Used to re-age cached time-dependent "
                    "features so a stale row cannot cause offline/online skew.",
        examples=["2011-12-09T09:30:00"],
    )
    threshold: float | None = Field(None, ge=0, le=1)
    explain: bool = False


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    prediction: int
    risk_band: Literal["low", "medium", "high"]
    threshold: float
    model_id: str
    model_version: str
    model_stage: str
    feature_source: Literal["request", "feature_store"]
    feature_staleness_days: int
    served_at_utc: str
    latency_ms: float
    request_id: str
    expected_value_of_contact_gbp: float | None = None
    top_contributions: list[dict] | None = None


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CustomerFeatures] = Field(..., min_length=1)
    threshold: float | None = Field(None, ge=0, le=1)


class BatchPredictResponse(BaseModel):
    n: int
    predictions: list[PredictResponse]
    model_id: str
    total_latency_ms: float
    rows_per_second: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_id: str | None
    model_stage: str | None
    feature_store_rows: int
    uptime_seconds: float
    version: str


class MetadataResponse(BaseModel):
    model_id: str
    model_version: str
    model_stage: str
    registered_at: str
    n_features: int
    feature_columns: list[str]
    metrics_at_promotion: dict
    label_definition: dict
    train_snapshots: list[str]
    threshold_default: float


class MetricsResponse(BaseModel):
    """Operational counters the monitoring plan reads (Prometheus-style)."""

    requests_total: int
    errors_total: int
    predictions_total: int
    error_rate: float
    latency_ms_avg: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    latency_ms_p99: float | None
    latency_slo_ms_p95: float
    slo_breached: bool
    mean_predicted_probability: float | None
    positive_rate: float | None
    feature_store_lookup_misses: int
    stale_feature_servings: int
