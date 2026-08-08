"""API contract tests.

Run in-process with ``TestClient`` so they need no server and no network. They
assert the *contract* the consumer depends on - status codes, required response
fields, validation behaviour - rather than the numeric prediction, which is the
model's business and is covered by the offline evaluation.

These tests require a trained model in the registry. If none exists they skip
rather than fail, so a fresh clone can run the suite before training.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import CFG
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def has_model(client) -> bool:
    return bool(client.get("/health").json()["model_loaded"])


@pytest.fixture(scope="module")
def sample_features() -> dict:
    fs = CFG.get_path("paths.feature_store")
    if not fs.exists():
        pytest.skip("feature store not built; run `python -m src.train`")
    row = pd.read_parquet(fs).iloc[0]
    payload = {"customer_id": str(row["customer_id"])}
    payload.update({c: float(row[c]) for c in NUMERIC_FEATURES})
    payload.update({c: str(row[c]) for c in CATEGORICAL_FEATURES})
    return payload


# ------------------------------- ops --------------------------------------
def test_health_returns_expected_shape(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"status", "model_loaded", "model_id", "version"}
    assert body["status"] in {"ok", "degraded"}


def test_openapi_schema_is_generated(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert {"/predict", "/predict/batch", "/health", "/metadata",
            "/metrics"} <= set(paths)


def test_metadata_exposes_model_identity(client, has_model):
    if not has_model:
        pytest.skip("no trained model")
    body = client.get("/metadata").json()
    assert body["model_id"] and body["model_version"]
    assert body["n_features"] == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert body["label_definition"]["horizon_days"] == CFG["label"]["horizon_days"]


def test_metrics_counters_increase(client, has_model, sample_features):
    if not has_model:
        pytest.skip("no trained model")
    before = client.get("/metrics").json()["predictions_total"]
    client.post("/predict", json={"features": sample_features})
    after = client.get("/metrics").json()["predictions_total"]
    assert after == before + 1


# ---------------------------- predict -------------------------------------
def test_predict_with_explicit_features(client, has_model, sample_features):
    if not has_model:
        pytest.skip("no trained model")
    r = client.post("/predict", json={"features": sample_features})
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["churn_probability"] <= 1.0
    assert b["prediction"] in (0, 1)
    assert b["risk_band"] in {"low", "medium", "high"}
    assert b["feature_source"] == "request"
    # Model identity must always be present, or a logged prediction cannot be
    # attributed to a model version after the fact.
    assert b["model_id"] and b["model_version"] and b["request_id"]


def test_predict_by_customer_id_uses_feature_store(client, has_model, sample_features):
    if not has_model:
        pytest.skip("no trained model")
    r = client.post("/predict", json={"customer_id": sample_features["customer_id"]})
    assert r.status_code == 200
    assert r.json()["feature_source"] == "feature_store"


def test_prediction_is_deterministic(client, has_model, sample_features):
    if not has_model:
        pytest.skip("no trained model")
    a = client.post("/predict", json={"features": sample_features}).json()
    b = client.post("/predict", json={"features": sample_features}).json()
    assert a["churn_probability"] == b["churn_probability"]


def test_threshold_override_changes_label_not_probability(client, has_model,
                                                          sample_features):
    if not has_model:
        pytest.skip("no trained model")
    lo = client.post("/predict", json={"features": sample_features,
                                       "threshold": 0.01}).json()
    hi = client.post("/predict", json={"features": sample_features,
                                       "threshold": 0.99}).json()
    assert lo["churn_probability"] == hi["churn_probability"]
    assert lo["prediction"] == 1 and hi["prediction"] == 0


def test_unknown_customer_returns_404(client, has_model):
    if not has_model:
        pytest.skip("no trained model")
    r = client.post("/predict", json={"customer_id": "NO_SUCH_CUSTOMER_XYZ"})
    assert r.status_code == 404
    assert "cold-start" in r.json()["error"]


def test_empty_request_is_rejected(client):
    assert client.post("/predict", json={}).status_code in (422, 503)


# ---------------------------- validation ----------------------------------
def test_negative_recency_rejected(client, sample_features):
    bad = {**sample_features, "recency_days": -1}
    assert client.post("/predict", json={"features": bad}).status_code == 422


def test_ratio_above_one_rejected(client, sample_features):
    bad = {**sample_features, "active_days_ratio": 1.7}
    assert client.post("/predict", json={"features": bad}).status_code == 422


def test_missing_feature_rejected(client, sample_features):
    bad = {k: v for k, v in sample_features.items() if k != "orders_90d"}
    assert client.post("/predict", json={"features": bad}).status_code == 422


def test_unexpected_field_rejected(client, sample_features):
    """`extra="forbid"` catches typos and stale clients instead of silently
    ignoring a field the caller believed was being used."""
    bad = {**sample_features, "recency_dayz": 5}
    assert client.post("/predict", json={"features": bad}).status_code == 422


def test_unknown_country_does_not_error(client, has_model, sample_features):
    """A country never seen in training must fall into the infrequent bucket."""
    if not has_model:
        pytest.skip("no trained model")
    payload = {**sample_features, "country_grp": "Wakanda"}
    assert client.post("/predict", json={"features": payload}).status_code == 200


# ------------------------------ batch -------------------------------------
def test_batch_predict_returns_all_rows(client, has_model, sample_features):
    if not has_model:
        pytest.skip("no trained model")
    items = [{**sample_features, "customer_id": f"c{i}"} for i in range(25)]
    r = client.post("/predict/batch", json={"items": items})
    assert r.status_code == 200
    b = r.json()
    assert b["n"] == 25 and len(b["predictions"]) == 25
    assert b["rows_per_second"] > 0


def test_batch_matches_single_predictions(client, has_model, sample_features):
    """Batch and online paths must agree exactly - same model, same transform."""
    if not has_model:
        pytest.skip("no trained model")
    single = client.post("/predict", json={"features": sample_features}).json()
    batch = client.post("/predict/batch",
                        json={"items": [sample_features]}).json()
    assert batch["predictions"][0]["churn_probability"] == pytest.approx(
        single["churn_probability"], abs=1e-9)


def test_oversized_batch_rejected(client, has_model, sample_features):
    if not has_model:
        pytest.skip("no trained model")
    n = CFG["serving"]["max_batch_size"] + 1
    items = [{**sample_features, "customer_id": f"c{i}"} for i in range(n)]
    assert client.post("/predict/batch", json={"items": items}).status_code == 413


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422


# ------------------------------ admin -------------------------------------
def test_reload_endpoint_reports_model_ids(client, has_model):
    if not has_model:
        pytest.skip("no trained model")
    b = client.post("/admin/reload").json()
    assert b["status"] == "reloaded" and b["current_model_id"]
