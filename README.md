# Customer Churn Prediction — Mini Production ML System

MLME assignment. The system predicts whether a customer who bought something in
the last 180 days will stop buying in the next 90 days, and serves that
prediction through an API.

The model itself is simple on purpose. Most of the work here is the stuff around
it: the ingestion step, the shared feature code, the promotion check before a
model goes live, and the monitoring that tells you when it has gone stale.

**Data:** UCI Online Retail — 541,909 transactions from a UK online gift shop,
Dec 2010 to Dec 2011, 4,372 customers. It is included in `data/raw/` so
everything runs offline.

---

## Submission files

| What | Where |
|---|---|
| Notebook (full analysis, already run) | `notebooks/2025EM1100339_KinjalJoshi_Notebook.ipynb` |
| Design document | `docs/2025EM1100339_KinjalJoshi_DesignDocument.docx` |
| Architecture diagram | `docs/2025EM1100339_KinjalJoshi_ArchitectureDiagram.png` |
| Slides (demo) | `docs/2025EM1100339_KinjalJoshi_PPT.pptx` |
| Screenshots | `docs/screenshots/` |
| Code | `src/`, `scripts/`, `tests/`, `configs/` |

---

## How to run it

```bash
pip install -r requirements.txt

make data      # clean the raw CSV into an event table, and make 9 daily CSV files
make ingest    # run the daily-batch ingestion (380,179 -> 399,654 events)
make train     # build features, train 4 models, run the promotion check
make test      # 68 tests
```

Then open the notebook. It is saved with all outputs, so you can read it without
running anything, but `make data` and `make ingest` need to have been run first
if you want to re-execute the cells.

To try the API:

```bash
make serve     # starts on http://127.0.0.1:8000  (docs at /docs)
make load      # measures latency and throughput, in a second terminal
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"17850"}'
```

```json
{
  "customer_id": "17850",
  "churn_probability": 0.574661,
  "prediction": 1,
  "risk_band": "medium",
  "model_id": "retail-churn-90d:v6",
  "model_version": "v6",
  "feature_source": "feature_store",
  "latency_ms": 8.41
}
```

---

## Results

I trained four models so each one had something to prove against the one below it.

| Model | Validation AUC | Test AUC |
|---|---|---|
| Predict the base rate | 0.500 | 0.500 |
| Recency rule (sort by days since last order) | 0.665 | 0.657 |
| Logistic regression | 0.753 | **0.743** |
| LightGBM + calibration | **0.792** | 0.726 |

The recency rule matters most as a baseline, because that is basically what a
retention team already does. Beating it by 0.127 AUC is what makes the model
worth having.

**The interesting bit:** LightGBM wins on validation but loses on test. Churn
rate drops from 0.475 to 0.393 across the folds because of Christmas trading, and
LightGBM had fitted patterns from March–May that don't hold in August. The
logistic model was too simple to fit those patterns, so it didn't lose them
either. I promoted LightGBM because validation was the only fold I was allowed to
select on, but in a real deployment I'd canary it rather than switch over. This is
discussed in section B10 of the notebook.

**API performance:** p50 9.7 ms, p95 11.1 ms against a 150 ms target, and 6,324
rows/sec in batch mode. Measured over real HTTP, not in-process.

**Monitoring:** the drift check gives WARN on the real August data (the customer
base genuinely ages, so some drift is expected) and ALERT on an injected bug that
triples revenue. A missing column is blocked before it reaches the model.

---

## Why the model is worth deploying

A contact costs £8, a saved customer is worth £45 of margin, and a retention
offer works about 30% of the time. So the campaign only breaks even if precision
in the contacted group is above `8 / (0.30 x 45) = 0.593`. Random targeting only
gets the base rate (0.39–0.48), so it loses money. That 0.593 is used as an actual
promotion gate, not just a comment.

---

## Project structure

```
configs/config.yaml     all paths, thresholds and settings in one file
src/
  features.py           the 23 features - imported by BOTH training and serving
  data_prep.py          raw CSV -> clean event table, with a cleaning audit
  ingest.py             daily batch ingestion: validate, dedupe, merge, log
  labels.py             snapshots, labels, and the temporal train/test split
  train.py              trains the 4 models and runs the promotion gates
  evaluate.py           metrics, campaign value, promote/don't promote
  registry.py           model versions and stages, with rollback
  monitoring.py         drift (PSI/KS), data quality and schema checks
  retraining.py         when to retrain, and when to refuse
  serving/app.py        the FastAPI service
scripts/
  make_daily_batches.py splits history into daily CSVs for the ingestion demo
  load_test.py          latency and throughput measurement
tests/                  68 tests
data/                   raw CSV, warehouse, feature store, drift reference
models/                 saved models + registry.json
artifacts/              eval reports, monitoring reports, logs
```

---

## Things I'd flag myself

- **The 30% offer success rate is assumed, not measured.** All the money numbers
  depend on it. Properly measuring it needs a randomised control group, which this
  dataset can't give me. That would be my first next step.
- **LightGBM being worse on test** (see above). Not a bug, but worth knowing.
- **24.9% of transactions have no customer ID** and had to be dropped, so the
  model only really applies to registered customers.
- **One year of data** means I can't separate seasonal effects from a real trend.
- **The daily files are simulated** by splitting up a static export, so I never
  had to deal with genuinely late-arriving data.
- **One server process** handles about 100 requests/sec. Real use would need
  several, plus a proper feature store instead of a parquet file.

---

## Data source

UCI Machine Learning Repository, *Online Retail*. Chen, D., Sain, S.L. and Guo, K.
(2012). Data mining for the online retail industry. *Journal of Database Marketing
and Customer Strategy Management*, 19(3), 197–208.
