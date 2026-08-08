.PHONY: help setup data ingest train test serve load drift retrain pipeline clean docker
PY := python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup:   ## install dependencies
	pip install -r requirements.txt

data:    ## clean raw CSV -> event warehouse, truncated for the ingestion demo
	$(PY) -m src.data_prep --until 2011-11-29
	$(PY) -m scripts.make_daily_batches --last-n-days 10

ingest:  ## run micro-batch ingestion on every pending daily file
	$(PY) -m src.ingest --all

train:   ## full training pipeline: features, models, guardrail, registry
	$(PY) -m src.train

test:    ## run the test suite
	$(PY) -m pytest tests/ -v

serve:   ## start the inference API
	uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --reload

load:    ## measure latency / throughput (needs `make serve` in another shell)
	$(PY) -m scripts.load_test --url http://127.0.0.1:8000 --n 400

drift:   ## run the monitoring suite on the latest snapshot
	$(PY) -m src.monitoring_cli || $(PY) -c "print('see notebooks/ for the drift demo')"

pipeline: data ingest train test  ## end-to-end reproduction

docker:  ## build and run the container
	docker build -t churn-api:latest .
	docker run --rm -p 8000:8000 churn-api:latest

clean:   ## remove generated artefacts (keeps raw data)
	rm -rf models/*.joblib models/registry.json artifacts/eval/* \
	       artifacts/monitoring/* artifacts/logs/* data/warehouse/* \
	       data/feature_store/* data/reference/* data/incoming/*.csv
