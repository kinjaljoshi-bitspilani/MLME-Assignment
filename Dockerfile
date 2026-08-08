# Multi-stage build: wheels are compiled in a builder image so the runtime
# layer carries no compiler toolchain. Keeps the final image small and shrinks
# the CVE surface, which matters for anything internet-facing.
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim AS runtime
# libgomp is LightGBM's OpenMP runtime - required at inference, not just build.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/reference/ ./data/reference/
COPY data/feature_store/ ./data/feature_store/

# Never run as root in a container that accepts network traffic.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Orchestrators need a real readiness signal, not just "process alive".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if b['model_loaded'] else 1)"
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
