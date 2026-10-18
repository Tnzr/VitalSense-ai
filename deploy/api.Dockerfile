# Asculto API app (FastAPI) — local/dev and Stage-2 portability image.
# Build from the repo root:  docker build -f deploy/api.Dockerfile -t asculto-api .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/ml \
    ASCULTO_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml/requirements.txt ml/requirements.txt
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r ml/requirements.txt -r webapp/requirements.txt

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "webapp.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
