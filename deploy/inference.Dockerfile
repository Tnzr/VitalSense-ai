# Asculto inference worker — ONNX Runtime CPU, scale-to-zero.
# Build from the repo root:  docker build -f deploy/inference.Dockerfile -t asculto-worker .
#
# Checkpoints/ONNX artifacts are NOT baked in: mount a volume at /models (the
# `onnx_manifest.json` + exported graphs) or bake them in a release image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/ml \
    ASCULTO_BACKEND=onnx \
    ASCULTO_ONNX_DIR=/models \
    ASCULTO_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml/requirements.txt ml/requirements.txt
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r ml/requirements.txt -r webapp/requirements.txt \
        onnxruntime onnx onnxconverter-common

COPY . .
RUN mkdir -p /models

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "deploy.inference_worker:app", "--host", "0.0.0.0", "--port", "8080"]
