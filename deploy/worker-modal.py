"""Asculto batch/GPU worker on Modal (serverless, scale-to-zero).

Stage-1 uses Modal only for batch or large-model GPU inference; the steady-state
report path runs on CPU (Cloudflare Containers). Stage-2 can keep Modal for burst
GPU alongside Triton on Kubernetes.

Deploy:
  modal deploy deploy/worker-modal.py
Run a batch locally:
  modal run deploy/worker-modal.py --job-file jobs.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.14.1", "torchaudio==2.11.0", "numpy", "scipy", "pandas",
        "matplotlib", "librosa", "soundfile", "fastapi", "python-multipart",
        "onnxruntime", "onnx", "onnxconverter-common",
    )
    .env({"PYTHONPATH": "/root/asculto:/root/asculto/ml", "ASCULTO_BACKEND": "onnx",
          "ASCULTO_ONNX_DIR": "/models", "ASCULTO_DEVICE": "cpu"})
    .add_local_dir("ml", "/root/asculto/ml")
    .add_local_dir("webapp", "/root/asculto/webapp")
    .add_local_dir("deploy", "/root/asculto/deploy")
)

app = modal.App("asculto-inference")


def _run_job(job: dict) -> dict:
    # imported inside the container so the local dirs are on PYTHONPATH
    from inference_worker import run_job

    return run_job(job)


@app.function(image=image, cpu=2.0, memory=4096, timeout=600)
def run_report(job: dict) -> dict:
    return _run_job(job)


@app.function(image=image, cpu=2.0, memory=4096, timeout=900)
def run_batch(jobs: list[dict]) -> list[dict]:
    return [_run_job(j) for j in jobs]


@app.function(image=image, gpu="A10G", memory=8192, timeout=900)
def run_report_gpu(job: dict) -> dict:
    """Reserved for large models / batch; CPU is cheaper for the current heads."""
    return _run_job(job)


@app.local_entrypoint()
def main(job_file: str = ""):
    if not job_file:
        print("pass --job-file jobs.jsonl")
        return
    jobs = [json.loads(line) for line in Path(job_file).read_text().splitlines() if line.strip()]
    for result in run_batch.map([jobs]):
        for r in result:
            print(json.dumps({"filename": r.get("filename"), "verdict": r.get("verdict")}))
