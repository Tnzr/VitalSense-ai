"""Asculto web app: Inference-as-a-Service + Educational Interactive.

Run:  uvicorn webapp.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.dirname(HERE)

from . import course, game  # noqa: E402
from . import inference, report  # noqa: E402
from . import llm as llm_mod  # noqa: E402
from . import playback  # noqa: E402
from .config import DISCLAIMER  # noqa: E402

MAX_UPLOAD_BYTES = int(os.environ.get("ASCULTO_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.environ.get("ASCULTO_MAX_FILES", "32"))
MAX_REQUEST_BYTES = int(os.environ.get("ASCULTO_MAX_REQUEST_MB", "100")) * 1024 * 1024
_CHUNK = 1 << 20


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    inference.get_registry()  # warm-load checkpoint ensembles
    yield


class _RequestSizeLimit:
    """Reject oversized bodies before multipart parsing spools them to disk."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = globals()["MAX_REQUEST_BYTES"]
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            resp = JSONResponse({"detail": f"request body exceeds {limit // (1024 * 1024)} MB"}, status_code=413)
            return await resp(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail="request body too large")
            return message

        await self.app(scope, limited_receive, send)


app = FastAPI(title="Asculto — Inference as a Service", version="0.1.0", lifespan=_lifespan)
app.add_middleware(_RequestSizeLimit)
app.mount("/static", StaticFiles(directory=os.path.join(WEBAPP, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(WEBAPP, "templates"))


def _opts(heads: List[str], explanation: str, figures: str, disclaimer: bool, audio: bool,
          temporal: bool = False, window_s: float = 3.0, hop_s: float = 0.5,
          temporal_mode: str = "single", scales: str = "1,3,15",
          patient_context: str = "", ai_narrative: bool = True) -> report.ReportOptions:
    valid = [h for h in heads if h in report.DEFAULT_HEADS] or list(report.DEFAULT_HEADS)
    try:
        scale_list = [max(0.5, min(float(s), 60.0)) for s in str(scales).split(",") if s.strip()]
    except ValueError:
        scale_list = [1.0, 3.0, 15.0]
    if not scale_list:
        scale_list = [1.0, 3.0, 15.0]
    return report.ReportOptions(
        heads=valid,
        explanation=explanation if explanation in ("brief", "standard", "detailed") else "standard",
        figures=figures if figures in ("none", "key", "all") else "all",
        include_disclaimer=disclaimer,
        audio_playback=audio,
        temporal=temporal,
        window_s=max(0.5, min(float(window_s or 3.0), 10.0)),
        hop_s=max(0.1, min(float(hop_s or 0.5), 5.0)),
        temporal_mode=temporal_mode if temporal_mode in ("single", "multiscale") else "single",
        scales=scale_list,
        patient_context=patient_context,
        ai_narrative=ai_narrative,
    )




def _check_file_count(files: List[UploadFile]) -> None:
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=413, detail=f"at most {MAX_FILES_PER_REQUEST} files per request")


def _save_upload(upload: UploadFile) -> str:
    """Stream an upload to a temp file, rejecting anything over MAX_UPLOAD_BYTES."""
    suffix = os.path.splitext(os.path.basename(upload.filename or "clip.wav"))[1].lower()
    if not suffix or len(suffix) > 6 or not suffix[1:].isalnum():
        suffix = ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    written = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := upload.file.read(_CHUNK):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError(f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
                f.write(chunk)
    except BaseException:
        os.remove(path)
        raise
    return path


def _audio_data_uri(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        mime = "audio/wav" if path.lower().endswith(".wav") else "audio/octet-stream"
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    except OSError:
        return None


@app.get("/health")
def health() -> JSONResponse:
    reg = inference.get_registry()
    return JSONResponse({
        "status": "ok",
        "backend": os.environ.get("ASCULTO_BACKEND", "torch"),
        "device": str(reg.device),
        "loaded_heads": reg.available(),
        "errors": reg.errors,
        "data_dir": os.path.basename(inference.config.DATA_DIR),
    })


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    reg = inference.get_registry()
    return templates.TemplateResponse(request, "index.html", {
        "heads": [h for h in inference.config.model_registry()],
        "loaded": reg.available(),
        "errors": reg.errors,
        "disclaimer": DISCLAIMER,
        "domains": game.DOMAINS,
        "examples": course.landing_examples(),
        "knowledge": course.knowledge,
        "audio_url": course.audio_url,
    })


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    files: List[UploadFile] = File(...),
    heads: List[str] = Form(default=[]),
    explanation: str = Form("standard"),
    figures: str = Form("all"),
    include_disclaimer: bool = Form(True),
    audio_playback: bool = Form(True),
    temporal: bool = Form(False),
    window_s: float = Form(3.0),
    hop_s: float = Form(0.5),
    temporal_mode: str = Form("single"),
    scales: str = Form("1,3,15"),
    patient_context: str = Form(""),
    ai_narrative: bool = Form(True),
):
    _check_file_count(files)
    opts = _opts(heads, explanation, figures, include_disclaimer, audio_playback, temporal, window_s, hop_s,
                 temporal_mode, scales, patient_context, ai_narrative)
    results = []
    for up in files:
        path = None
        try:
            path = _save_upload(up)
            rep = report.analyze_file(path, opts)
            rep["audio_uri"] = _audio_data_uri(path) if audio_playback and len(files) == 1 else None
            rep["size_bytes"] = os.path.getsize(path)
            results.append(rep)
        except Exception as e:  # pragma: no cover - user-facing error
            results.append({"filename": up.filename, "error": f"{type(e).__name__}: {e}"})
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

    if len(results) == 1:
        return templates.TemplateResponse(request, "report.html", {
            "report": results[0], "disclaimer": DISCLAIMER,
        })
    rows = [report.batch_row(r) for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    return templates.TemplateResponse(request, "batch.html", {
        "rows": rows, "errors": errors, "n": len(results),
        "disclaimer": DISCLAIMER,
    })


@app.post("/api/analyze")
async def api_analyze(
    files: List[UploadFile] = File(...),
    heads: str = Form(""),
    explanation: str = Form("standard"),
    figures: str = Form("all"),
    temporal: bool = Form(False),
    window_s: float = Form(3.0),
    hop_s: float = Form(0.5),
    temporal_mode: str = Form("single"),
    scales: str = Form("1,3,15"),
    patient_context: str = Form(""),
    ai_narrative: bool = Form(True),
):
    """Programmatic Inference-as-a-Service endpoint (JSON)."""
    _check_file_count(files)
    opts = _opts([h for h in heads.split(",") if h], explanation, figures, True, False,
                 temporal, window_s, hop_s, temporal_mode, scales, patient_context, ai_narrative)
    out = []
    for up in files:
        path = None
        try:
            path = _save_upload(up)
            rep = report.analyze_file(path, opts)
            out.append(rep)
        except Exception as e:
            out.append({"filename": up.filename, "error": f"{type(e).__name__}: {e}"})
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
    return JSONResponse({"count": len(out), "results": out})


@app.post("/api/llm/report")
async def api_llm_report(payload: dict) -> JSONResponse:
    """Personalize an auscultation report from findings + patient context and
    optional conversation history. Returns the same structured narrative shape
    whether the LLM is configured or the offline synthesizer is used."""
    findings = payload.get("findings") or []
    ctx = str(payload.get("patient_context") or "")
    history = payload.get("history") or []
    return JSONResponse(llm_mod.generate_narrative(findings, ctx, history))


@app.get("/game", response_class=HTMLResponse)
def game_page(request: Request):
    return templates.TemplateResponse(request, "game.html", {
        "domains": game.DOMAINS, "disclaimer": DISCLAIMER,
    })


@app.get("/api/game/round")
def game_round(domain: str = "heart_types") -> JSONResponse:
    try:
        return JSONResponse(game.new_round(domain))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/game/answer")
async def game_answer(payload: dict) -> JSONResponse:
    rid = payload.get("round_id")
    answer = payload.get("answer")
    if not rid or answer is None:
        raise HTTPException(status_code=400, detail="round_id and answer are required")
    return JSONResponse(game.grade(rid, answer))


@app.get("/game/audio/{sample_id}")
def game_audio(sample_id: str, task: str = "heart", scheme: str = "10class"):
    """Dataset clip re-encoded for browser playback (4 kHz WAVs decode as
    silence in HTML5 `<audio>`; see `webapp.app.playback`)."""
    path = game.resolve_audio_path(sample_id, task, scheme)
    if not path:
        raise HTTPException(status_code=404, detail="clip not found")
    return Response(content=playback.playback_bytes_for(path), media_type="audio/wav")


# ------------------------------------------------------------ training course
@app.get("/course", response_class=HTMLResponse)
def course_home(request: Request):
    return templates.TemplateResponse(request, "course.html", {
        "modules": course.MODULES, "disclaimer": DISCLAIMER,
    })


@app.get("/course/{module_id}", response_class=HTMLResponse)
def course_module(request: Request, module_id: str):
    mod = course.by_id(module_id)
    if not mod:
        raise HTTPException(status_code=404, detail="module not found")
    domain = dict(game.DOMAINS.get(mod.practice_domain, {}))
    if mod.practice_domain:
        domain["key"] = mod.practice_domain
    return templates.TemplateResponse(request, "course_module.html", {
        "module": mod,
        "modules": course.MODULES,
        "module_index": next((i for i, m in enumerate(course.MODULES) if m.id == module_id), 0),
        "practice_domain": domain,
        "disclaimer": DISCLAIMER,
        "knowledge": course.knowledge,
        "audio_url": course.audio_url,
    })


@app.post("/api/course/quiz")
async def course_quiz(payload: dict) -> JSONResponse:
    module_id = payload.get("module_id")
    answers = payload.get("answers") or []
    return JSONResponse(course.grade_quiz(module_id, answers))


@app.get("/demo/report", response_class=HTMLResponse)
def demo_report(request: Request, sample_id: str = "", task: str = "sound", scheme: str = "binary",
                heads: str = "", temporal_mode: str = "multiscale", explanation: str = "detailed",
                figures: str = "all"):
    """Render a full report for a built-in dataset clip (shareable + screenshot-friendly)."""
    if not sample_id:
        manifest = game._manifest(task, scheme)
        sample_id = str(manifest.iloc[0]["sample_id"])
    path = game.resolve_audio_path(sample_id, task, scheme)
    if not path:
        raise HTTPException(status_code=404, detail=f"sample {sample_id} not found for {task}/{scheme}")
    opts = _opts([h for h in heads.split(",") if h] or list(report.DEFAULT_HEADS),
                 explanation, figures, True, True, True, 3.0, 0.5, temporal_mode, "1,3,15")
    rep = report.analyze_file(path, opts)
    rep["audio_uri"] = _audio_data_uri(path)
    rep["demo"] = {"sample_id": sample_id, "task": task, "scheme": scheme}
    return templates.TemplateResponse(request, "report.html", {"report": rep, "disclaimer": DISCLAIMER})


@app.get("/demo/batch", response_class=HTMLResponse)
def demo_batch(request: Request, n: int = 4):
    """Render a batch table over a few built-in dataset clips."""
    opts = _opts(list(report.DEFAULT_HEADS), "standard", "all", True, False, False)
    picks = [("sound", "binary"), ("heart", "binary"), ("lung", "binary"), ("heart", "10class"), ("lung", "6class")]
    results = []
    for task, scheme in picks[: max(1, min(n, len(picks)))]:
        manifest = game._manifest(task, scheme)
        sid = str(manifest.iloc[0]["sample_id"])
        path = game.resolve_audio_path(sid, task, scheme)
        if path:
            results.append(report.analyze_file(path, opts))
    rows = [report.batch_row(r) for r in results]
    return templates.TemplateResponse(request, "batch.html", {
        "rows": rows, "errors": [], "n": len(rows), "disclaimer": DISCLAIMER,
    })
