import io

import pytest
from fastapi.testclient import TestClient

from webapp.app import main


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["loaded_heads"], list)


def test_index_renders(client):
    assert client.get("/").status_code == 200


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
    files = [("files", ("big.wav", io.BytesIO(b"\0" * 4096), "audio/wav"))]
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert "upload limit" in res["error"]


def test_request_size_limit_rejects_before_parsing(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 1024)
    files = [("files", ("big.wav", io.BytesIO(b"\0" * 4096), "audio/wav"))]
    assert client.post("/api/analyze", files=files).status_code == 413


def test_file_count_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_FILES_PER_REQUEST", 1)
    files = [("files", (f"c{i}.wav", io.BytesIO(b"RIFF"), "audio/wav")) for i in range(2)]
    assert client.post("/api/analyze", files=files).status_code == 413


def test_save_upload_sanitizes_suffix(monkeypatch):
    class Up:
        filename = "../../evil.<script>"
        file = io.BytesIO(b"abc")

    path = main._save_upload(Up())
    try:
        assert path.endswith(".wav")
    finally:
        import os
        os.remove(path)


def test_results_keep_original_filename(client):
    files = [("files", ("C:\\clips\\patient 7.wav", b"", "audio/wav"))]
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200
    assert r.json()["results"][0]["filename"] == "patient 7.wav"


def test_matrix_page_renders(client):
    r = client.get("/matrix")
    assert r.status_code == 200
    assert "Sound matrix" in r.text


def test_matrix_mel_unknown_clip_404(client):
    assert client.get("/api/matrix/mel/does-not-exist?task=heart&scheme=10class").status_code == 404


def test_api_analyze_returns_viz_and_quant(client):
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    t = np.arange(4000 * 6) / 4000
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 120 * t)).astype("float32"), 4000, format="WAV", subtype="PCM_16")
    files = [("files", ("tone.wav", io.BytesIO(buf.getvalue()), "audio/wav"))]
    rep = client.post("/api/analyze", files=files, data={"figures": "none", "ai_narrative": "false"}).json()["results"][0]
    assert rep["viz"]["mel"]["n_mels"] == 64
    assert {s["title"] for s in rep["quant"]["sections"]} >= {"Signal", "Spectrum"}
