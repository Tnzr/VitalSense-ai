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
