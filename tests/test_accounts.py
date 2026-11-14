import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from webapp.app import accounts, longitudinal, main


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("ASCULTO_DB", str(tmp_path_factory.mktemp("db") / "test.db"))
    with TestClient(main.app) as c:
        yield c
    mp.undo()


def _tone() -> bytes:
    buf = io.BytesIO()
    t = np.arange(4000 * 6) / 4000
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 120 * t)).astype("float32"), 4000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _register(client, email, password="correct horse"):
    client.cookies.clear()
    return client.post("/register", data={"email": email, "password": password}, follow_redirects=False)


def test_password_hash_roundtrip():
    h = accounts.hash_password("s3cret-pass")
    assert h.startswith("scrypt$") and "s3cret" not in h
    assert accounts.verify_password("s3cret-pass", h)
    assert not accounts.verify_password("wrong", h)
    assert not accounts.verify_password("x", "garbage")


def test_register_login_logout(client):
    r = _register(client, "a@example.com")
    assert r.status_code == 303 and r.headers["location"] == "/exams"
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/exams").status_code == 200
    token = client.cookies.get(accounts.SESSION_COOKIE)
    client.post("/logout", follow_redirects=False)
    assert accounts.user_for_token(token) is None
    client.cookies.clear()
    r = client.get("/exams", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert client.post("/login", data={"email": "A@example.com ", "password": "wrong-pass"}).status_code == 401
    r = client.post("/login", data={"email": "A@example.com", "password": "correct horse"}, follow_redirects=False)
    assert r.status_code == 303


def test_register_validation(client):
    assert _register(client, "dup@example.com").status_code == 303
    assert _register(client, "dup@example.com").status_code == 400
    assert _register(client, "short@example.com", "1234567").status_code == 400
    assert _register(client, "not-an-email").status_code == 400


def test_open_redirect_blocked(client):
    client.cookies.clear()
    r = client.post("/register", data={"email": "r@example.com", "password": "correct horse", "next": "//evil.example"},
                    follow_redirects=False)
    assert r.headers["location"] == "/exams"


def test_saved_exams_are_private_and_exportable(client):
    _register(client, "owner@example.com")
    files = [("files", ("tone.wav", io.BytesIO(_tone()), "audio/wav"))]
    data = {"figures": "none", "ai_narrative": "false", "save_exam": "true", "site": "mitral",
            "recorded_at": "2026-11-10T08:30", "note": "seated"}
    assert client.post("/analyze", files=files, data=data).status_code == 200
    files = [("files", ("tone.wav", io.BytesIO(_tone()), "audio/wav"))]
    rep = client.post("/api/analyze", files=files, data={**data, "recorded_at": "2026-11-10T09:00", "site": "bogus"}).json()
    assert rep["results"][0]["saved_exam_id"]
    s = client.get("/api/exams").json()
    assert s["n_exams"] == 2 and s["n_days"] == 1
    assert [r["site"] for r in s["rows"]] == ["mitral", ""]
    assert s["rows"][0]["note"] == "seated"
    assert "audio_uri" not in str(s)
    csv = client.get("/exams/export.csv").text
    assert csv.splitlines()[0].startswith("id,recorded_at,date")
    exam_id = s["rows"][0]["id"]

    _register(client, "other@example.com")
    assert client.get("/api/exams").json()["n_exams"] == 0
    client.post(f"/exams/{exam_id}/delete")
    client.cookies.clear()
    client.post("/login", data={"email": "owner@example.com", "password": "correct horse"})
    assert client.get("/api/exams").json()["n_exams"] == 2
    client.post(f"/exams/{exam_id}/delete")
    assert client.get("/api/exams").json()["n_exams"] == 1


def test_save_requires_sign_in_for_api(client):
    client.cookies.clear()
    files = [("files", ("tone.wav", io.BytesIO(_tone()), "audio/wav"))]
    assert client.post("/api/analyze", files=files, data={"save_exam": "true", "figures": "none"}).status_code == 401
    assert client.get("/api/exams").status_code == 401


def test_delete_account_removes_exams(client):
    _register(client, "gone@example.com")
    files = [("files", ("tone.wav", io.BytesIO(_tone()), "audio/wav"))]
    client.post("/api/analyze", files=files, data={"save_exam": "true", "figures": "none", "ai_narrative": "false"})
    r = client.post("/account/delete", data={"password": "nope"}, follow_redirects=False)
    assert r.headers["location"] == "/exams?error=password"
    assert client.post("/account/delete", data={"password": "correct horse"}, follow_redirects=False).status_code == 303
    assert accounts.authenticate("gone@example.com", "correct horse") is None


def _exam(when, p_heart, hr=None, pred_type="N"):
    heads = [
        {"key": "heart_binary", "label": "Heart", "kind": "binary", "pred": "abnormal" if p_heart > 0.5 else "normal",
         "confidence": max(p_heart, 1 - p_heart), "classes": ["normal", "abnormal"], "probs": [1 - p_heart, p_heart]},
        {"key": "heart_types", "label": "Heart type", "kind": "multiclass", "pred": pred_type, "confidence": 0.5,
         "classes": ["N", "AS"], "probs": [0.5, 0.5]},
    ]
    return {"id": 1, "recorded_at": when, "quality_ok": True, "heads": heads, "metrics": {"heart_rate": hr} if hr else {}}


def test_summary_consistency_and_trend():
    exams = [_exam(f"2026-11-{d:02d}T{h:02d}:00", 0.9 - 0.08 * d + 0.01 * h, 90 - d) for d in range(1, 8) for h in (8, 12)]
    s = longitudinal.summarize(exams)
    assert s["n_exams"] == 14 and s["n_days"] == 7 and s["span_days"] == 7
    assert s["consistency"]["repeat_days"] == 7 and s["consistency"]["score"] > 90
    tr = s["trends"]["heart_binary"]["trend"]
    assert tr["direction"] == "falling" and tr["slope_per_week"] == pytest.approx(-0.56, abs=1e-3)
    assert s["trends"]["heart_rate"]["trend"]["direction"] == "falling"
    assert s["trends"]["heart_binary"]["vs_baseline"]["delta"] < 0
    assert s["days"][0]["types"]["heart_types"] == {"mode": "N", "share": 1.0, "n": 2}
    assert any("not a health" in line for line in s["narrative"])


def test_summary_disagreeing_repeats_score_low():
    exams = [_exam("2026-11-01T08:00", 0.1), _exam("2026-11-01T08:05", 0.9)]
    c = longitudinal.summarize(exams)["consistency"]
    assert c["agreement"] == 0.5 and c["score"] < 40


def test_summary_empty_and_single_day():
    assert longitudinal.summarize([])["consistency"] is None
    s = longitudinal.summarize([_exam("2026-11-01T08:00", 0.2)])
    assert s["consistency"] is None and s["trends"]["heart_binary"]["trend"] is None


def test_exam_from_report_extracts_metrics():
    rep = {"route": "heart", "verdict": "no-flag", "quality": {"duration_s": 6.0, "ok": True, "flags": []},
           "heads": [{"key": "heart_binary", "label": "H", "kind": "binary", "pred": "normal", "confidence": 0.8,
                      "classes": ["normal", "abnormal"], "probs": [0.8, 0.2], "probs_std": [0.05, 0.05]}],
           "series": {"heart_binary": {"probs": [[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.8, 0.2]]}},
           "quant": {"sections": [{"title": "Rhythm", "rows": [
               {"metric": "Heart rate estimate", "value": 72.4}, {"metric": "Breathing rate estimate", "value": 0.0},
               {"metric": "Beat interval variability (CV)", "value": "n/a"}]}]}}
    ex = longitudinal.exam_from_report(rep)
    assert ex["metrics"] == {"heart_rate": 72.4}
    assert ex["heads"][0]["window_agreement"] == 0.75 and ex["heads"][0]["fold_std"] == 0.05


def test_cross_site_posts_rejected(client):
    client.cookies.clear()
    r = client.post("/login", data={"email": "a@example.com", "password": "correct horse"},
                    headers={"origin": "https://evil.example"}, follow_redirects=False)
    assert r.status_code == 403
    r = client.post("/register", data={"email": "x@example.com", "password": "correct horse"},
                    headers={"origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 303


def test_login_throttled_after_repeated_failures(client, monkeypatch):
    monkeypatch.setattr(main, "_failed_logins", {})
    _register(client, "throttle@example.com")
    client.cookies.clear()
    for _ in range(main.LOGIN_MAX_FAILS["email"]):
        assert client.post("/login", data={"email": "throttle@example.com", "password": "bad-pass!"}).status_code == 401
    r = client.post("/login", data={"email": "throttle@example.com", "password": "correct horse"})
    assert r.status_code == 429


def test_expired_login_counters_are_pruned(monkeypatch):
    old = main.time.time() - main.LOGIN_WINDOW_S - 1
    store = {f"email:u{i}@example.com": [old] for i in range(main.LOGIN_PRUNE_AT + 5)}
    monkeypatch.setattr(main, "_failed_logins", store)
    assert not main._login_blocked(["email:new@example.com", "ip:1.2.3.4"])
    assert store == {}
