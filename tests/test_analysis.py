import base64

import numpy as np
import torch

from webapp.app import analysis


def _beats(bpm: float, sr: int = 4000, seconds: float = 15.0) -> torch.Tensor:
    t = np.arange(int(sr * seconds)) / sr
    y = np.zeros_like(t)
    for onset in np.arange(0.2, seconds, 60.0 / bpm):
        m = (t >= onset) & (t < onset + 0.08)
        y[m] += np.sin(2 * np.pi * 60 * (t[m] - onset)) * np.hanning(m.sum())
    return torch.from_numpy((0.5 * y).astype(np.float32))


def _value(q, metric):
    for sec in q["sections"]:
        for r in sec["rows"]:
            if r["metric"] == metric:
                return r["value"]
    raise KeyError(metric)


def test_heart_rate_estimate_on_synthetic_beats():
    q = analysis.quantify(_beats(72), 4000, source="heart")
    assert abs(_value(q, "Heart rate estimate") - 72) < 4
    assert _value(q, "Cardiac periodicity") > 0.5
    assert abs(sum(b["pct"] for b in q["bands"]) - 100) < 1


def test_viz_payload_shapes_and_resampling():
    p = analysis.viz_payload(_beats(60, sr=48000, seconds=3.0), 48000)
    mel = p["mel"]
    assert p["sr"] == 4000 and abs(p["duration"] - 3.0) < 0.01
    assert len(base64.b64decode(mel["data"])) == mel["n_mels"] * mel["frames"]
    assert mel["n_mels"] == len(mel["hz"]) == 64
    assert len(p["wave"]["min"]) == len(p["wave"]["max"]) <= 1200


def test_certainty_rows_use_fold_spread_and_series():
    head = {"key": "heart_types", "label": "Heart typing", "classes": ["A", "B", "C"], "pred": "A",
            "probs": [0.7, 0.2, 0.1], "probs_std": [0.1, 0.05, 0.05], "n_models": 5}
    series = {"heart_types": {"probs": [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]}}
    q = analysis.quantify(_beats(80), 4000, source="heart", heads=[head], series=series)
    assert _value(q, "Heart typing") == 70.0
    note = [r for s in q["sections"] for r in s["rows"] if r["metric"] == "Heart typing"][0]["note"]
    assert "± 10.0% across 5 folds" in note and "margin 0.50" in note and "50% of windows agree" in note


def test_periodicity_long_envelope_is_fast():
    import time

    env_sr = 50
    t = np.arange(30 * 60 * env_sr) / env_sr  # 30-minute envelope, 90k points
    env = np.cos(2 * np.pi * (72 / 60) * t)
    t0 = time.perf_counter()
    rate, strength = analysis._periodicity(env, env_sr, 40, 200)
    assert time.perf_counter() - t0 < 2.0
    assert abs(rate - 72) < 2 and strength > 0.9


def test_lung_normal_has_lung_description():
    from webapp.app import config

    assert "breath" in config.knowledge_for("lung_types", "Normal")[1]
    assert "S1" in config.knowledge_for("heart_types", "Normal")[1]
