"""Quantified acoustic analysis and browser visualization payloads for one clip.

Everything here is computed from the audio itself (no model), so it works even
when no checkpoints are loaded. Metrics are descriptive estimates for education
and research — not clinical measurements.
"""
from __future__ import annotations

import base64
from typing import Dict, List, Optional

import numpy as np
import torch

VIZ_SR = 4000
_MEL_BINS = 64
_MEL_NFFT = 256
_MEL_HOP = 80  # 20 ms at 4 kHz
_DB_RANGE = 80.0
_WAVE_POINTS = 1200
MAX_SECONDS = 120  # explorer + quantified metrics cover at most the first 2 minutes

BANDS = [
    (20, 150, "Low 20–150 Hz", "S1/S2, S3/S4 gallops, low rumbles"),
    (150, 400, "Low-mid 150–400 Hz", "most murmurs, rhonchi"),
    (400, 1000, "Mid 400–1000 Hz", "breath sounds, wheezes"),
    (1000, 2000, "High 1–2 kHz", "crackles, high-pitched wheeze"),
]


def to_viz_rate(wav: torch.Tensor, sr: int) -> np.ndarray:
    """Mono float64 at VIZ_SR, DC-removed and peak-normalized."""
    import torchaudio

    w = wav.detach().float().reshape(-1)[:int(MAX_SECONDS * sr)]
    if sr != VIZ_SR:
        w = torchaudio.functional.resample(w, sr, VIZ_SR)
    y = w.numpy().astype(np.float64)
    y = y - y.mean() if y.size else y
    peak = np.max(np.abs(y)) if y.size else 0.0
    return y / peak if peak > 0 else y


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def viz_payload(wav: torch.Tensor, sr: int) -> Dict:
    """Log-mel (dB, uint8-quantized, base64) + min/max waveform envelope."""
    import torchaudio

    y = to_viz_rate(wav, sr)
    if y.size < _MEL_NFFT:
        y = np.pad(y, (0, _MEL_NFFT - y.size))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=VIZ_SR, n_fft=_MEL_NFFT, hop_length=_MEL_HOP, n_mels=_MEL_BINS,
        f_min=0.0, f_max=VIZ_SR / 2.0, power=2.0, center=True,
    )(torch.from_numpy(y.astype(np.float32))).numpy()
    db = 10.0 * np.log10(mel + 1e-10)
    top = float(db.max())
    lo = top - _DB_RANGE
    q = np.clip((db - lo) / _DB_RANGE, 0.0, 1.0)
    q = np.round(q * 255).astype(np.uint8)  # (mels, frames), row 0 = lowest band
    centers = _mel_to_hz(np.linspace(_hz_to_mel(0), _hz_to_mel(VIZ_SR / 2), _MEL_BINS + 2)[1:-1])

    n = y.size
    edges = np.linspace(0, n, min(_WAVE_POINTS, n) + 1).astype(int)
    wmin = [round(float(y[a:b].min()), 3) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    wmax = [round(float(y[a:b].max()), 3) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return {
        "sr": VIZ_SR,
        "duration": n / VIZ_SR,
        "source_duration": wav.reshape(-1).shape[0] / sr,
        "mel": {
            "n_mels": int(q.shape[0]), "frames": int(q.shape[1]),
            "hop_s": _MEL_HOP / VIZ_SR, "db_min": round(lo, 1), "db_max": round(top, 1),
            "hz": [round(float(c), 1) for c in centers],
            "data": base64.b64encode(q.tobytes()).decode(),
        },
        "wave": {"min": wmin, "max": wmax},
    }


def _bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    hi = min(hi, sr / 2 * 0.95)
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def _envelope(y: np.ndarray, sr: int, lo: float, hi: float, env_sr: int) -> np.ndarray:
    from scipy.signal import hilbert

    e = np.abs(hilbert(_bandpass(y, sr, lo, hi)))
    step = max(sr // env_sr, 1)
    n = (e.size // step) * step
    e = e[:n].reshape(-1, step).mean(axis=1)
    return e - e.mean()


def _periodicity(env: np.ndarray, env_sr: int, min_rate: float, max_rate: float):
    """Autocorrelation rate estimate (cycles/min) and its normalized peak strength."""
    if env.size < 4 or not np.any(env):
        return None, 0.0
    nfft = 1 << int(2 * env.size - 1).bit_length()
    spec = np.fft.rfft(env, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[:env.size]  # O(n log n) autocorrelation
    ac = ac / (ac[0] + 1e-12)
    lag_lo = max(int(env_sr * 60.0 / max_rate), 1)
    lag_hi = min(int(env_sr * 60.0 / min_rate), ac.size - 1)
    if lag_hi <= lag_lo:
        return None, 0.0
    lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi + 1]))
    return 60.0 * env_sr / lag, float(max(ac[lag], 0.0))


def _interval_cv(env: np.ndarray, env_sr: int, rate: Optional[float]) -> Optional[float]:
    from scipy.signal import find_peaks

    if not rate:
        return None
    period = env_sr * 60.0 / rate
    peaks, _ = find_peaks(env, distance=max(int(0.6 * period), 1), height=env.mean() + 0.5 * env.std())
    if peaks.size < 4:
        return None
    iv = np.diff(peaks) / env_sr
    return float(iv.std() / (iv.mean() + 1e-12))


def _row(metric: str, value, unit: str = "", note: str = "") -> Dict:
    if isinstance(value, float):
        value = round(value, 1 if unit in ("%", "dB", "dBFS", "Hz", "bpm", "breaths/min") else 3)
    return {"metric": metric, "value": value, "unit": unit, "note": note}


def quantify(wav: torch.Tensor, sr: int, source: Optional[str] = None,
             heads: Optional[List[Dict]] = None, series: Optional[Dict] = None) -> Dict:
    """Automated quantified report: signal, spectral, rhythm and model-certainty metrics."""
    from scipy.signal import welch

    raw = wav.detach().float().reshape(-1).numpy().astype(np.float64)
    y = to_viz_rate(wav, sr)
    sections = []

    rms = float(np.sqrt(np.mean(raw ** 2))) if raw.size else 0.0
    frame = max(int(0.05 * VIZ_SR), 1)
    nfr = y.size // frame
    energies = (y[:nfr * frame].reshape(nfr, frame) ** 2).mean(axis=1) if nfr else np.array([0.0])
    snr = 10 * np.log10((np.percentile(energies, 90) + 1e-12) / (np.percentile(energies, 10) + 1e-12))
    sections.append({"title": "Signal", "rows": [
        _row("Duration", raw.size / sr if sr else 0.0, "s"),
        *([_row("Analysed span", y.size / VIZ_SR, "s", f"first {MAX_SECONDS} s; later audio is not measured")]
          if sr and raw.size / sr > MAX_SECONDS else []),
        _row("Input sample rate", int(sr), "Hz",
             "phone/computer mic rate — model was trained on 4 kHz stethoscope clips" if sr > 8000 else ""),
        _row("RMS level", 20 * np.log10(rms + 1e-12), "dBFS"),
        _row("Peak level", 20 * np.log10(float(np.max(np.abs(raw))) + 1e-12) if raw.size else -240.0, "dBFS"),
        _row("Clipped samples", 100 * float(np.mean(np.abs(raw) > 0.99)) if raw.size else 0.0, "%"),
        _row("Dynamic SNR estimate", float(snr), "dB", "loud vs quiet 50 ms frames (p90/p10)"),
    ]})

    f, p = welch(y, fs=VIZ_SR, nperseg=min(1024, max(y.size, 8)))
    sel = (f >= 20) & (f <= 2000)
    fs_, ps = f[sel], p[sel]
    total = float(ps.sum()) + 1e-20
    bands = []
    for lo, hi, label, note in BANDS:
        m = (fs_ >= lo) & (fs_ < hi)
        bands.append({"label": label, "lo": lo, "hi": hi, "note": note, "pct": round(100 * float(ps[m].sum()) / total, 1)})
    cum = np.cumsum(ps) / total
    centroid = float((fs_ * ps).sum() / total)
    rolloff = float(fs_[min(int(np.searchsorted(cum, 0.85)), fs_.size - 1)]) if fs_.size else 0.0
    dominant = float(fs_[int(np.argmax(ps))]) if fs_.size else 0.0
    flatness = float(np.exp(np.mean(np.log(ps + 1e-20))) / (np.mean(ps) + 1e-20)) if ps.size else 0.0
    sections.append({"title": "Spectrum", "rows": [
        _row("Dominant frequency", dominant, "Hz"),
        _row("Spectral centroid", centroid, "Hz", "centre of mass of the spectrum"),
        _row("85% roll-off", rolloff, "Hz", "85% of energy lies below this"),
        _row("Spectral flatness", flatness, "", "0 = tonal (e.g. wheeze), 1 = noise-like"),
    ] + [_row(f"Energy {b['label']}", b["pct"], "%", b["note"]) for b in bands]})

    rhythm = []
    if y.size >= VIZ_SR * 3:
        if source in (None, "heart", "mixed"):
            env = _envelope(y, VIZ_SR, 25, 150, 50)
            hr, strength = _periodicity(env, 50, 40, 200)
            cv = _interval_cv(env, 50, hr)
            rhythm += [
                _row("Heart rate estimate", hr or 0.0, "bpm", "autocorrelation of the 25–150 Hz envelope"),
                _row("Cardiac periodicity", strength, "", "0–1; higher = more regular cycle"),
                _row("Beat interval variability (CV)", cv if cv is not None else "n/a", "",
                     "high values suggest an irregular rhythm"),
            ]
        if source in (None, "lung", "mixed"):
            env = _envelope(y, VIZ_SR, 150, 1000, 10)
            br, strength = _periodicity(env, 10, 6, 40)
            rhythm += [
                _row("Breathing rate estimate", br or 0.0, "breaths/min", "autocorrelation of the 150–1000 Hz envelope"),
                _row("Respiratory periodicity", strength, "", "0–1; higher = more regular cycle"),
            ]
    if rhythm:
        sections.append({"title": "Rhythm", "rows": rhythm})

    certainty = []
    for h in heads or []:
        probs = np.asarray(h["probs"], dtype=float)
        k = probs.size
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        ent = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(k)) if k > 1 else 0.0
        margin = float(probs[top] - probs[order[1]]) if k > 1 else 1.0
        parts = [h["pred"]]
        std = h.get("probs_std")
        if std:
            parts.append(f"± {100 * float(std[top]):.1f}% across {h.get('n_models', '?')} folds")
        parts += [f"margin {margin:.2f}", f"entropy {ent:.2f}"]
        ser = (series or {}).get(h["key"])
        if ser and ser.get("probs"):
            arg = np.asarray(ser["probs"]).argmax(axis=1)
            parts.append(f"{100 * float(np.mean(arg == top)):.0f}% of windows agree")
        certainty.append(_row(h["label"], 100 * float(probs[top]), "%", " · ".join(parts)))
    if certainty:
        sections.append({"title": "Model certainty", "rows": certainty})

    return {"sections": sections, "bands": bands}
