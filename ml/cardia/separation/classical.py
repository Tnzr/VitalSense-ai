"""Training-free heart/lung separation.

Heart sounds repeat beat-to-beat and sit mostly below ~200 Hz; breath sounds are broadband
and follow a slower respiratory cycle. Two cues are combined into a soft STFT mask for the
heart; the lung estimate is the residual, so ``heart + lung == mixture``:

* ``band``      - low-frequency prior (pass < ``lo`` Hz, cosine roll-off to ``hi`` Hz)
* ``repet``     - REPET: median of frames one beat period apart (period from envelope autocorrelation)
* ``repet_sim`` - REPET-SIM: median of the most similar frames (tolerates irregular rhythm, e.g. AF)
* ``floor``     - remove the per-frequency stationary floor (room/manikin rumble that both stems
  share) before REPET, and split the floor itself evenly
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, istft, sosfiltfilt, stft

N_FFT, HOP = 256, 32


def _spec(x: np.ndarray, sr: int):
    f, _, Z = stft(x, sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
    return f, Z


def _inverse(Z: np.ndarray, sr: int, n: int) -> np.ndarray:
    _, y = istft(Z, sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def band_prior(f: np.ndarray, lo: float = 150.0, hi: float = 400.0) -> np.ndarray:
    w = np.clip((hi - f) / (hi - lo), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * w))[:, None]


def beat_period(x: np.ndarray, sr: int, bpm: Tuple[float, float] = (40.0, 200.0)) -> float:
    """Dominant beat period (s) from the 20-150 Hz envelope autocorrelation."""
    y = sosfiltfilt(butter(4, [20, 150], btype="band", fs=sr, output="sos"), x)
    env_sr = 100
    env = np.abs(y)[: len(y) // (sr // env_sr) * (sr // env_sr)].reshape(-1, sr // env_sr).mean(1)
    env = env - env.mean()
    nfft = 1 << int(2 * env.size - 1).bit_length()
    s = np.fft.rfft(env, nfft)
    ac = np.fft.irfft(s * np.conj(s), nfft)[: env.size]
    lo, hi = int(env_sr * 60 / bpm[1]), min(int(env_sr * 60 / bpm[0]), env.size - 1)
    if hi <= lo:
        return 1.0
    return (lo + int(np.argmax(ac[lo:hi]))) / env_sr


def _repet_model(V: np.ndarray, period_frames: int) -> np.ndarray:
    T = V.shape[1]
    out = np.empty_like(V)
    for t in range(T):
        idx = np.arange(t % period_frames, T, period_frames)
        out[:, t] = np.median(V[:, idx], axis=1)
    return np.minimum(out, V)


def _repet_sim_model(V: np.ndarray, f: np.ndarray, k: int = 10, min_gap: int = 8,
                     fmax: float = 250.0) -> np.ndarray:
    feat = np.log1p(V[f <= fmax] / (V.mean() + 1e-12))
    feat = feat - feat.mean(0, keepdims=True)
    feat /= np.linalg.norm(feat, axis=0, keepdims=True) + 1e-9
    S = feat.T @ feat
    T = V.shape[1]
    out = np.empty_like(V)
    for t in range(T):
        order = np.argsort(-S[t])
        picked = []
        for j in order:
            if all(abs(j - p) >= min_gap for p in picked):
                picked.append(j)
                if len(picked) == k:
                    break
        out[:, t] = np.median(V[:, picked], axis=1)
    return np.minimum(out, V)


def separate(x: np.ndarray, sr: int, method: str = "floor_repet_sim",
             lo: float = 150.0, hi: float = 400.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (heart, lung) estimates for a mono waveform."""
    x = np.asarray(x, dtype=np.float64)
    f, Z = _spec(x, sr)
    V = np.abs(Z) + 1e-12
    floor = np.percentile(V, 20, axis=1, keepdims=True) if "floor" in method else np.zeros((V.shape[0], 1))
    D = np.maximum(V - floor, 1e-12)
    mask = np.ones_like(V)
    if "repet_sim" in method:
        mask = _repet_sim_model(D, f) / D
    elif "repet" in method:
        mask = _repet_model(D, max(1, int(round(beat_period(x, sr) * sr / HOP)))) / D
    if "floor" in method:
        mask = (mask * D + 0.5 * np.minimum(floor, V)) / V
    if "band" in method:
        mask = mask * band_prior(f, lo, hi)
    heart = _inverse(Z * mask, sr, len(x))
    return heart, (x - heart).astype(np.float32)
