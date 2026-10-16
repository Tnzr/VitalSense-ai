"""Dataset statistics and signal-quality indicators (SQI) for dry runs/notebooks."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from cardia.data.transforms import load_wav


def class_distribution(manifest: pd.DataFrame) -> pd.DataFrame:
    """Counts per task_label x source, plus a ``total`` column."""
    table = pd.crosstab(manifest["task_label"], manifest["source"])
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def recording_table(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("sample_id", "group", "source", "task_label", "target", "gender", "location")
            if c in manifest.columns]
    return manifest[cols].reset_index(drop=True)


def duration_stats(manifest: pd.DataFrame) -> List[float]:
    import soundfile as sf

    return [float(sf.info(p).duration) for p in manifest["file_path"]]


def _sqi(wav: np.ndarray, sr: int) -> dict:
    frame = max(int(0.05 * sr), 1)
    n = len(wav) // frame
    energy = (wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12 if n else np.array([1e-12])
    snr = 10.0 * np.log10(np.percentile(energy, 90) / np.percentile(energy, 10))
    spec = np.abs(np.fft.rfft(wav)) ** 2 + 1e-12
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    return {
        "duration_s": len(wav) / float(sr),
        "rms": float(np.sqrt(np.mean(wav ** 2))),
        "snr_estimate": float(snr),
        "flatness": flatness,
        "clip_ratio": float(np.mean(np.abs(wav) >= 0.999)),
    }


def sqi_stats(manifest: pd.DataFrame, n: int = 50, seed: int = 0) -> pd.DataFrame:
    """SQI on up to ``n`` recordings: duration, RMS, SNR estimate (dB), flatness, clip ratio."""
    sub = manifest.sample(n=min(n, len(manifest)), random_state=seed) if n < len(manifest) else manifest
    rows = []
    for _, r in sub.iterrows():
        wav, sr = load_wav(r["file_path"])
        rows.append({"sample_id": r["sample_id"], "task_label": r["task_label"], **_sqi(wav.numpy(), sr)})
    return pd.DataFrame(rows)
