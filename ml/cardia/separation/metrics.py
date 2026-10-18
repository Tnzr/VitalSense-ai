"""Separation metrics (scale-invariant SDR)."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    est, ref = est.astype(np.float64) - est.mean(), ref.astype(np.float64) - ref.mean()
    a = float(est @ ref) / (float(ref @ ref) + eps)
    target = a * ref
    return float(10 * np.log10((target @ target + eps) / (((est - target) ** 2).sum() + eps)))


Separator = Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]]


def evaluate(separate: Separator, triplets: List, sr: int = 4000) -> Dict:
    """SI-SDR of each stem and improvement over the unprocessed mixture (SI-SDRi)."""
    rows = []
    for meta, m, h, l in triplets:
        eh, el = separate(m, sr)
        n = min(len(eh), len(h))
        rows.append({
            "mix_id": meta.mix_id,
            "heart_si_sdr": si_sdr(eh[:n], h[:n]), "lung_si_sdr": si_sdr(el[:n], l[:n]),
            "heart_si_sdri": si_sdr(eh[:n], h[:n]) - si_sdr(m[:n], h[:n]),
            "lung_si_sdri": si_sdr(el[:n], l[:n]) - si_sdr(m[:n], l[:n]),
        })
    keys = ["heart_si_sdr", "lung_si_sdr", "heart_si_sdri", "lung_si_sdri"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update({f"{k}_median": float(np.median([r[k] for r in rows])) for k in keys})
    return {"summary": summary, "per_clip": rows, "n": len(rows)}
