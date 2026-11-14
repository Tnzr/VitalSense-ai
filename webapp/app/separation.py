"""Heart / lung source separation for reports.

Runs the trained mask separator (``ml/cardia/separation``) and re-analyzes each stem with
its own domain's heads, so a mixed recording yields both a heart and a lung condition read.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from . import analysis, playback

STEM_HEADS = {"heart": ("heart_types", "heart_binary"), "lung": ("lung_types", "lung_binary")}
SEP_SR = 4000


def checkpoint() -> Optional[str]:
    from cardia.separation.model import default_checkpoint
    path = default_checkpoint()
    return path if os.path.isfile(path) else None


def _audio_uri(x: np.ndarray, sr: int) -> Optional[str]:
    try:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, playback.to_playback(x, sr), playback.PLAYBACK_SR, format="WAV", subtype="PCM_16")
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def separate_report(wav: torch.Tensor, sr: int, reg, heads: List[str], audio: bool = True) -> Dict:
    """``{"stems": {"heart": {...}, "lung": {...}}, ...}`` or ``{"error": ...}``."""
    path = checkpoint()
    if not path:
        return {"error": "No separation checkpoint found (train with `python -m cardia.separation.train`)."}
    from cardia.separation.model import load_separator

    x = wav.detach().float().reshape(-1)[: int(analysis.MAX_SECONDS * sr)].numpy()
    heart, lung = load_separator(path)(x, sr)
    total = float((heart ** 2).sum() + (lung ** 2).sum()) or 1.0
    out: Dict = {"method": "mask separator (conv + BiGRU, trained on synthetic HLS-CMDS mixtures)",
                 "seconds": len(heart) / SEP_SR, "stems": {}}
    for name, y in (("heart", heart), ("lung", lung)):
        t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        preds = []
        for key in STEM_HEADS[name]:
            head = reg.get(key)
            if key not in heads or head is None or not head.ready:
                continue
            p = head.predict(t, SEP_SR)
            preds.append({k: p[k] for k in ("key", "label", "kind", "classes", "probs", "pred", "confidence")}
                         | {"probs_std": p.get("probs_std"),
                            "top": sorted(zip(p["classes"], p["probs"]), key=lambda cp: -cp[1])[:3]})
        out["stems"][name] = {
            "energy_share": float((y ** 2).sum()) / total,
            "viz": analysis.viz_payload(t, SEP_SR),
            "audio_uri": _audio_uri(y, SEP_SR) if audio else None,
            "heads": preds,
        }
    return out
