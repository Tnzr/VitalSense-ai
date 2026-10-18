"""Sound matrix: one educational tile per heart/lung condition class."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from . import analysis, config, game

DOMAINS = [
    ("heart_types", "Heart sounds", "heart", "10class"),
    ("lung_types", "Lung sounds", "lung", "6class"),
]

# (what to listen for, what to look for in the log-mel plot)
CUES: Dict[str, Dict[str, tuple]] = {
    "heart_types": {
        "Normal": ("Two crisp sounds per beat — S1 'lub', S2 'dub' — with quiet gaps between.",
                   "Evenly spaced pairs of short vertical bursts below ~200 Hz."),
        "Late Diastolic Murmur": ("Low rumble just before S1 (presystolic), as in mitral stenosis.",
                                  "Low-frequency smear filling the gap right before each S1 burst."),
        "Mid Systolic Murmur": ("Crescendo–decrescendo 'diamond' between S1 and S2 (ejection murmur).",
                                "Diamond-shaped energy between S1 and S2, often up to 300–400 Hz."),
        "Late Systolic Murmur": ("Murmur starting mid/late systole and running up to S2, often after a click.",
                                 "Energy that builds in the second half of systole and stops at S2."),
        "Atrial Fibrillation": ("Irregularly irregular beat spacing; S1 loudness varies beat to beat.",
                                "Beat bursts at uneven intervals — no repeating pattern."),
        "S3": ("Extra low-pitched thud shortly after S2 — 'Ken-tuc-ky' cadence.",
               "A faint third low-frequency blob just after each S2."),
        "S4": ("Extra low-pitched sound just before S1 — 'Ten-nes-see' cadence.",
               "A faint low-frequency blob immediately preceding each S1."),
        "Early Systolic Murmur": ("Short murmur right after S1 that fades well before S2.",
                                  "Energy trailing S1 that dies out early in systole."),
        "Tachycardia": ("Regular but fast rhythm (>100 bpm); diastole shortens so beats crowd together.",
                        "Tightly packed, regularly spaced beat bursts."),
        "AV Block": ("Slow or dropped beats with long pauses.",
                     "Long silent gaps between beat bursts."),
    },
    "lung_types": {
        "Normal": ("Soft, low-pitched rustle; inspiration longer than expiration; no added sounds.",
                   "Broad, gentle breathing envelopes below ~600 Hz, no streaks or bands."),
        "Wheezing": ("Continuous musical, high-pitched whistle, usually on expiration.",
                     "Horizontal tonal bands (narrow lines) lasting hundreds of ms."),
        "Fine Crackles": ("Brief, high-pitched pops late in inspiration — like hair rubbed near the ear.",
                          "Many thin vertical streaks reaching high frequencies."),
        "Coarse Crackles": ("Louder, lower-pitched bubbling pops in early inspiration and expiration.",
                            "Fewer, broader vertical streaks with more low-frequency energy."),
        "Rhonchi": ("Low-pitched snoring/gurgling continuous sound; may clear with a cough.",
                    "Low-frequency horizontal bands, often below ~200 Hz."),
        "Pleural Rub": ("Creaking, leathery grating in both breathing phases — like walking on snow.",
                        "Irregular rough bursts in both phases, mid-frequency texture."),
    },
}


def tiles(per_class: int = 3) -> List[Dict]:
    """Per-domain list of tiles with example sample ids and teaching cues."""
    out = []
    for key, title, task, scheme in DOMAINS:
        try:
            m = game._manifest(task, scheme)
        except Exception as e:  # dataset not downloaded
            out.append({"key": key, "title": title, "task": task, "scheme": scheme, "tiles": [],
                        "error": f"{type(e).__name__}: {e}"})
            continue
        rows = []
        for label in game._classes_for(key, m):
            sub = m[m["task_label"] == label]
            if "is_clean" in sub.columns:
                sub = sub.sort_values("is_clean", ascending=False, kind="stable")
            listen, look = CUES.get(key, {}).get(label, ("", ""))
            rows.append({
                "label": label,
                "count": int(len(sub)),
                "sample_ids": [str(s) for s in sub["sample_id"].head(per_class)],
                "listen": listen,
                "look": look,
                "finding": config.KNOWLEDGE.get(label, ("", ""))[1] if label != "Normal" else "",
            })
        out.append({"key": key, "title": title, "task": task, "scheme": scheme, "tiles": rows})
    return out


@lru_cache(maxsize=256)
def mel_for(sample_id: str, task: str, scheme: str) -> Optional[Dict]:
    import soundfile as sf
    import torch

    path = game.resolve_audio_path(sample_id, task, scheme)
    if not path:
        return None
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return analysis.viz_payload(torch.from_numpy(data[:, 0].copy()), int(sr))
