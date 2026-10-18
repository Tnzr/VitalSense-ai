"""Parametrized report builder: run requested heads, compute signal quality,
render figures, and compose plain-language health-awareness narrative."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from . import analysis, config
from . import llm as llm_mod
from .inference import fig_to_data_uri, get_registry, head_prob_fig

DEFAULT_HEADS = ["source", "heart_binary", "heart_types", "lung_binary", "lung_types"]


@dataclass
class ReportOptions:
    heads: List[str] = field(default_factory=lambda: list(DEFAULT_HEADS))
    explanation: str = "standard"  # brief | standard | detailed
    figures: str = "all"  # none | key | all
    include_disclaimer: bool = True
    audio_playback: bool = True
    temporal: bool = False
    window_s: float = 3.0
    hop_s: float = 0.5
    temporal_mode: str = "single"  # single | multiscale
    scales: List[float] = field(default_factory=lambda: [1.0, 3.0, 15.0])
    patient_context: str = ""
    ai_narrative: bool = True


def _quality(wav: torch.Tensor, sr: int) -> Dict:
    y = wav.numpy().astype(np.float64)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    clip_ratio = float(np.mean(np.abs(y) > 0.99))
    flags = []
    if sr > 8000:
        flags.append(f"recorded at {sr} Hz (phone/computer mic?) — models were trained on 4 kHz digital-stethoscope "
                     "clips, so expect lower reliability")
    if duration < 5:
        flags.append("short recording (<5 s) — results may be unreliable")
    if clip_ratio > 0.001:
        flags.append("possible clipping detected")
    if rms < 0.005:
        flags.append("very low signal level")
    return {
        "duration_s": round(duration, 2),
        "rms": round(rms, 5),
        "clip_ratio": round(clip_ratio, 6),
        "flags": flags,
        "ok": len(flags) == 0,
    }


def _narrative_for(head_key: str, pred: str, explanation: str) -> List[str]:
    finding, awareness = config.KNOWLEDGE.get(pred, ("", ""))
    lines = []
    if finding:
        lines.append(finding)
    if explanation in ("standard", "detailed") and awareness:
        lines.append(awareness)
    if explanation == "detailed":
        if pred.lower() in ("normal",) or pred == "Normal":
            lines.append("Keep monitoring if symptoms persist; no action is required based on this clip alone.")
        else:
            lines.append("Because the model flagged a possible abnormality, consider discussing this with a "
                         "healthcare professional, especially if you have symptoms.")
    return lines


def analyze_waveform(wav: torch.Tensor, sr: int, filename: str, options: ReportOptions) -> Dict:
    reg = get_registry()
    report: Dict = {
        "filename": filename,
        "sample_rate": sr,
        "quality": _quality(wav, sr),
        "heads": [],
        "figures": {},
        "narrative": [],
        "disclaimer": config.DISCLAIMER,
        "credits": config.DATASET_CREDIT,
        "requested_heads": options.heads,
        "explanation": options.explanation,
        "figures_level": options.figures,
    }

    # run requested heads
    for key in options.heads:
        head = reg.get(key)
        if head is None or not head.ready:
            continue
        out = head.predict(wav, sr)
        report["heads"].append({k: out[k] for k in
                                ("key", "label", "kind", "classes", "probs", "probs_std", "pred", "confidence",
                                 "n_models")})
        report.setdefault("_filtered", {})[key] = out["filtered"]

    report["viz"] = analysis.viz_payload(wav, sr)
    if not report["heads"]:
        report["quant"] = analysis.quantify(wav, sr)
        report["narrative"].append(
            "No prediction heads were available. Ensure model checkpoints exist and the service loaded them.")
        return report

    # processing figure base: prefer source routing band, else first head
    primary = report["heads"][0]
    filt = report["_filtered"].get(primary["key"])
    raw_np = wav.numpy()

    if options.figures != "none":
        from cardia.wandb_utils import viz

        report["figures"]["waveform"] = fig_to_data_uri(
            viz.waveform_fig(raw_np, sr, "Input — raw waveform"))
        report["figures"]["overlay"] = fig_to_data_uri(
            viz.overlay_fig(raw_np, filt.numpy(), sr, "Processing — raw vs band-passed"))
        from cardia.data.transforms import MelSpec

        mel = MelSpec(sr, 64, 400, 160)
        spec = mel(filt).numpy()
        report["figures"]["spectrogram"] = fig_to_data_uri(
            viz.spec_fig(spec, "Processing — log-mel spectrogram"))

    for h in report["heads"]:
        report["figures"][f"probs_{h['key']}"] = fig_to_data_uri(
            head_prob_fig(h["probs"], h["classes"], f"Output — {h['label']} (ensemble of {h['n_models']})"))

    # sliding-window temporal analysis (probability vs time, paired with log-mel)
    if options.temporal:
        from cardia.data.transforms import MelSpec
        from cardia.wandb_utils import viz

        mel = MelSpec(sr, 64, 400, 160)
        duration = float(wav.shape[0]) / sr
        report["series"] = {}
        report["temporal"] = {"window_s": options.window_s, "hop_s": options.hop_s,
                              "mode": options.temporal_mode}
        for h in report["heads"]:
            head = reg.get(h["key"])
            if head is None or not head.ready:
                continue
            filt_h = report.get("_filtered", {}).get(h["key"])
            spec_h = mel(filt_h).numpy() if filt_h is not None else mel(filt).numpy()
            if options.temporal_mode == "multiscale":
                ms = head.predict_multiscale(wav, sr, scales=tuple(options.scales))
                report["figures"][f"multiscale_{h['key']}"] = fig_to_data_uri(
                    viz.multiscale_fig(spec_h, duration, ms["scales"], ms["macro"]["probs"], h["classes"],
                                       f"{h['label']} — multi-scale probability vs time "
                                       f"(scales {', '.join(f'{s:g}s' for s in options.scales)})"))
                report.setdefault("multiscale", {})[h["key"]] = {
                    "fused": ms["fused"], "macro": ms["macro"],
                    "scales": [s["scale_s"] for s in ms["scales"]],
                }
                meso = ms["scales"][len(ms["scales"]) // 2]  # middle scale for the live view
                report["series"][h["key"]] = {
                    "label": h["label"], "classes": ms["classes"],
                    "times": meso["times"], "probs": meso["probs"],
                    "window_s": meso["scale_s"], "hop_s": options.hop_s,
                }
            else:
                ser = head.predict_series(wav, sr, options.window_s, options.hop_s)
                report["series"][h["key"]] = {
                    "label": h["label"], "classes": ser["classes"],
                    "times": ser["times"], "probs": ser["probs"],
                    "window_s": options.window_s, "hop_s": options.hop_s,
                }
                report["figures"][f"temporal_{h['key']}"] = fig_to_data_uri(
                    viz.temporal_paired_fig(
                        spec_h, duration, ser["times"], ser["probs"], h["classes"],
                        f"{h['label']} — probability vs time (window {options.window_s}s, hop {options.hop_s}s)"))
            # synchronized inference timeline (spectrogram + stacked probs + confidence)
            ser_live = report["series"][h["key"]]
            report["figures"][f"timeline_{h['key']}"] = fig_to_data_uri(
                viz.inference_timeline_fig(
                    spec_h, duration, ser_live["times"], ser_live["probs"], h["classes"],
                    f"{h['label']} — synchronized inference timeline"))

    # narrative
    source = next((h for h in report["heads"] if h["kind"] == "source"), None)
    if source:
        report["narrative"].append(
            f"Source routing: this clip was classified as **{source['pred']}** "
            f"(confidence {source['confidence']:.0%}).")
        for line in _narrative_for("source", source["pred"], options.explanation):
            report["narrative"].append(line)

    for h in report["heads"]:
        if h["kind"] == "source":
            continue
        report["narrative"].append(
            f"{h['label']}: **{h['pred']}** (confidence {h['confidence']:.0%}).")
        for line in _narrative_for(h["key"], h["pred"], options.explanation):
            report["narrative"].append(line)

    # overall summary
    abnormal = [h for h in report["heads"]
                if h["kind"] == "binary" and h["pred"] == "abnormal"]
    if abnormal:
        report["verdict"] = "abnormal-flag"
        report["narrative"].insert(0, "Summary: at least one screening head flagged a possible abnormality. "
                                      "Please read the disclaimers and consider professional review.")
    else:
        report["verdict"] = "no-flag"
        report["narrative"].insert(0, "Summary: no screening head flagged an abnormality. This does not rule "
                                      "out disease.")

    route = source["pred"] if source else None
    order = ["lung_types", "lung_binary", "heart_types", "heart_binary"] if route == "lung" else \
        ["heart_types", "heart_binary", "lung_types", "lung_binary"]
    routed = {"heart": {"heart_types", "heart_binary"}, "lung": {"lung_types", "lung_binary"}}.get(route, set(order))
    present = {h["key"] for h in report["heads"]}
    report["route"] = route
    report["panels"] = [{"key": k, "routed": k in routed} for k in order if k in present]
    report["knowledge"] = {c: config.KNOWLEDGE[c] for h in report["heads"] for c in h["classes"]
                           if c in config.KNOWLEDGE}
    report["quant"] = analysis.quantify(wav, sr, route, report["heads"], report.get("series"))

    # AI personalized narrative (LLM when configured, else offline synthesizer)
    findings_payload = [
        {k: h[k] for k in ("key", "label", "kind", "pred", "confidence", "classes", "probs")}
        for h in report["heads"]
    ]
    report["findings_payload"] = findings_payload
    report["patient_context"] = options.patient_context
    if options.ai_narrative:
        report["ai_narrative"] = llm_mod.generate_narrative(findings_payload, options.patient_context)

    report.pop("_filtered", None)
    return report


def analyze_file(path: str, options: ReportOptions) -> Dict:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data[:, 0]).squeeze()
    import os

    return analyze_waveform(wav, int(sr), os.path.basename(path), options)


def batch_row(report: Dict) -> Dict:
    source = next((h for h in report["heads"] if h["kind"] == "source"), None)
    screen = [h for h in report["heads"] if h["kind"] == "binary"]
    typing = [h for h in report["heads"] if h["kind"] == "multiclass"]
    top_type = max(typing, key=lambda h: h["confidence"]) if typing else None
    return {
        "filename": report["filename"],
        "source": source["pred"] if source else "-",
        "source_conf": round(source["confidence"], 3) if source else None,
        "screening": ", ".join(f"{h['label'].split(' ')[0]}:{h['pred']}" for h in screen) or "-",
        "top_type": top_type["pred"] if top_type else "-",
        "top_type_conf": round(top_type["confidence"], 3) if top_type else None,
        "verdict": report.get("verdict", "-"),
        "quality_ok": report["quality"]["ok"],
        "quality_flags": "; ".join(report["quality"]["flags"]),
    }
