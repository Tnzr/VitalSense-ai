"""Inference core: warm-loaded 5-fold checkpoint ensembles, waveform
preprocessing, prediction, and figure rendering."""
from __future__ import annotations

import base64
import glob
import io
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import config

ML = config.ML
if ML not in sys.path:
    sys.path.insert(0, ML)

from cardia.config import Config  # noqa: E402
from cardia.data.transforms import MelSpec, band_pass, peak_normalize, _fixed_length  # noqa: E402
from cardia.models.cnn import build_cnn  # noqa: E402
from cardia.models.temporal import build_temporal  # noqa: E402
from cardia.models.transformer import build_transformer  # noqa: E402
from cardia.wandb_utils import viz  # noqa: E402


def _builder(stage: int):
    return {2: build_cnn, 3: build_temporal, 4: build_transformer}[stage]


def _ckpt_files(head: config.Head) -> List[str]:
    suf = f"-{head.scheme}" if head.scheme != "binary" else ""
    pats = [
        os.path.join(head.out_dir, "checkpoints", f"stage{head.stage}-{head.task}-{head.model}{suf}-allfolds-s*-fold*_best.pt"),
        os.path.join(head.out_dir, "checkpoints", f"stage{head.stage}-{head.task}-{head.model}{suf}-fold*-s*_best.pt"),
    ]
    return sorted({p for pat in pats for p in glob.glob(pat)})


class HeadModel:
    """An ensemble of fold checkpoints for one prediction head."""

    def __init__(self, head: config.Head, device: torch.device) -> None:
        self.head = head
        self.device = device
        self.cfg = Config(
            task=head.task, stage=head.stage, model=head.model, class_scheme=head.scheme,
            seed=config.SEED, data_dir=config.DATA_DIR, out_dir=head.out_dir, device=str(device),
        )
        self.mel = MelSpec(self.cfg.sample_rate, self.cfg.n_mels, self.cfg.n_fft, self.cfg.hop_length)
        self.models: List[nn.Module] = []
        self.ckpts: List[str] = []
        for ckpt in _ckpt_files(head):
            net = _builder(head.stage)(self.cfg, len(head.classes)).to(device)
            sd = torch.load(ckpt, map_location=device)["model_state"]
            if "_pos" in sd:  # ASTSmall learned positional embedding
                net._pos = nn.Parameter(sd["_pos"].to(device))
            net.load_state_dict(sd)
            net.eval()
            self.models.append(net)
            self.ckpts.append(ckpt)

    @property
    def ready(self) -> bool:
        return len(self.models) > 0

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        import torchaudio

        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        wav = peak_normalize(wav)
        wav = band_pass(wav, self.cfg.sample_rate, self.cfg.band_low, self.cfg.band_high)
        return _fixed_length(wav, self.cfg.max_samples)

    def predict(self, wav: torch.Tensor, sr: int) -> Dict:
        filt = self.preprocess(wav, sr)
        spec = self.mel(filt).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            folds = np.stack([torch.softmax(net(spec), dim=1).cpu().numpy()[0] for net in self.models]) \
                if self.models else np.full((1, len(self.head.classes)), 1.0 / len(self.head.classes))
        probs = folds.mean(axis=0)
        pred_idx = int(probs.argmax())
        return {
            "key": self.head.key,
            "label": self.head.label,
            "kind": self.head.kind,
            "classes": self.head.classes,
            "probs": probs.tolist(),
            "pred": self.head.classes[pred_idx],
            "pred_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probs_std": folds.std(axis=0).tolist(),
            "n_models": len(self.models),
            "filtered": filt,
        }

    def _prep_window(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        import torch.nn.functional as F
        import torchaudio

        if sr != self.cfg.sample_rate:
            w = torchaudio.functional.resample(w, sr, self.cfg.sample_rate)
        w = peak_normalize(w)
        w = band_pass(w, self.cfg.sample_rate, self.cfg.band_low, self.cfg.band_high)
        if self.head.stage == 4:
            # Transformer uses a fixed-length learned positional embedding
            w = _fixed_length(w, self.cfg.max_samples)
        elif w.shape[0] < self.cfg.n_fft:
            w = F.pad(w, (0, self.cfg.n_fft - w.shape[0]))
        return w

    def predict_series(self, wav: torch.Tensor, sr: int, window_s: float = 3.0, hop_s: float = 0.5) -> Dict:
        """Sweep the clip with a sliding window; return per-window head probs."""
        W = max(int(window_s * sr), 1)
        H = max(int(hop_s * sr), 1)
        n = int(wav.shape[0])
        starts = [0] if n <= W else list(range(0, n - W + 1, H))
        times, probs = [], []
        with torch.no_grad():
            for s in starts:
                w = wav[s:s + W]
                f = self._prep_window(w, sr)
                spec = self.mel(f).unsqueeze(0).unsqueeze(0).to(self.device)
                p = np.zeros(len(self.head.classes))
                for net in self.models:
                    p += torch.softmax(net(spec), dim=1).cpu().numpy()[0]
                p /= max(len(self.models), 1)
                times.append((s + W / 2.0) / sr)
                probs.append(p)
        return {
            "key": self.head.key,
            "label": self.head.label,
            "classes": self.head.classes,
            "times": times,
            "probs": np.asarray(probs).tolist(),
            "window_s": window_s,
            "hop_s": hop_s,
        }

    def predict_multiscale(self, wav: torch.Tensor, sr: int,
                           scales: List[float] = (1.0, 3.0, 15.0)) -> Dict:
        """Multi-resolution sweep: short windows localise micro events, longer
        windows capture macro context. Also returns the whole-clip (macro)
        distribution and a length-weighted fused prediction."""
        scale_series = []
        for sc in scales:
            hop = max(0.25, round(sc / 4.0, 2)) if sc < 15 else sc
            ser = self.predict_series(wav, sr, window_s=sc, hop_s=hop)
            scale_series.append({"scale_s": sc, "times": ser["times"], "probs": ser["probs"]})
        macro = self.predict(wav, sr)
        # length-weighted fusion: longer windows carry more weight (macro is most reliable)
        acc = np.zeros(len(self.head.classes))
        wsum = 0.0
        for s in scale_series:
            acc += s["scale_s"] * np.asarray(s["probs"]).mean(axis=0)
            wsum += s["scale_s"]
        fused = (acc / max(wsum, 1e-9))
        return {
            "key": self.head.key,
            "label": self.head.label,
            "classes": self.head.classes,
            "scales": scale_series,
            "macro": {"probs": macro["probs"], "pred": macro["pred"], "confidence": macro["confidence"]},
            "fused": {"probs": fused.tolist(), "pred": self.head.classes[int(fused.argmax())],
                      "confidence": float(fused.max())},
        }


class Registry:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or config.DEVICE)
        self.heads: Dict[str, HeadModel] = {}
        self.errors: Dict[str, str] = {}

    def load(self) -> None:
        for head in config.model_registry():
            try:
                self.heads[head.key] = HeadModel(head, self.device)
                if not self.heads[head.key].ready:
                    self.errors[head.key] = "no checkpoints found"
            except Exception as e:  # pragma: no cover - defensive
                self.errors[head.key] = f"{type(e).__name__}: {e}"

    def get(self, key: str) -> Optional[HeadModel]:
        return self.heads.get(key)

    def available(self) -> List[str]:
        return [k for k, h in self.heads.items() if h.ready]


# module-level singleton, loaded on startup
REGISTRY: Optional[Registry] = None


def get_registry() -> Registry:
    global REGISTRY
    if REGISTRY is None:
        backend = os.environ.get("ASCULTO_BACKEND", "torch").lower()
        if backend == "onnx":
            from .onnx_inference import OnnxRegistry

            REGISTRY = OnnxRegistry()
        else:
            REGISTRY = Registry()
        REGISTRY.load()
    return REGISTRY


# ---------------------------------------------------------------- figures
def fig_to_data_uri(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def head_prob_fig(probs, classes, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = np.asarray(probs, dtype=float)
    idx = int(p.argmax())
    colors = ["#4C72B0"] * len(classes)
    colors[idx] = "#C44E52"
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.8), 3.4))
    ax.bar(range(len(classes)), p, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylim(0, 1)
    labels = [c if len(c) <= 14 else c[:12] + ".." for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("class")
    ax.set_ylabel("post-softmax probability")
    ax.set_title(title)
    for i, v in enumerate(p):
        if v >= 0.02:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    return fig
