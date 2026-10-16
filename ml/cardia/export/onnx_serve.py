"""Serving-oriented ONNX export for the Asculto heads.

Unlike `export/mobile.py` (which benchmarks a *single* fold with naive dynamic
INT8), this exports **one artifact per head** wrapping the full 5-fold ensemble
(post-softmax average) and applies the quantization that is actually fast on
CPU:

  * ResNet-18 heads  -> FP16 (dynamic INT8 regresses CPU latency, see
                        docs/MobileCompute.md: 8.4 ms -> 30.3 ms).
  * Transformer heads -> dynamic INT8 (weights dominate; ~2 MB, faster on CPU).

Static QDQ is available via ``--quant-cnn static_qdq`` when a calibration set of
log-mel frames is supplied.

Outputs (``--out-dir``):
  onnx/<head>_fp32.onnx, onnx/<head>_<quant>.onnx
  onnx_manifest.json  (classes, input spec, per-file sha256 + bytes, validation)

Usage:
  python -m cardia.export.onnx_serve --binary-out runs --mc-out <mc> --out-dir runs/onnx_serve
  python -m cardia.export.onnx_serve --validate-clips /path/to/wavs --quant-cnn static_qdq \
      --calib-dir /path/to/wavs
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ML not in sys.path:
    sys.path.insert(0, ML)

from cardia.config import DEFAULT_MC_OUT, Config, HEART_TYPES, LUNG_TYPES  # noqa: E402
from cardia.models.cnn import build_cnn  # noqa: E402
from cardia.models.temporal import build_temporal  # noqa: E402
from cardia.models.transformer import build_transformer  # noqa: E402

T_FRAMES = 376  # 15 s @ 4 kHz, n_fft 400 / hop 160

HEADS = [
    dict(key="source", task="sound", scheme="binary", stage=2, model="resnet18", classes=3),
    dict(key="heart_binary", task="heart", scheme="binary", stage=2, model="resnet18", classes=2),
    dict(key="heart_types", task="heart", scheme="10class", stage=4, model="transformer",
         classes=len(HEART_TYPES)),
    dict(key="lung_binary", task="lung", scheme="binary", stage=2, model="resnet18", classes=2),
    dict(key="lung_types", task="lung", scheme="6class", stage=4, model="transformer",
         classes=len(LUNG_TYPES)),
]

_CNN_STAGES = {2, 3}


def _builder(stage: int):
    return {2: build_cnn, 3: build_temporal, 4: build_transformer}[stage]


def _ckpt_files(out_dir: str, head: dict) -> List[str]:
    suf = f"-{head['scheme']}" if head["scheme"] != "binary" else ""
    pats = [
        os.path.join(out_dir, "checkpoints",
                     f"stage{head['stage']}-{head['task']}-{head['model']}{suf}-allfolds-s*-fold*_best.pt"),
        os.path.join(out_dir, "checkpoints",
                     f"stage{head['stage']}-{head['task']}-{head['model']}{suf}-fold*-s*_best.pt"),
    ]
    return sorted({p for pat in pats for p in glob.glob(pat)})


class Ensemble(nn.Module):
    """Average post-softmax probabilities across fold models (serving contract)."""

    def __init__(self, nets: List[nn.Module]) -> None:
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = None
        for net in self.nets:
            p = torch.softmax(net(x), dim=1)
            acc = p if acc is None else acc + p
        return acc / max(len(self.nets), 1)


def _load_ensemble(head: dict, ckpts: List[str], device: torch.device) -> Ensemble:
    nets = []
    for ckpt in ckpts:
        cfg = Config(task=head["task"], stage=head["stage"], model=head["model"],
                     class_scheme=head["scheme"], data_dir="", out_dir="")
        net = _builder(head["stage"])(cfg, head["classes"]).to(device)
        sd = torch.load(ckpt, map_location=device)["model_state"]
        if "_pos" in sd:
            net._pos = nn.Parameter(sd["_pos"].to(device))
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    return Ensemble(nets).to(device).eval()


def _export_fp32(net: nn.Module, path: str) -> None:
    x = torch.randn(1, 1, 64, T_FRAMES)
    torch.onnx.export(
        net, x, path, input_names=["input"], output_names=["probs"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )


def _quantize_dynamic(fp32: str, out: str) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(fp32, out, weight_type=QuantType.QInt8)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[onnx_serve] dynamic quant failed: {e}")
        return False


def _to_fp16(fp32: str, out: str) -> bool:
    try:
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(fp32)
        onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), out)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[onnx_serve] fp16 conversion unavailable ({e}); keeping fp32")
        return False


class _MelCalibrationReader:
    """Yields log-mel frames from WAV clips for static QDQ calibration."""

    def __init__(self, wavs: List[str], n_mels: int = 64) -> None:
        from cardia.data.transforms import MelSpec, band_pass, peak_normalize, _fixed_length
        from cardia.config import Config

        cfg = Config(task="sound", stage=2, model="resnet18", class_scheme="binary",
                     data_dir="", out_dir="")
        self.mel = MelSpec(cfg.sample_rate, n_mels, cfg.n_fft, cfg.hop_length)
        self.cfg = cfg
        self._band_pass = band_pass
        self._peak = peak_normalize
        self._fixed = _fixed_length
        self._wavs = wavs
        self._i = 0

    def _load(self, path: str) -> np.ndarray:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        import torchaudio

        w = torch.from_numpy(data[:, 0])
        if sr != self.cfg.sample_rate:
            w = torchaudio.functional.resample(w, sr, self.cfg.sample_rate)
        w = self._fixed(self._band_pass(self._peak(w), self.cfg.sample_rate,
                                        self.cfg.band_low, self.cfg.band_high), self.cfg.max_samples)
        return self.mel(w).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)

    def get_next(self):  # ONNX Runtime CalibrationDataReader protocol
        while self._i < len(self._wavs):
            path = self._wavs[self._i]
            self._i += 1
            try:
                return {"input": self._load(path)}
            except Exception:
                continue
        return None


def _quantize_static(fp32: str, out: str, calib_wavs: List[str]) -> bool:
    if not calib_wavs:
        print("[onnx_serve] static_qdq requested without calibration clips; skipping")
        return False
    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

        quantize_static(fp32, out, _MelCalibrationReader(calib_wavs),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[onnx_serve] static QDQ failed: {e}")
        return False


def _validate(net: nn.Module, onnx_path: str, clips: Optional[List[str]] = None,
              n: int = 8) -> Optional[Dict]:
    """Torch ensemble vs ONNX Runtime: max prob deviation + argmax agreement.

    Uses real clips when provided, else random log-mel frames (shape/sanity check).
    """
    try:
        import onnxruntime as ort
    except Exception:  # pragma: no cover
        return None
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    reader = _MelCalibrationReader(clips) if clips else None
    rng = np.random.default_rng(0)
    max_dev, agree, used = 0.0, 0, 0
    for _ in range(n):
        if reader is not None:
            item = reader.get_next()
            if item is None:
                break
            x = item["input"]
        else:
            x = rng.standard_normal((1, 1, 64, T_FRAMES)).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()[0]
        got = sess.run(None, {name: x})[0][0]
        max_dev = max(max_dev, float(np.max(np.abs(ref - got))))
        agree += int(int(ref.argmax()) == int(got.argmax()))
        used += 1
    if not used:
        return None
    return {"inputs": used, "source": "clips" if reader is not None else "random",
            "max_prob_deviation": round(max_dev, 5), "argmax_agreement": round(agree / used, 3)}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(binary_out: str, mc_out: str, out_dir: str, quant_cnn: str = "fp16",
        quant_transformer: str = "dynamic", calib_dir: Optional[str] = None,
        validate_clips: Optional[str] = None) -> dict:
    onnx_dir = os.path.join(out_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    device = torch.device("cpu")
    calib_wavs = sorted(glob.glob(os.path.join(calib_dir, "**", "*.wav"), recursive=True)) if calib_dir else []
    val_clips = sorted(glob.glob(os.path.join(validate_clips, "**", "*.wav"), recursive=True)) if validate_clips else None

    heads_out = []
    for head in HEADS:
        src = mc_out if head["scheme"] != "binary" else binary_out
        ckpts = _ckpt_files(src, head)
        if not ckpts:
            print(f"[onnx_serve] skip {head['key']}: no checkpoints in {src}")
            continue
        net = _load_ensemble(head, ckpts, device)
        fp32 = os.path.join(onnx_dir, f"{head['key']}_fp32.onnx")
        _export_fp32(net, fp32)

        is_cnn = head["stage"] in _CNN_STAGES
        strategy = quant_cnn if is_cnn else quant_transformer
        target = os.path.join(onnx_dir, f"{head['key']}_{strategy}.onnx")
        ok = False
        if strategy == "fp16":
            ok = _to_fp16(fp32, target)
        elif strategy == "dynamic":
            ok = _quantize_dynamic(fp32, target)
        elif strategy == "static_qdq":
            ok = _quantize_static(fp32, target, calib_wavs)
        if not ok:
            target, strategy = fp32, "fp32"

        rec = {
            "key": head["key"], "task": head["task"], "scheme": head["scheme"],
            "stage": head["stage"], "model": head["model"], "classes": head["classes"],
            "folds": len(ckpts), "quant": strategy,
            "fp32": {"file": os.path.basename(fp32), "bytes": os.path.getsize(fp32),
                     "sha256": _sha256(fp32)},
            "served": {"file": os.path.basename(target), "bytes": os.path.getsize(target),
                       "sha256": _sha256(target)},
        }
        clips = sorted(glob.glob(os.path.join(validate_clips, "**", "*.wav"), recursive=True)) if validate_clips else None
        rec["validation"] = _validate(net, target, clips=val_clips)
        heads_out.append(rec)
        print(f"[onnx_serve] {head['key']}: {len(ckpts)} folds -> {strategy} "
              f"({rec['served']['bytes']/1e6:.2f} MB); {rec['validation']}")

    manifest = {
        "input": {"name": "input", "shape": [1, 1, 64, T_FRAMES], "n_mels": 64,
                  "frames": T_FRAMES, "sample_rate": 4000, "clip_s": 15.0},
        "output": {"name": "probs", "note": "post-softmax ensemble average"},
        "heads": heads_out,
    }
    path = os.path.join(out_dir, "onnx_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[onnx_serve] wrote {path}")
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export the Asculto head ensembles to serving ONNX")
    p.add_argument("--binary-out", default=os.path.join(ML, "runs"))
    p.add_argument("--mc-out", default=DEFAULT_MC_OUT)
    p.add_argument("--out-dir", default=os.path.join(ML, "runs", "onnx_serve"))
    p.add_argument("--quant-cnn", choices=["fp16", "dynamic", "static_qdq", "none"], default="fp16")
    p.add_argument("--quant-transformer", choices=["fp16", "dynamic", "none"], default="dynamic")
    p.add_argument("--calib-dir", default=None, help="WAV dir for static QDQ calibration")
    p.add_argument("--validate-clips", default=None, help="WAV dir to sanity-check quantized accuracy")
    args = p.parse_args(argv)
    run(args.binary_out, args.mc_out, args.out_dir, args.quant_cnn, args.quant_transformer,
        args.calib_dir, args.validate_clips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
