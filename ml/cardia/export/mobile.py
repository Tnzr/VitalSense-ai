"""Mobile readiness pipeline: params, FLOPs, ONNX export, dynamic INT8
quantization, and CPU latency for every served head.

Writes measured numbers to <out-dir>/mobile_analysis.json and a Markdown
summary to <out-dir>/mobile_analysis.md.

Usage:
  python -m cardia.export.mobile --binary-out runs --mc-out <mc> --out-dir runs/export
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
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


def _builder(stage: int):
    return {2: build_cnn, 3: build_temporal, 4: build_transformer}[stage]


HEADS = [
    dict(key="source", task="sound", scheme="binary", stage=2, model="resnet18", classes=3),
    dict(key="heart_binary", task="heart", scheme="binary", stage=2, model="resnet18", classes=2),
    dict(key="heart_types", task="heart", scheme="10class", stage=4, model="transformer", classes=len(HEART_TYPES)),
    dict(key="lung_binary", task="lung", scheme="binary", stage=2, model="resnet18", classes=2),
    dict(key="lung_types", task="lung", scheme="6class", stage=4, model="transformer", classes=len(LUNG_TYPES)),
]

T_FRAMES = 376  # 15 s @ 4 kHz, n_fft 400 / hop 160


def _ckpt(out_dir: str, head: dict) -> Optional[str]:
    suf = f"-{head['scheme']}" if head["scheme"] != "binary" else ""
    pats = [
        os.path.join(out_dir, "checkpoints",
                     f"stage{head['stage']}-{head['task']}-{head['model']}{suf}-allfolds-s*-fold0_best.pt"),
        os.path.join(out_dir, "checkpoints",
                     f"stage{head['stage']}-{head['task']}-{head['model']}{suf}-fold0-s*_best.pt"),
    ]
    files = sorted({p for pat in pats for p in glob.glob(pat)})
    return files[0] if files else None


def _load_model(head: dict, ckpt: str, device: torch.device) -> nn.Module:
    cfg = Config(task=head["task"], stage=head["stage"], model=head["model"],
                 class_scheme=head["scheme"], data_dir="", out_dir="")
    net = _builder(head["stage"])(cfg, head["classes"]).to(device)
    sd = torch.load(ckpt, map_location=device)["model_state"]
    if "_pos" in sd:
        net._pos = nn.Parameter(sd["_pos"].to(device))
    net.load_state_dict(sd)
    net.eval()
    return net


def _params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def _flops(net: nn.Module, shape=(1, 1, 64, T_FRAMES)) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(*shape)
    try:
        with FlopCounterMode(display=False) as fc:
            net(x)
        return int(fc.get_total_flops())
    except Exception:
        return -1


def _export_onnx(net: nn.Module, path: str, dynamic_t: bool) -> None:
    x = torch.randn(1, 1, 64, T_FRAMES)
    torch.onnx.export(
        net, x, path, input_names=["input"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )


def _quantize(fp32_path: str, int8_path: str) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[export] quantize failed: {e}")
        return False


def _ort_latency(path: str, runs: int = 25) -> Dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 1, 64, T_FRAMES).astype(np.float32)
    for _ in range(3):
        sess.run(None, {name: x})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": float(np.mean(ts)), "p50_ms": float(np.percentile(ts, 50)),
            "p95_ms": float(np.percentile(ts, 95))}


def run(binary_out: str, mc_out: str, out_dir: str) -> dict:
    os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    device = torch.device("cpu")
    results = []
    for head in HEADS:
        src = mc_out if head["scheme"] != "binary" else binary_out
        ckpt = _ckpt(src, head)
        if ckpt is None:
            print(f"[export] skip {head['key']}: no checkpoint")
            continue
        net = _load_model(head, ckpt, device)
        params = _params(net)
        flops = _flops(net)
        fp32 = os.path.join(out_dir, "onnx", f"{head['key']}_fp32.onnx")
        int8 = os.path.join(out_dir, "onnx", f"{head['key']}_int8.onnx")
        dynamic_t = head["stage"] != 4  # transformer has fixed positional embedding
        _export_onnx(net, fp32, dynamic_t)
        ok = _quantize(fp32, int8)
        fp32_mb = os.path.getsize(fp32) / 1e6
        int8_mb = os.path.getsize(int8) / 1e6 if ok and os.path.isfile(int8) else None
        lat_fp32 = _ort_latency(fp32)
        lat_int8 = _ort_latency(int8) if ok else None
        results.append({
            "key": head["key"], "model": head["model"], "stage": head["stage"],
            "classes": head["classes"], "params": params,
            "flops": flops, "input_frames": T_FRAMES,
            "fp32_mb": round(fp32_mb, 2), "int8_mb": round(int8_mb, 2) if int8_mb else None,
            "quantized": ok,
            "latency_fp32_ms": {k: round(v, 2) for k, v in lat_fp32.items()},
            "latency_int8_ms": {k: round(v, 2) for k, v in lat_int8.items()} if lat_int8 else None,
        })
        print(f"[export] {head['key']}: {params/1e3:.0f}K params, {flops/1e6:.0f} MFLOPs, "
              f"fp32 {fp32_mb:.2f} MB, int8 {int8_mb:.2f} MB, "
              f"lat {lat_fp32['mean_ms']:.1f} ms -> {lat_int8['mean_ms']:.1f} ms" if int8_mb else "")
    out = {"input": {"n_mels": 64, "frames": T_FRAMES, "sample_rate": 4000, "clip_s": 15.0}, "heads": results}
    with open(os.path.join(out_dir, "mobile_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    _write_markdown(out, os.path.join(out_dir, "mobile_analysis.md"))
    return out


def _write_markdown(out: dict, path: str) -> None:
    lines = ["# Mobile readiness — measured model profile", "",
             f"Input: 1×64×{out['input']['frames']} log-mel ({out['input']['clip_s']} s @ "
             f"{out['input']['sample_rate']} Hz). Latency = ONNX Runtime CPU, 25 runs.", "",
             "| head | model | classes | params | MFLOPs | fp32 MB | int8 MB | fp32 ms | int8 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in out["heads"]:
        lines.append(
            f"| {r['key']} | {r['model']} | {r['classes']} | {r['params']/1e3:.0f}K | "
            f"{r['flops']/1e6:.0f} | {r['fp32_mb']} | {r['int8_mb'] if r['int8_mb'] else '-'} | "
            f"{r['latency_fp32_ms']['mean_ms']} | "
            f"{r['latency_int8_ms']['mean_ms'] if r['latency_int8_ms'] else '-'} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export + quantize heads and measure mobile profile")
    p.add_argument("--binary-out", default=os.path.join(ML, "runs"))
    p.add_argument("--mc-out", default=DEFAULT_MC_OUT)
    p.add_argument("--out-dir", default=os.path.join(ML, "runs", "export"))
    args = p.parse_args(argv)
    out = run(args.binary_out, args.mc_out, args.out_dir)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
