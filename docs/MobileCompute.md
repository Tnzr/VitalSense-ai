# Mobile Compute & Quantization Analysis

> Companion to `ProductComputeArchitecture.md`, `WebApp.md`, `TechStack.md`.
> Model numbers are **measured** by `ml/cardia/export/mobile.py`
> (ONNX export → dynamic INT8 → ONNX Runtime CPU latency). Device numbers are
> representative classes, not a specific SKU.

## 1. Measured model profile

Input: `1 × 64 × 376` log-mel (15 s @ 4 kHz, n_fft 400, hop 160). Latency is
ONNX Runtime **CPU**, 25 runs. Source: `runs/export/mobile_analysis.json`.

| head | model | classes | params | GFLOPs | fp32 MB | int8 MB | CPU fp32 ms | CPU int8 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| source | ResNet-18 | 3 | 11.17M | 1.68 | 44.7 | 11.2 | 8.4 | 30.3 |
| heart_binary | ResNet-18 | 2 | 11.17M | 1.68 | 44.7 | 11.2 | 9.9 | 34.0 |
| heart_types | Transformer | 10 | 1.85M | 0.34 | 7.5 | 2.0 | 5.1 | 4.2 |
| lung_binary | ResNet-18 | 2 | 11.17M | 1.68 | 44.7 | 11.2 | 9.8 | 29.1 |
| lung_types | Transformer | 6 | 1.85M | 0.34 | 7.5 | 2.0 | 4.3 | 4.9 |

**Reading it:** the ResNet-18 heads are ~5× heavier in FLOPs and ~6× larger than
the Transformer heads. The Transformer is the most mobile-friendly served model;
ResNet-18 is the cost driver (3 of 5 heads).

## 2. Mobile compute landscape (representative)

| device class | example | NPU INT8 | CPU FP32 |
|---|---|---:|---:|
| Flagship | Snapdragon 8 Gen 3 / Apple A18 | 35–45 TOPS | ~150 GFLOPS |
| Upper mid-range | Snapdragon 7+ Gen 3 | 10–15 TOPS | ~80 GFLOPS |
| Mid-range | Snapdragon 6 Gen 1 | 3–5 TOPS | ~40 GFLOPS |
| Entry | Snapdragon 4-series | 1–2 TOPS | ~15 GFLOPS |

## 3. Latency expectations

Estimate `latency ≈ GFLOPs / (TOPS × efficiency)`; small workloads rarely
saturate an NPU, so a **10 % efficiency** factor is used for NPU and **30 %**
for CPU. These are order-of-magnitude planning figures.

**Single full-clip inference (one head, one 15 s clip):**

| head | flagship NPU | mid NPU | flagship CPU | mid CPU |
|---|---:|---:|---:|---:|
| ResNet-18 (1.68 GFLOPs) | ~0.4 ms | ~4 ms | ~37 ms | ~140 ms |
| Transformer (0.34 GFLOPs) | ~0.1 ms | ~0.9 ms | ~8 ms | ~28 ms |

**Multi-scale sweep (scales 1 s/3 s/15 s → ~75 windows) for a 15 s clip:**

| set | GFLOPs | flagship NPU | mid NPU | flagship CPU |
|---|---:|---:|---:|---:|
| 1 Transformer head | ~25 | ~0.6 ms | ~6 ms | ~0.6 s |
| 1 ResNet-18 head | ~126 | ~3 ms | ~32 ms | ~3 s |
| all 5 heads (3 ResNet + 2 Transformer) | ~430 | ~11 ms | ~110 ms | ~10 s |

**Conclusion:** the full multi-scale, all-head analysis is effectively real-time
on a flagship NPU (<20 ms) and interactive on a mid-range NPU (~0.1 s). On CPU
it is a multi-second batch job — so on-device inference should target the NPU
(ONNX Runtime + NNAPI/CoreML delegate, or TFLite). Memory is a non-issue:
int8 models are 2–11 MB plus a few MB of activations.

## 4. Quantization findings (measured)

Dynamic INT8 quantization was applied with ONNX Runtime (`QuantType.QInt8`):

- **Size:** ~4× smaller — ResNet-18 44.7 → 11.2 MB, Transformer 7.5 → 2.0 MB.
- **Transformer:** faster on CPU (5.1 → 4.2 ms); dynamic INT8 targets
  MatMul/Gemm, which dominate attention.
- **ResNet-18:** **slower** on CPU (8.4 → 30.3 ms). Dynamic quantization does not
  accelerate Conv; the extra quant/dequant overhead dominates. This is expected.

**Therefore:**
- Use **static INT8 (QDQ)** or **FP16** for the CNN heads — static quantization
  quantizes activations too and is what NPUs execute; FP16 is the simplest 2×
  win on mobile GPUs/NPUs.
- Use **dynamic INT8** for the Transformer heads (or FP16 on NPU).
- Always validate quantized accuracy against the fp32 baseline on the held-out
  folds before shipping (quantization-aware training if the drop is material).

## 5. Multi-scale context and compute

A single 1–3 s window is ambiguous for slowly evolving findings; the app
therefore evaluates **multiple scales** (e.g. 1 s micro, 3 s meso, 15 s macro /
whole clip) and fuses them (length-weighted). This is the accuracy/compute
trade-off:

- Micro windows localise transients (crackles, murmurs) but are noisy alone.
- The whole-clip (macro) pass is the most reliable and is the cheapest single
  call (1 inference), so it should always run.
- The per-scale sweeps add cost roughly linearly with the number of windows;
  the table in §3 bounds it.

For on-device, a good default is **macro (whole clip) + one meso scale** to keep
the sweep under ~0.1 s on a mid NPU, with micro scale on demand.

## 6. Recommended mobile architecture

1. **Export + quantize** the heads to ONNX (done) → static INT8 for CNNs, INT8
   or FP16 for the Transformer.
2. **Runtime:** ONNX Runtime Mobile with the NNAPI (Android) / CoreML (iOS)
   execution provider; TFLite as an alternative for tighter integration.
3. **Model choice:** make the **Transformer the primary mobile head** (0.34
   GFLOPs, 2 MB int8) and/or **distill ResNet-18 → MobileNetV3-Small** (the
   campaign already includes MobileNetV3; it needs an optimisation fix before it
   is competitive — see the benchmark limitations).
4. **Streaming:** run the macro pass once and the windowed sweep progressively as
   audio is captured, so the UI can update in near-real-time.
5. **Fallback:** if no NPU, run macro-only on CPU (tens of ms for the
   Transformer) and skip the sweep.

## 7. Reproduce

```bash
.venv/bin/python -m cardia.export.mobile \
  --binary-out ml/runs \
  --mc-out ml/runs/mc \
  --out-dir ml/runs/export
# writes mobile_analysis.json + mobile_analysis.md + onnx/{head}_{fp32,int8}.onnx
```
