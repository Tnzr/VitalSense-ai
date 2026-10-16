#!/usr/bin/env python3
"""Build the CardiaSense EDUCATIONAL Jupyter notebooks (technical walkthrough).

Each notebook mixes rich markdown explanations (objectives, technique, how to
read the outputs) with executable code cells. The executed notebooks double as
the solutions document (see scripts/export_solutions_pdf.sh).

Usage:
  python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREAMBLE = f'''
import os, sys
from pathlib import Path

ROOT = Path("{ROOT}")
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT))

from cardia.config import DEFAULT_DATA_DIR   # loads .env if present
DATA = os.environ.get("CARDIASENSE_DATA_DIR") or DEFAULT_DATA_DIR
print("Dataset root:", DATA)
'''

HEAD = """# CardiaSense — {title}

**Educational technical walkthrough · notebook {n} of 5**

**Objectives:** {objective}

**You will need:** the `cardiasense` venv (run `./setup.sh`), the dataset at
`$CARDIASENSE_DATA_DIR` (downloaded by `setup.sh`), and a trained head for the
notebooks that use checkpoints (or run the smoke train in notebook 03 first).

> Educational / research prototype — **not a medical device**.

---

"""


def notebook(title: str, objective: str, n: int, cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3 (cardiasense)", "language": "python",
                                 "name": "cardiasense"}
    nb.cells = [nbf.v4.new_markdown_cell(HEAD.format(title=title, objective=objective, n=n))]
    for cell_type, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if cell_type == "md" else nbf.v4.new_code_cell(src))
    return nb


def build_all(out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # ================================================================= 01
    cells = [
        ("md", """
### Step 1 — Load the manifest (CSV → recordings)

The dataset's ground truth lives in `HS.csv`, `LS.csv`, `Mix.csv`. Filenames are
reconstructed from the **ID columns** (never the Location column — `Apex` in the
CSV maps to `A` in the filename). The manifest builder also strips whitespace
(one `LS.csv` row has a leading space) and *asserts* file existence + expected
counts, so a corrupt download fails loudly instead of silently.

**What you should see:** 535 clips for the sound task (50 heart + 50 lung +
145 mixed sessions × 3 stems), and the class balance — note how imbalanced the
heart task is (≈89% abnormal), which is exactly why we use balanced metrics and
class-weighted losses later.
"""),
        ("code", PREAMBLE),
        ("code", """
from cardia.data.hls_cmds import build_task_manifest
m = build_task_manifest(DATA, "sound", "binary")
print("sound:", len(m), "files")
print(m["task_label"].value_counts())
for task in ("heart", "lung"):
    mm = build_task_manifest(DATA, task, "binary")
    print(task, len(mm), "samples")
    print(mm["task_label"].value_counts().to_dict())
"""),
        ("md", """
### Step 2 — Listen (the most important "feature")

Every clip is 15 s of 4 kHz, 16-bit mono audio. Listen to each type below.
- **Heart:** you should hear the *lub-dub* (S1–S2); a murmur is a longer,
  whooshing vibration between them.
- **Lung:** normal breathing is quiet; **wheezes** are high-pitched whistles,
  **crackles** are short pops (like hair being rubbed), **rhonchi** low rattles.

**Why this matters:** if you can hear it, a well-designed spectrogram + model
can learn it — these clips are your sanity check for everything that follows.
"""),
        ("code", """
import IPython.display as d
from cardia.data.transforms import load_wav, peak_normalize

for label in ("heart", "lung", "mixed"):
    row = m[m["task_label"] == label].iloc[0]
    wav, sr = load_wav(row["file_path"])
    display(d.Audio(wav.numpy(), rate=sr))
    print(label, row["sample_id"])
"""),
        ("md", """
### Step 3 — From waveform to log-mel spectrogram

A raw waveform is hard for a CNN. The standard acoustic front-end:

1. **Frame + STFT** — slice into ~100 ms windows (n_fft=400 @ 4 kHz), hop 40 ms
   (160 samples) → ~376 frames for 15 s.
2. **Mel filterbank** — compress frequency onto 64 mel bands (perceptual scale).
3. **Log** — `log(S + 1e-6)` converts the wide dynamic range of power into a
   perceptual "loudness" scale and stabilises training.

The result is a `64 × 376` image: **time on x, mel bin on y** — this is the
input every deep head consumes.
"""),
        ("code", """
import matplotlib.pyplot as plt
from cardia.config import Config
from cardia.data.transforms import MelSpec, load_wav, peak_normalize, preprocess
from cardia.wandb_utils import viz

cfg = Config()
row = m[m["task_label"] == "heart"].iloc[0]
raw, sr = load_wav(row["file_path"]); raw = peak_normalize(raw)
filt = preprocess(row["file_path"], cfg)          # band-pass 20-600 Hz (heart band)
mel = MelSpec(cfg.sample_rate, cfg.n_mels, cfg.n_fft, cfg.hop_length)
viz.waveform_fig(raw.numpy(), sr, f"raw {row['sample_id']}"); plt.show()
spec = mel(filt).numpy()
print("log-mel shape:", spec.shape)
viz.spec_fig(spec, f"log-mel {row['sample_id']} (64 mel bins × 376 frames)")
plt.show()
"""),
        ("md", """
### Step 4 — Signal quality (garbage in, garbage out)

`clip_ratio` (samples touching ±1), RMS level, and an SNR estimate tell us if a
recording is usable. A clipped or near-silent clip should be flagged before it
is fed to a model — the web app surfaces these flags on every report.
"""),
        ("code", """
from cardia.data.stats import sqi_stats
df = sqi_stats(m, n=50)
print(df[["sample_id", "duration_s", "rms", "snr_estimate", "clip_ratio"]].describe().T)
"""),
    ]
    paths.append(_write(out_dir, "01_dataset_exploration",
                        notebook("Dataset & signal foundations", "load the manifest, listen to the sounds, build a spectrogram, estimate quality", 1, cells)))

    # ================================================================= 02
    cells = [
        ("md", """
### Step 1 — Band-pass filtering (the task-dependent "lens")

Heart energy concentrates at **20–600 Hz**, lung sounds at **60–1500 Hz**
(Nyquist = 2000 Hz at 4 kHz). We apply a zero-phase Butterworth band-pass
(`scipy.signal.sosfiltfilt`) so the filter doesn't shift the signal in time.

**Technical note:** `sos` (second-order sections) + `sosfiltfilt` is numerically
stable and filters the signal *forwards then backwards*, removing phase lag —
essential for downstream time-aligned visualisation.
"""),
        ("code", PREAMBLE),
        ("code", """
import numpy as np
import pandas as pd
from cardia.config import Config
from cardia.data.hls_cmds import build_task_manifest
from cardia.data.transforms import load_wav, peak_normalize, preprocess
from cardia.wandb_utils import viz
import matplotlib.pyplot as plt

cfg = Config()
lm = build_task_manifest(DATA, "lung", "binary")
row = lm.iloc[0]
raw, sr = load_wav(row["file_path"]); raw = peak_normalize(raw)
filt = preprocess(row["file_path"], cfg)          # lung band 60-1500 Hz
viz.overlay_fig(raw.numpy(), filt.numpy(), sr, f"band-pass overlay {row['sample_id']}")
plt.show()
"""),
        ("md", """
### Step 2 — Augmentation (train-only, and deterministic)

To generalise past 535 clips we synthesise plausible variants during training:
**time roll**, **amplitude scaling**, **additive Gaussian noise** on the
waveform, and **SpecAugment** masks on the spectrogram. Everything is driven by
a seeded `torch.Generator`, re-seeded per epoch, so runs are reproducible
(fold-0 reruns are bit-identical).

**Watch the difference** — the right panel shows the masked/perturbed version.
"""),
        ("code", """
from cardia.data.transforms import Augment, MelSpec
aug = Augment(cfg, cfg.seed); aug.set_epoch(0)
w = aug.waveform(filt)
mel_spec = MelSpec(cfg.sample_rate, cfg.n_mels, cfg.n_fft, cfg.hop_length)
viz.aug_fig(mel_spec(filt).numpy(), aug.spec(mel_spec(w)).numpy(),
            f"augmentation {row['sample_id']}")
plt.show()
"""),
        ("md", """
### Step 3 — Leakage-safe, grouped, stratified splits

The **critical** methodological choice: a mixed session produces `H####`, `L####`
and `M####` clips that *must never* straddle train/val/test — otherwise the model
"recognises" the session instead of generalising.

The splitter:
1. **Outer** `StratifiedGroupKFold(5)` partitions **groups** (the session index)
   → fold *k* is TEST.
2. **Inner** re-split of the remaining groups → validation (~20%).
3. **Assertions** verify train/val/test group sets are pairwise disjoint and that
   the test folds cover every group exactly once.

Expected shape: ~60% train / 20% val / 20% test, with every fold containing all
classes (stratification).
"""),
        ("code", """
from cardia.data.splits import make_splits
m = build_task_manifest(DATA, "sound", "binary")
splits = make_splits(m, n_folds=5, seed=0)   # asserts group disjointness
rows = [{"fold": s.fold, "train": len(s.train), "val": len(s.val), "test": len(s.test)} for s in splits]
print(pd.DataFrame(rows).to_string(index=False))
"""),
    ]
    paths.append(_write(out_dir, "02_preprocessing_splits",
                        notebook("DSP, augmentation & validation discipline", "band-pass filtering, seeded augmentation, leakage-safe grouped CV splits", 2, cells)))

    # ================================================================= 03
    cells = [
        ("md", """
### Step 1 — Classical baseline (fast, interpretable)

**Features:** 13 MFCC coefficients + their Δ and Δ² (temporal dynamics), plus
spectral centroid / bandwidth, zero-crossing rate, RMS and spectral flatness —
aggregated to mean/std per clip.

**Models:** logistic regression (LR), random forest (RF), and a **calibrated**
SVM. All use `class_weight="balanced"` to counter the imbalance. Models fit on
the **train split only** (val is reserved for model selection — same protocol as
the deep stages), scored on the held-out test fold.

**Expected:** the classical baseline is the *gate champion* on every task — a
cheap, strong reference the deep tiers must beat by ≥ 1 balanced-accuracy point.
"""),
        ("code", PREAMBLE),
        ("code", """
from cardia.config import Config
from cardia.data.hls_cmds import build_task_manifest
from cardia.data.splits import make_splits
from cardia.train import classical_train
import wandb, pandas as pd

cfg = Config(task="heart", stage=1, out_dir=str(ROOT / "ml" / "runs"))
m = build_task_manifest(DATA, "heart", "binary")
splits = make_splits(m, n_folds=5, seed=0)
run = wandb.init(mode="disabled")
metrics = classical_train.run_classical(cfg, m, splits, ["normal", "abnormal"], run, "nb-walkthrough")
run.finish()
print(pd.DataFrame(metrics["_per_classifier"])[["classifier", "balanced_accuracy_mean", "macro_f1_mean"]].to_string(index=False))
"""),
        ("md", """
### Step 2 — Deep tier (smoke mode) via the CLI

The deep stages (stage 2 CNN → stage 3 temporal → stage 4 transformer) are
invoked through `cardia.cli`. `--smoke` cuts to 2 folds × 2 epochs so you can
see the whole path (dataloaders, class weights, cosine LR, early stopping,
checkpoint, metrics) in about a minute on GPU.

**What it prints:** val balanced accuracy / macro-F1 per epoch, then the
held-out test metrics with a confusion matrix, ECE and latency.
"""),
        ("code", """
import subprocess
rc = subprocess.run([sys.executable, "-m", "cardia.cli", "--task", "heart", "--stage", "2",
                     "--model", "resnet18", "--fold", "0", "--smoke", "--out-dir", "/tmp/cardiasense_nb_runs"],
                    cwd=str(ROOT / "ml"))
print("CLI exit code:", rc.returncode)
"""),
        ("md", """
### Step 3 — The campaign & the gate rule

`scripts/run_sequential.sh` runs all four stages for every task, then applies
the **gate**: a stage is adopted only if balanced accuracy **or** macro-F1
improves by ≥ 1 point over the best prior stage. Both metrics are logged, so the
decision is auditable in `ml/runs/metrics.json` and the benchmark table.
"""),
        ("code", """
import json
p = ROOT / "ml" / "runs" / "benchmark.json"
if p.exists():
    bench = json.loads(p.read_text())["benchmark"]
    print(pd.DataFrame(bench).to_string(index=False))
else:
    print("No benchmark.json yet — run:  cd ml && SMOKE=1 ./scripts/run_sequential.sh")
"""),
    ]
    paths.append(_write(out_dir, "03_benchmark_training",
                        notebook("Model tiers, training & the benchmark gate", "classical baseline, deep smoke training, campaign tables and the adopt/skip gate", 3, cells)))

    # ================================================================= 04
    cells = [
        ("md", """
### Step 1 — The encoder as a tokenizer (research direction A)

The AST-style transformer splits the spectrogram into **patches** and maps each
to a token; a `[CLS]` token pools the whole clip. `encode()` exposes these
internals: `(patch_tokens (T, d), cls (d))`.

**Hypothesis:** the encoder already *organises* these vectors so class
embeddings cluster — measure it below with the **cosine similarity between
class-mean CLS embeddings**. A sharp diagonal = classes separated in embedding
space → those vectors are a ready-made, ordered interface for a downstream LLM
(frozen encoder + linear projection, or patch tokens as soft prompts).

> If no transformer checkpoint is present this cell prints a note instead of
> failing — run `03`'s full campaign first to have one.
"""),
        ("code", PREAMBLE),
        ("code", """
import matplotlib.pyplot as plt
from cardia.config import HEART_TYPES

MC = str(ROOT / "ml" / "runs" / "mc")   # multiclass campaign (Transformer heads)
try:
    from cardia.features import embeddings as emb
    data = emb.corpus_embeddings(DATA, MC)
    M = emb.class_mean_matrix(data)
    print("clips:", len(data["rows"]["sample_id"]), "| embedding dim:", data["rows"]["cls"][0].shape)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(HEART_TYPES))); ax.set_yticks(range(len(HEART_TYPES)))
    ax.set_xticklabels([c[:10] for c in HEART_TYPES], rotation=45, ha="right")
    ax.set_yticklabels([c[:10] for c in HEART_TYPES])
    ax.set_title("Class-mean CLS embedding cosine similarity (encoder-organised)")
    plt.show()
except FileNotFoundError as e:
    print("Transformer checkpoint not available here:", e)
"""),
        ("md", """
### Step 2 — A product-style report in one call

The web app runs **5-fold ensembles** of the heads, sweeps the clip in a sliding
window, and renders a **synchronized inference timeline** (spectrogram on top,
stacked-area probabilities in the middle, confidence line at the bottom — all
time-aligned). This cell produces the same objects the report page renders, so
you can inspect the structure directly.
"""),
        ("code", """
import sys, torch, soundfile as sf
sys.path.insert(0, str(ROOT / "webapp"))
from webapp.app import inference, report, game

reg = inference.get_registry()               # warm-loads head ensembles (CPU here)
opts = report.ReportOptions(heads=["source", "heart_types"], figures="all",
                            temporal=True, temporal_mode="multiscale", explanation="standard")
path = game.resolve_audio_path("M0001", "sound", "binary")
wav, sr = sf.read(path, dtype="float32", always_2d=True)
rep = report.analyze_waveform(torch.from_numpy(wav[:, 0]).squeeze(), int(sr), "M0001.wav", opts)
print("verdict:", rep["verdict"], "| heads used:", [h["key"] for h in rep["heads"]])
print("figures:", list(rep["figures"].keys())[:6], "...")
"""),
        ("md", """
### Step 3 — What ships in the repo

The GitHub README plays sample clips directly (HTML5 `<audio>` — GitHub renders
them from committed WAVs) and the walkthrough embeds web-app screenshots. The
commands below point you at those artifacts; `scripts/screenshot_walkthrough.py`
regenerates the screenshots and `scripts/make_audio_assets.py` the clips.
"""),
        ("code", """
print("Audio assets:   docs/assets/audio/*.wav   (played inline in the README)")
print("Screenshots:    docs/walkthrough/*.png    (instructional walkthrough)")
print("Report:         ml/report/cardiasense_report.pdf (regenerate via cardia.report.report)")
"""),
    ]
    paths.append(_write(out_dir, "04_report_embeddings_webapp",
                        notebook("Embeddings, product report & web app", "encoder token embeddings, ensemble report internals, README audio/screenshots", 4, cells)))

    # ================================================================= 05
    cells = [
        ("md", """
### Step 1 — From probabilities to a personalized narrative

The heads output probabilities — a patient wants *words*. The LLM integration
(`webapp/app/llm.py`) has two interchangeable providers that return the **same
JSON shape**:

- `LLMClient` — calls an OpenAI-compatible endpoint (`CARDIASENSE_LLM_*`), asked
  for strict JSON, grounded in the findings + patient context.
- `LLMSynthesizer` — an **offline rule-based fallback** built from a per-class
  knowledge base (medical / diet / environment outlines + context hints), so the
  product works with no API key at all.

Below we synthesize from a plausible finding vector and a short patient note.
Notice the **personalization** (context hints folded into the medical
recommendations) and the guardrail: every output carries the disclaimer.
"""),
        ("code", PREAMBLE),
        ("code", """
from webapp.app import llm as llm_mod
import json

findings = [
    {"key": "heart_binary", "label": "Heart screening", "kind": "binary", "pred": "abnormal",
     "confidence": 0.82, "classes": ["normal", "abnormal"], "probs": [0.18, 0.82]},
    {"key": "lung_types", "label": "Lung typing", "kind": "multiclass", "pred": "Wheezing",
     "confidence": 0.71, "classes": ["Normal", "Wheezing", "Fine Crackles", "Coarse Crackles",
                                     "Rhonchi", "Pleural Rub"]},
]
out = llm_mod.generate_narrative(findings, patient_context="58y, asthma, mild palpitations")
print("provider:", out["provider"], "| model:", out["model"], "| personalized:", out["personalized"])
print("\\nSUMMARY:", out["summary"])
print("\\nFINDINGS:"); [print(" -", f) for f in out["findings"]]
for area, recs in out["recommendations"].items():
    print(f"\\n[{area}]"); [print("  -", r) for r in recs]
"""),
        ("md", """
### Step 2 — Follow-up conversation (chat semantics)

The report page's chat sends `{findings, patient_context, history}` to
`POST /api/llm/report`; the LLM (or synthesizer) answers with the same shape, so
follow-ups inherit the auscultation context. With a key set, the live model
handles the dialogue; without one, the synthesizer keeps the flow working.
"""),
        ("code", """
if os.environ.get("CARDIASENSE_LLM_API_KEY"):
    narr = llm_mod.generate_narrative(findings, "asthma",
                                      [{"role": "user", "content": "should I worry about the wheeze?"}])
    print("provider:", narr["provider"], "|", narr.get("summary", "")[:160])
else:
    print("No CARDIASENSE_LLM_API_KEY — using the offline synthesizer (shown above). "
          "Set it (or a local Ollama base URL) to exercise the live LLM path.")
"""),
        ("md", """
### Step 3 — Two research directions worth your time

1. **Token/embedding architecture (done in part, notebook 04).** The encoder is
   already a tokenizer: reuse its patch/CLS embeddings as the audio interface to
   a language model (frozen encoder + linear projection, or patch tokens as soft
   prompts). The served Transformer is ~0.34 GFLOPs / 2 MB int8 — cheap enough to
   run on-device and let the cloud LLM do the talking.

2. **Whole-breathing-cycle modeling.** A 1–3 s snippet can miss the pattern;
   proper auscultation wants the whole cycle (the app's macro pass already uses
   the full 15 s clip). Open paths: envelope-based **cycle segmentation** with
   per-cycle embeddings + cross-cycle attention, **long-context encoders** for
   minute-long exams, and **cycle-aware training** to reduce short-clip label
   noise. Full design + evaluation plan in `docs/LLM_Integration.md`.
"""),
    ]
    paths.append(_write(out_dir, "05_llm_integration",
                        notebook("LLM integration & forward architecture", "provider/synthesizer narrative, personalization, chat, and two research directions", 5, cells)))

    return paths


def _write(out_dir: str, name: str, nb: nbf.NotebookNode) -> str:
    path = os.path.join(out_dir, f"{name}.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    return path


if __name__ == "__main__":
    out = os.path.join(ROOT, "notebooks")
    for p in build_all(out):
        print("wrote", p)
    print("notebooks ->", out)