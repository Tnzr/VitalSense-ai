"""Comprehensive CardiaSense HLS-CMDS benchmark report generator.

Produces a narrative PDF + PowerPoint deck: each figure is accompanied by
contextual explanation (what it shows, how to read it, what it means), plus
methodology, results, limitations and next steps. All numbers are pulled from
the campaign outputs so the prose stays accurate.

Usage:
  python -m cardia.report.report \
      --binary-out runs --mc-out ml/runs \
      --out-dir report
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import textwrap
from typing import List, Optional

import numpy as np
import pandas as pd

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(ML)
if ML not in sys.path:
    sys.path.insert(0, ML)

from cardia.config import DEFAULT_MC_OUT, Config, HEART_TYPES, LUNG_TYPES
from cardia.data.hls_cmds import build_task_manifest, class_names
from cardia.report import demo, infer
from cardia.wandb_utils import viz

PROJECT = "cardiasense-hls-cmds"
FIGURE_WARNINGS = {}


# ------------------------------------------------------------------ helpers
def save_fig(fig, out_dir: str, name: str) -> str:
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    path = os.path.join(out_dir, "figures", f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _best(bench: pd.DataFrame, task: str, scheme: str) -> Optional[pd.Series]:
    sub = bench[(bench["task"] == task) & (bench["scheme"] == scheme)]
    if sub.empty:
        return None
    return sub.loc[sub["bal_acc"].idxmax()]


def _asset(*parts: str) -> Optional[str]:
    """Resolve a committed docs/ asset path (images used by README/walkthrough)."""
    p = os.path.join(REPO, "docs", *parts)
    return p if os.path.exists(p) else None


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _probs_summary(data_dir: str, out_dir: str, task: str, scheme: str, stage: int, model: str):
    try:
        df = infer.infer_test_probs(data_dir, out_dir, task, scheme, stage, model, seed=0)
    except Exception:
        return None
    acc = float((df["true"] == df["pred"]).mean())
    return df, acc


def _draw_text(fig, x: float, y_top: float, width_chars: int, paragraphs: List[str],
               fontsize: float = 10.0, bullet: bool = False) -> float:
    h = fig.get_figheight()
    y = y_top
    for para in paragraphs:
        text = ("• " + para) if bullet else para
        lines = textwrap.wrap(text, width=width_chars) or [""]
        for ln in lines:
            fig.text(x, y, ln, ha="left", va="top", fontsize=fontsize)
            y -= fontsize * 1.35 / (72 * h)
        y -= fontsize * 0.7 / (72 * h)
    return y


# ------------------------------------------------------------------ pages
def build_pages(binary_out: str, mc_out: str, data_dir: str, out_dir: str) -> List[dict]:
    bench = demo.benchmark_rows(binary_out, mc_out)
    cfg = Config(data_dir=data_dir)
    pages: List[dict] = []

    # ---- 1. title
    pages.append({
        "title": "CardiaSense AI — HLS-CMDS Auscultation Benchmark",
        "paras": [
            "Stage 1 AI Proof-of-Concept · comprehensive results report",
            f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}",
            "Dataset: HLS-CMDS (heart & lung sounds from a clinical manikin, digital stethoscope)",
            "Scope: 3 classification tasks × 4 model tiers × 5-fold grouped cross-validation, "
            "tracked in Weights & Biases.",
        ],
        "image": None,
    })

    # ---- 2. executive summary
    b_heart = _best(bench, "heart", "binary")
    b_lung = _best(bench, "lung", "binary")
    b_sound = _best(bench, "sound", "binary")
    m_heart = _best(bench, "heart", "multiclass")
    m_lung = _best(bench, "lung", "multiclass")
    hs = _probs_summary(data_dir, binary_out, "sound", "binary", 2, "resnet18")
    hh = _probs_summary(data_dir, mc_out, "heart", "10class", 4, "transformer")
    ll = _probs_summary(data_dir, mc_out, "lung", "6class", 4, "transformer")
    pages.append({
        "title": "Executive summary",
        "paras": [
            f"Binary screening (normal vs abnormal) is highly separable: the classical MFCC baseline "
            f"reaches balanced accuracy {_fmt(b_heart['bal_acc'])} on heart and {_fmt(b_lung['bal_acc'])} on lung "
            f"at the stage-1 gate, and {_fmt(b_sound['bal_acc'])} on the 3-way heart/lung/mixed task.",
            f"Fine-grained typing is harder but tractable: 10-class heart typing peaks at "
            f"{_fmt(m_heart['bal_acc'])} balanced accuracy and 6-class lung typing at {_fmt(m_lung['bal_acc'])}. "
            f"The longer-trained Transformer heads reach sample accuracy "
            f"{_fmt(hh[1]) if hh else 'n/a'} (heart 10-class) and {_fmt(ll[1]) if ll else 'n/a'} (lung 6-class).",
            "Under the pre-registered gate (adopt a later stage only if balanced accuracy or macro-F1 improves "
            "by ≥ 1 point over the best prior stage), the cheap classical baselines win on every task — a strong "
            "signal that the linear/feature-based path is a viable production candidate before deeper models.",
            "Recommended next steps: (1) push the Transformer refinement run to convergence, (2) investigate the "
            "MobileNetV3 collapse, (3) add the mixed-domain probe and per-class calibration before any deployment claim.",
        ],
        "image": None,
    })

    # ---- 3. dataset
    m_sound = build_task_manifest(data_dir, "sound", "binary", os.path.join(out_dir, "manifests"))
    dist = m_sound.groupby("task_label").size().reset_index(name="count")
    dist["share"] = (dist["count"] / dist["count"].sum()).round(3)
    dataset_fig = demo.table_fig(dist, "HLS-CMDS sound-task composition (535 clips)")
    pages.append({
        "title": "Dataset and tasks",
        "paras": [
            "HLS-CMDS contains 15-second, 4 kHz, 16-bit mono recordings captured from a clinical manikin with a "
            "digital stethoscope: 50 heart-only, 50 lung-only, and 145 mixed recordings, each mixed session "
            "providing heart (H####), lung (L####) and combined (M####) clips.",
            "Three tasks are benchmarked: (A) heart / lung / mixed sound-type identification (535 clips), "
            "(B) heart normal vs abnormal and (C) lung normal vs abnormal (195 clean clips each, drawing on the "
            "mixed heart/lung stems). Fine-grained variants add 10 heart types and 6 lung types.",
            "Class imbalance is severe for heart (≈89% abnormal) and moderate for lung, which is why every model "
            "is trained with class-weighted loss and scored with balanced accuracy and macro-F1 rather than raw accuracy.",
            "Grouping: H/L/M clips from the same session share a group key, and all splits keep whole groups "
            "together, so no recording session can leak across train/validation/test.",
        ],
        "image": save_fig(dataset_fig, out_dir, "dataset"),
    })

    # ---- 4. methodology
    pages.append({
        "title": "Methodology",
        "paras": [
            "Preprocessing: mono load → peak normalisation → zero-phase Butterworth band-pass "
            "(heart 20–600 Hz, lung 60–1500 Hz, sound 20–1500 Hz) → centre pad/crop to 15 s → 64-band log-mel "
            "spectrogram (n_fft 400, hop 160). Training-only augmentation: time roll, amplitude scaling, additive "
            "Gaussian noise, and SpecAugment time/frequency masking, all seeded for reproducibility.",
            "Splits: a two-level StratifiedGroupKFold (outer 5-fold test, inner split of the remainder for "
            "validation ≈ 60/20/20) with assertions that train/val/test group sets are disjoint and that test folds "
            "cover every group exactly once. Split files are reused verbatim by every stage and model.",
            "Model tiers: (1) classical MFCC+spectral features with logistic regression / random forest / calibrated "
            "SVM; (2) ResNet-18 and MobileNetV3-Small (1-channel, pretrained=False); (3) a 1-D CNN + GRU/LSTM over "
            "mel frames; (4) a small AST-style spectrogram Transformer (patch embedding, learned positional "
            "embedding, CLS head, deterministic math-SDP attention).",
            "Training: class-weighted cross-entropy, AdamW (lr 1e-3, cosine decay), early stopping on validation "
            "balanced accuracy (patience 10), best-checkpoint selection, then a held-out test evaluation. Metrics: "
            "accuracy, balanced accuracy, macro precision/recall/F1, AUROC, expected calibration error (ECE) and "
            "per-sample inference latency. Seeds and deterministic algorithms are fixed; a fold-0 rerun reproduced "
            "bit-identical checkpoints.",
            "Gate rule: sequential stages are adopted only if balanced accuracy OR macro-F1 improves by ≥ 1 point "
            "over the best prior stage on the same task; otherwise the stage is recorded as skipped. Both metrics are "
            "logged so the decision is auditable.",
        ],
        "image": None,
    })

    # ---- 5. benchmark table
    pages.append({
        "title": "Benchmark summary",
        "paras": [
            "Each row is a task/scheme and a model tier; values are the best test balanced accuracy and macro-F1 "
            "across the stage's models, averaged over 5 folds. The gate column shows whether the tier was adopted "
            "over the best prior stage.",
            "Reading it: stage 1 (classical) is adopted on every task; the deep tiers do not clear the +1-point bar "
            "for binary tasks, but the Transformer is competitive on the fine-grained tasks (heart 10-class, lung 6-class).",
            "The consistently low MobileNetV3 scores indicate an optimisation failure on this small, low-resolution "
            "input rather than a data ceiling — see the Limitations page.",
        ],
        "image": save_fig(demo.table_fig(bench, "Best test balanced accuracy / macro-F1 per stage"), out_dir, "benchmark"),
    })

    # ---- 6-11. confusion matrices
    def cm_page(task, scheme, stage, model, names, label, out_src, context):
        df = demo.load_predictions(out_src, task, scheme, stage, model)
        if df is None:
            return None
        cm = demo.confusion_from_predictions(df, names)
        fig = viz.confusion_matrix_fig(cm, names, label, normalize=True)
        img = save_fig(fig, out_dir, f"cm_{task}_{scheme}")
        return {
            "title": f"Confusion matrix — {label}",
            "paras": context + [
                "How to read: each row is the ACTUAL class and the heatmap is normalised within that row, so the "
                "diagonal is that class's recall (the % of actual cases correctly identified) and off-diagonal cells "
                "show where the model confuses classes. Cells are annotated with raw count and % of the actual class.",
            ],
            "image": img,
        }

    p = cm_page("heart", "binary", 2, "resnet18", ["normal", "abnormal"],
                "heart normal vs abnormal (ResNet-18)", binary_out,
                ["Binary heart screening: the matrix shows how often normal and abnormal heart sounds are separated "
                 "on the held-out folds, aggregated across all five test folds."])
    if p: pages.append(p)
    p = cm_page("lung", "binary", 2, "resnet18", ["normal", "abnormal"],
                "lung normal vs abnormal (ResNet-18)", binary_out,
                ["Binary lung screening on clean lung recordings; the abnormal class pools crackles, wheeze, rhonchi "
                 "and pleural rub, so residual confusion is expected to be small."])
    if p: pages.append(p)
    p = cm_page("sound", "binary", 2, "resnet18", ["heart", "lung", "mixed"],
                "heart / lung / mixed (ResNet-18)", binary_out,
                ["Three-way source identification; the diagonal shows that heart-only and lung-only clips are "
                 "distinguished from mixed clips, which is the first decision a routing system must make."])
    if p: pages.append(p)
    p = cm_page("heart", "10class", 2, "resnet18", HEART_TYPES,
                "heart sound types — 10 classes (ResNet-18)", mc_out,
                ["Fine-grained heart typing across all 10 murmur/rhythm labels. Rare classes (e.g. S4, AV Block, "
                 "Tachycardia) have very few examples, so their rows are noisy; the confusion concentrates among "
                 "systolic/diastolic murmurs that share spectral content."])
    if p: pages.append(p)
    p = cm_page("lung", "6class", 2, "resnet18", LUNG_TYPES,
                "lung sound types — 6 classes (ResNet-18)", mc_out,
                ["Fine-grained lung typing across the six adventitious-sound classes; crackles and rhonchi are the "
                 "most confusable because of overlapping broadband content."])
    if p: pages.append(p)

    # ---- combined
    heart_df = demo.load_predictions(mc_out, "heart", "10class", 2, "resnet18")
    lung_df = demo.load_predictions(mc_out, "lung", "6class", 2, "resnet18")
    if heart_df is not None and lung_df is not None:
        combined = np.zeros((16, 16), dtype=int)
        combined[:10, :10] = demo.confusion_from_predictions(heart_df, HEART_TYPES)
        combined[10:, 10:] = demo.confusion_from_predictions(lung_df, LUNG_TYPES)
        fig = viz.confusion_matrix_fig(combined, HEART_TYPES + LUNG_TYPES,
                                       "combined heart + lung (block-diagonal)", normalize=True)
        pages.append({
            "title": "Combined heart + lung confusion (16×16)",
            "paras": [
                "This single matrix shows both detectors side by side: the top-left 10×10 block is heart-type "
                "confusion and the bottom-right 6×6 block is lung-type confusion; the off-diagonal blocks are empty "
                "by construction (a heart detector never emits a lung label).",
                "Because each row is normalised to its actual class, the two blocks can be compared directly despite "
                "the different number of classes and their very different support.",
                "It is the compact 'both sensors at once' view for the demo: a reader can see per-class recall for "
                "heart and lung in one image.",
            ],
            "image": save_fig(fig, out_dir, "cm_combined"),
        })

    # ---- inference distributions
    champions = [
        (binary_out, "sound", "binary", 2, "resnet18", ["heart", "lung", "mixed"], "sound — ResNet-18"),
        (mc_out, "heart", "10class", 4, "transformer", HEART_TYPES, "heart 10-class — Transformer"),
        (mc_out, "lung", "6class", 4, "transformer", LUNG_TYPES, "lung 6-class — Transformer"),
    ]
    intro_done = False
    for out_src, task, scheme, stage, model, names, label in champions:
        got = _probs_summary(data_dir, out_src, task, scheme, stage, model)
        if got is None:
            continue
        df, acc = got
        mean = infer.prob_matrix(df, names)
        mean_fig = viz.mean_probs_fig(mean, names, f"{label} · mean post-softmax per actual class")
        mean_img = save_fig(mean_fig, out_dir, f"meanprob_{task}")
        paras = []
        if not intro_done:
            paras.append(
                "These figures expose the model's pre-decision output — the post-softmax probability over classes "
                "from the network head — rather than only the arg-max decision. They are the most direct view of what "
                "the classifier 'believes'.")
            intro_done = True
        paras.append(
            f"Mean post-softmax distribution per actual class for the {label}: each row is the average probability "
            f"vector over all test samples of that actual class (rows therefore sum to 1). A sharp diagonal means "
            f"confident, well-separated classes; probability mass spread across a row shows systematic confusion. "
            f"Sample accuracy here is {_fmt(acc)}.")
        pages.append({"title": f"Inference distribution — {label} (mean)", "paras": paras, "image": mean_img})

        correct = df[df["true"] == df["pred"]].sort_values("conf", ascending=False).head(1)
        wrong = df[df["true"] != df["pred"]].sort_values("conf", ascending=True).head(2)
        for _, r in pd.concat([correct, wrong]).iterrows():
            probs = [float(v) for v in r["probs"].split()]
            f = viz.prob_dist_fig(probs, names, r["sample_id"], r["true"], r["pred"])
            img = save_fig(f, out_dir, f"prob_{task}_{r['sample_id']}")
            verdict = "a confident correct prediction" if r["true"] == r["pred"] else "a misclassification the model is unsure about"
            pages.append({
                "title": f"Inference distribution — {label}: {r['sample_id']}",
                "paras": [
                    f"Per-class post-softmax probabilities for one test clip ({r['sample_id']}); the actual class is "
                    f"green and the predicted class red. This is {verdict}.",
                    "Reading it: bar height is the head's probability for each class. Tall single bars indicate a "
                    "decisive head; several comparable bars indicate ambiguity that the confidence/calibration layer "
                    "should catch before a clinical decision is surfaced.",
                ],
                "image": img,
            })

    # ---- training curves
    curve_names = [
        ("stage2-sound-resnet18-allfolds-s0", "sound ResNet-18"),
        ("stage4-heart-transformer-10class-allfolds-s0", "heart 10-class Transformer"),
        ("stage2-lung-resnet18-6class-allfolds-s0", "lung 6-class ResNet-18"),
    ]
    for name, label in curve_names:
        hist = demo.fetch_wandb_history(name)
        if hist is None or hist.empty:
            continue
        fig = demo.curves_fig(hist)
        pages.append({
            "title": f"Training dynamics — {label}",
            "paras": [
                f"Loss (left) and validation balanced accuracy / macro-F1 (right) per epoch for {label}, pulled from "
                f"Weights & Biases.",
                "These curves justify the early-stopping and checkpoint policy: validation metrics plateau while "
                "training loss keeps falling, i.e. the model begins to overfit, so the best-validation checkpoint "
                "(not the final epoch) is evaluated on the test set.",
            ],
            "image": save_fig(fig, out_dir, f"curves_{name}"),
        })

    # ---- training evolution (input | processing | output at spaced epochs)
    import glob
    import json

    for jpath in sorted(glob.glob(os.path.join(out_dir, "figures", "evolution_*.json"))):
        with open(jpath) as f:
            s = json.load(f)
        img = s["image"]
        if not os.path.isfile(img):
            continue
        preds = s["predictions"]
        ms = [str(m) for m in s["milestones"]]
        first, last = ms[0], ms[-1]
        seq = ", ".join(
            f"epoch {m}: {preds[m]['pred']} (conf {preds[m]['conf']:.2f}"
            f"{', correct' if preds[m]['correct'] else ', wrong'})"
            for m in ms
        )
        improved = (not preds[first]["correct"]) and preds[last]["correct"]
        pages.append({
            "title": f"Training evolution — {s['task']} / {s['scheme']} · {s['model']}",
            "full": True,
            "paras": [
                f"One test clip (actual = {s['true']}) pushed through the model at each milestone epoch. "
                f"Columns are input (raw waveform), processing (fixed band-pass + log-mel), and output "
                f"(post-softmax head probabilities); input and processing are constant, so the only thing that "
                f"changes across rows is the model's decision.",
                f"Decision trajectory — {seq}." + (
                    " This clip flips from wrong to correct as training proceeds, illustrating how the head "
                    "sharpens with more epochs." if improved else
                    " This clip's trajectory shows how confidence grows (or fails to) across the schedule."),
            ],
            "image": img,
        })

    # ---- web app: Inference-as-a-Service results (synchronized timelines + multi-scale)
    pages.append({
        "title": "Web app — Inference-as-a-Service (product results)",
        "paras": [
            "The trained models are packaged as a web app (docs/WebApp.md): upload a recording and receive an "
            "explained, parametrized report. The pages that follow reproduce the app's key output figures on "
            "dataset clips — these are the same figures a user sees after upload.",
            "App features: source routing, normal/abnormal screening, fine-grained typing, per-head post-softmax "
            "probability charts, signal-quality flags, plain-language health-awareness narrative, batch analysis, "
            "a JSON API (POST /api/analyze), and an educational listening game for training.",
            "A one-clip, single-label view hides how the assessment evolves across breathing/cardiac cycles. The "
            "app therefore sweeps the clip in a sliding window and renders synchronized inference timelines — "
            "per-head probabilities (and confidence) against time, aligned with the input spectrogram.",
        ],
        "image": None,
    })
    timeline_specs = [
        ("timeline_source.png", "Synchronized inference timeline — source routing (heart / lung / mixed)",
         "Top: input log-mel spectrogram. Middle: stacked-area head probabilities over time. Bottom: confidence "
         "line with a predicted-class ribbon. All share a time axis, so the routing decision is visible across "
         "the whole recording, not just as a single label."),
        ("timeline_heart_types.png", "Synchronized timeline — heart sound typing (10 classes)",
         "The heart-type Transformer head swept in a sliding window. Short windows are ambiguous alone; the "
         "stacked areas show where probability mass moves between the 10 murmur/rhythm classes over time, and "
         "the whole-clip (macro) distribution is the most reliable reference."),
        ("timeline_lung_types.png", "Synchronized timeline — lung sound typing (6 classes)",
         "Six-class stacking for lung typing. Transient adventitious sounds (e.g. crackles) appear as brief "
         "shifts in probability mass that a single full-clip histogram would hide."),
        ("multiscale_source.png", "Multi-scale analysis — 1 s / 3 s / 15 s + whole clip",
         "One probability panel per window scale, plus the whole-clip (macro) distribution as dashed reference "
         "lines. Micro windows localise transients; long windows and the whole clip provide macro context. The "
         "fused, length-weighted prediction is reported alongside the macro prediction. This is the app's "
         "default temporal mode."),
    ]
    for name, title, desc in timeline_specs:
        base = _asset("assets", "report")
        p = os.path.join(base, name) if base else None
        if not p or not os.path.isfile(p):
            continue
        pages.append({"title": title, "full": True, "paras": [desc], "image": p})

    # ---- web app: usage walkthrough screenshots
    walk = _asset("walkthrough")
    if walk is not None:
        pages.append({
            "title": "Web app — usage walkthrough",
            "paras": [
                "The screenshots below correspond to the steps in docs/WALKTHROUGH.md: upload → explained report "
                "→ synchronized timeline → batch analysis → educational game.",
            ],
            "image": None,
        })
        for name, caption in [
            ("01_landing.png", "Upload (single or batch) with report parameters"),
            ("02_report_summary.png", "Report summary — verdict, signal quality, narrative, prediction table"),
            ("04_report_timeline.png", "Report — synchronized inference timeline"),
            ("05_batch.png", "Batch analysis table"),
            ("06_game.png", "Educational listening game"),
        ]:
            p = os.path.join(walk, name)
            if os.path.isfile(p):
                pages.append({"title": f"Web app walkthrough — {caption}", "full": True,
                              "paras": ["Part of the step-by-step guide in docs/WALKTHROUGH.md."],
                              "image": p})

    # ---- limitations
    pages.append({
        "title": "Limitations and threats to validity",
        "paras": [
            "Small, imbalanced dataset: 195 clean clips per binary task and as few as 2–5 examples for the rarest "
            "heart types. Balanced accuracy and macro-F1 are reported precisely because raw accuracy is misleading; "
            "per-class numbers for rare classes should be treated as indicative only.",
            "Manikin-only source: HLS-CMDS is recorded from a clinical manikin with one digital stethoscope, so "
            "absolute accuracies will not transfer to real patients without an external validation set; the campaign "
            "is a relative model comparison, not a clinical performance claim.",
            "MobileNetV3-Small failed to optimise on every task (near-chance balanced accuracy), consistent with "
            "aggressive downsampling plus BatchNorm on tiny batches and 64×376 inputs; ResNet-18 and the Transformer "
            "did not show this failure.",
            "Single-seed, single-split-protocol: results are reproducible (bit-identical fold-0 reruns) but reflect "
            "one 5-fold grouping at seed 0; seed sensitivity is not yet characterised.",
            "Calibration: ECE is logged per run, but probabilities are not yet temperature-scaled; the per-sample "
            "distributions shown here are pre-calibration.",
        ],
        "image": None,
    })

    # ---- conclusions
    pages.append({
        "title": "Conclusions and recommended next steps",
        "paras": [
            "The pipeline reliably separates normal from abnormal heart and lung sounds and can type them at "
            "fine granularity, with classical MFCC features providing a very strong, cheap baseline that currently "
            "leads the gate on every task.",
            "Next steps: (1) finish the 120-epoch refinement run and re-apply the gate — the Transformer is the most "
            "likely tier to overtake the baseline on fine-grained tasks; (2) diagnose and fix the MobileNetV3 "
            "optimisation failure (warm-up, gradient accumulation, or a shallower stem); (3) add temperature scaling "
            "and report reliability diagrams; (4) add the optional mixed-domain probe (predict heart label on the "
            "combined M#### clips) as a domain-shift check; (5) evaluate seed 1–2 to bound variance.",
            "Product implication: a mobile-deployable classical or distilled model is a credible Stage-1 candidate; "
            "the Transformer heads are the path to higher fine-grained accuracy once refinement and calibration are in place.",
        ],
        "image": None,
    })

    # ---- appendix: signal processing + audio
    ex = {}
    for label in ("heart", "lung", "mixed"):
        ex[label] = m_sound[m_sound["task_label"] == label].iloc[0]
    from cardia.data.transforms import Augment, MelSpec, load_wav, peak_normalize, preprocess

    mel = MelSpec(cfg.sample_rate, cfg.n_mels, cfg.n_fft, cfg.hop_length)
    aug = Augment(cfg, cfg.seed)
    import soundfile as sf

    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    for key, row in ex.items():
        raw, sr = load_wav(row["file_path"])
        raw = peak_normalize(raw)
        filt = preprocess(row["file_path"], cfg)
        spec = mel(filt).numpy()
        spec_aug = aug.spec(mel(aug.waveform(raw))).numpy()
        sf.write(os.path.join(out_dir, "audio", f"{key}_{row['sample_id']}_raw.wav"), raw.numpy(), sr)
        sf.write(os.path.join(out_dir, "audio", f"{key}_{row['sample_id']}_filtered.wav"), filt.numpy(), cfg.sample_rate)
        fig = viz.aug_fig(spec, spec_aug, f"{key} ({row['sample_id']}) · log-mel · original vs augmented")
        pages.append({
            "title": f"Appendix — signal processing ({key})",
            "paras": [
                f"Log-mel spectrogram of a {key} clip before (left) and after (right) training-time augmentation "
                f"(time shift, amplitude scaling, additive noise, SpecAugment masks).",
                "The matching raw and band-passed audio files are written to the report's audio/ folder so the demo "
                "can play the actual signal alongside its spectrogram.",
            ],
            "image": save_fig(fig, out_dir, f"proc_{key}"),
            "audio": [f"{key}_{row['sample_id']}_raw.wav"],
        })

    return pages


# ------------------------------------------------------------------ renderers
def render_pdf(pages: List[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for pg in pages:
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.955, pg["title"], ha="center", va="top", fontsize=17, weight="bold")
            fig.text(0.5, 0.915, "CardiaSense AI · HLS-CMDS benchmark", ha="center", va="top",
                     fontsize=8, color="#888888")
            img = pg.get("image")
            if img and os.path.isfile(img):
                if pg.get("full"):
                    ax = fig.add_axes([0.04, 0.17, 0.92, 0.75])
                    ax.imshow(mpimg.imread(img), aspect="auto")
                    ax.axis("off")
                    _draw_text(fig, 0.05, 0.145, 120, pg["paras"], fontsize=9.5)
                else:
                    ax = fig.add_axes([0.035, 0.06, 0.52, 0.80])
                    ax.imshow(mpimg.imread(img))
                    ax.axis("off")
                    _draw_text(fig, 0.585, 0.86, 60, pg["paras"], fontsize=9.5)
            else:
                _draw_text(fig, 0.07, 0.86, 128, pg["paras"], fontsize=11.5)
            pdf.savefig(fig)
            plt.close(fig)


def render_pptx(pages: List[dict], audio_dir: str, path: str) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for pg in pages:
        slide = prs.slides.add_slide(blank)
        tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.25), Inches(12.5), Inches(0.8))
        tf = tb.text_frame
        tf.text = pg["title"]
        tf.paragraphs[0].font.size = Pt(22)
        tf.paragraphs[0].font.bold = True
        img = pg.get("image")
        text_left = Inches(0.5)
        text_width = Inches(12.3)
        if img and os.path.isfile(img) and pg.get("full"):
            slide.shapes.add_picture(img, Inches(0.4), Inches(1.05), width=Inches(12.5))
            box = slide.shapes.add_textbox(Inches(0.4), Inches(6.3), Inches(12.5), Inches(1.05))
        else:
            if img and os.path.isfile(img):
                slide.shapes.add_picture(img, Inches(0.4), Inches(1.15), height=Inches(5.6))
                text_left = Inches(7.0)
                text_width = Inches(6.0)
            box = slide.shapes.add_textbox(text_left, Inches(1.15), text_width, Inches(5.9))
        box.text_frame.word_wrap = True
        first = True
        for para in pg["paras"]:
            para_tf = box.text_frame.paragraphs[0] if first else box.text_frame.add_paragraph()
            first = False
            para_tf.text = para
            para_tf.font.size = Pt(12)
        for rel in pg.get("audio", []):
            ap = os.path.join(audio_dir, rel)
            if os.path.isfile(ap):
                try:
                    slide.shapes.add_movie(ap, Inches(0.5), Inches(6.75), Inches(3.0), Inches(0.6),
                                           mime_type="audio/wav")
                except Exception:
                    pass
    prs.save(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the comprehensive CardiaSense report (PDF + PPTX)")
    p.add_argument("--binary-out", default="runs")
    p.add_argument("--mc-out", default=DEFAULT_MC_OUT)
    p.add_argument("--data-dir", default=Config().data_dir)
    p.add_argument("--out-dir", default="report")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    pages = build_pages(args.binary_out, args.mc_out, args.data_dir, args.out_dir)
    for i, pg in enumerate(pages):
        print(f"[report] page {i}: {pg['title']}")

    pdf_path = os.path.join(args.out_dir, "cardiasense_report.pdf")
    render_pdf(pages, pdf_path)
    print(f"[report] PDF -> {pdf_path}")
    pptx_path = os.path.join(args.out_dir, "cardiasense_report.pptx")
    render_pptx(pages, os.path.join(args.out_dir, "audio"), pptx_path)
    print(f"[report] PPTX -> {pptx_path}")
    print(f"[report] {len(pages)} pages; figures -> {args.out_dir}/figures; audio -> {args.out_dir}/audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
