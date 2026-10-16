"""Automated demo packager: gathers figures + audio and produces a PDF and a
PowerPoint deck summarizing the HLS-CMDS benchmark campaign.

Reads local campaign outputs (metrics.json, benchmark.json, checkpoints/
predictions CSVs) from the binary and multiclass runs, recomputes input/
processing figures and confusion matrices, optionally pulls training curves
from WandB, and writes:

  <demo-dir>/cardiasense_demo.pdf
  <demo-dir>/cardiasense_demo.pptx
  <demo-dir>/figures/*.png
  <demo-dir>/audio/*.wav

Usage:
  python -m cardia.report.demo \
      --binary-out ml/runs --mc-out ml/runs \
      --demo-dir demo
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ML not in sys.path:
    sys.path.insert(0, ML)

from cardia.config import DEFAULT_MC_OUT, Config, HEART_TYPES, LUNG_TYPES
from cardia.data.hls_cmds import build_task_manifest, class_names
from cardia.data.transforms import Augment, MelSpec, load_wav, peak_normalize
from cardia.wandb_utils import viz

PROJECT = "cardiasense-hls-cmds"


# ---------------------------------------------------------------- data utils
def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    import json

    with open(path) as f:
        return json.load(f)


def benchmark_rows(binary_out: str, mc_out: str) -> pd.DataFrame:
    rows = []

    def collect(out_dir: str, scheme: str) -> None:
        bench = _load_json(os.path.join(out_dir, "benchmark.json")).get("benchmark", [])
        for r in bench:
            if r.get("best_test_balanced_accuracy") is None:
                continue
            rows.append(
                {
                    "task": r["task"], "scheme": scheme, "stage": r["stage"],
                    "bal_acc": r["best_test_balanced_accuracy"],
                    "macro_f1": r.get("best_test_macro_f1"),
                    "gate": r.get("gate_decision", ""),
                }
            )

    collect(binary_out, "binary")
    collect(mc_out, "multiclass")
    return pd.DataFrame(rows)


def predictions_files(out_dir: str, task: str, scheme: str, stage: int, model: str) -> List[str]:
    suf = f"-{scheme}" if scheme != "binary" else ""
    pats = [
        os.path.join(out_dir, "checkpoints", f"stage{stage}-{task}-{model}{suf}-allfolds-s*-fold*_predictions.csv"),
        os.path.join(out_dir, "checkpoints", f"stage{stage}-{task}-{model}{suf}-fold*-s*_predictions.csv"),
        os.path.join(out_dir, "checkpoints", f"stage{stage}-{task}-classical-{model}{suf}-s*_predictions.csv"),
    ]
    return sorted({f for p in pats for f in glob.glob(p)})


def load_predictions(out_dir: str, task: str, scheme: str, stage: int, model: str) -> Optional[pd.DataFrame]:
    files = predictions_files(out_dir, task, scheme, stage, model)
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def confusion_from_predictions(df: pd.DataFrame, names: List[str]) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(df["true"], df["pred"], labels=names).astype(int)


def fetch_wandb_history(run_name: str) -> Optional[pd.DataFrame]:
    """Pull epoch history for a run by display name (best-effort, online only)."""
    try:
        import wandb

        api = wandb.Api()
        if not api.api_key:
            return None
        runs = api.runs(path=PROJECT, filters={"display_name": run_name})
        if not runs:
            return None
        run = runs[0]
        return run.history()
    except Exception:
        return None


# ---------------------------------------------------------------- figure utils
def title_fig(title: str, subtitle: str) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.62, title, ha="center", va="center", fontsize=26, weight="bold")
    fig.text(0.5, 0.42, subtitle, ha="center", va="center", fontsize=13, color="#333333")
    fig.text(0.5, 0.28, f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}", ha="center",
             va="center", fontsize=10, color="#666666")
    return fig


def table_fig(df: pd.DataFrame, title: str) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, max(2.2, 0.4 * (len(df) + 2))))
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold")
    t = ax.table(cellText=df.round(3).values.astype(str), colLabels=df.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    fig.tight_layout()
    return fig


def curves_fig(history: pd.DataFrame) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history.get("train/loss", []), label="train")
    axes[0].plot(history["epoch"], history.get("val/loss", []), label="val")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history.get("val/balanced_accuracy", []), label="val bal-acc")
    axes[1].plot(history["epoch"], history.get("val/macro_f1", []), label="val macro-F1")
    axes[1].set_title("validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- pptx renderer
def render_pptx(slides: List[dict], audio_dir: str, path: str) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    for s in slides:
        slide = prs.slides.add_slide(blank)
        if slide.shapes.title:
            slide.shapes.title.text = s.get("title", "")
        if "fig_path" in s:
            slide.shapes.add_picture(s["fig_path"], Inches(0.5), Inches(1.1), width=Inches(12.3))
        if "text" in s:
            tb = slide.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(12.3), Inches(5.8))
            tb.text_frame.word_wrap = True
            tb.text_frame.text = s["text"]
        if s.get("audio"):
            for rel, poster in s["audio"]:
                audio_path = os.path.join(audio_dir, rel)
                poster_path = os.path.join(os.path.dirname(path), "figures", poster)
                if not os.path.exists(poster_path):
                    poster_path = None
                try:
                    slide.shapes.add_movie(audio_path, Inches(0.5), Inches(6.4), Inches(2.4), Inches(0.8),
                                           poster_frame_image=poster_path, mime_type="audio/wav")
                except Exception:
                    pass  # audio folder shipped alongside; caption still references it

    prs.save(path)


# ---------------------------------------------------------------- main
def build_slides(binary_out: str, mc_out: str, data_dir: str, demo_dir: str,
                 curves_names: List[str]) -> tuple:
    """Return (slides, audio_rel_paths). slides = [{title, fig, text, audio}]."""
    os.makedirs(os.path.join(demo_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "audio"), exist_ok=True)
    from cardia.report import infer

    cfg = Config(data_dir=data_dir)

    # --- dataset overview
    m_sound = build_task_manifest(data_dir, "sound", "binary", os.path.join(demo_dir, "manifests"))
    dist = m_sound.groupby("task_label").size().reset_index(name="count")
    dist["share"] = dist["count"] / dist["count"].sum()

    slides = [{
        "title": "CardiaSense — HLS-CMDS Benchmark",
        "fig": title_fig("CardiaSense AI PoC", "HLS-CMDS auscultation benchmark · 3 tasks × 4 model stages · WandB-tracked"),
    }]
    slides.append({"title": "Dataset overview", "fig": table_fig(dist, "HLS-CMDS dataset — sound-task class split (535 clips)")})

    # --- benchmark summary first (classification results up front)
    bench = benchmark_rows(binary_out, mc_out)
    slides.append({"title": "Benchmark summary", "fig": table_fig(bench, "Benchmark — best test balanced accuracy / macro-F1 per stage")})

    # --- confusion matrices (normalized relative to ACTUAL class)
    def add_cm(out_dir: str, task: str, scheme: str, stage: int, model: str, names: List[str], label: str) -> None:
        df = load_predictions(out_dir, task, scheme, stage, model)
        if df is None:
            return
        cm = confusion_from_predictions(df, names)
        slides.append({
            "title": label,
            "fig": viz.confusion_matrix_fig(cm, names, label, normalize=True),
        })

    add_cm(binary_out, "heart", "binary", 2, "resnet18", ["normal", "abnormal"], "Heart normal/abnormal (ResNet-18)")
    add_cm(binary_out, "lung", "binary", 2, "resnet18", ["normal", "abnormal"], "Lung normal/abnormal (ResNet-18)")
    add_cm(binary_out, "sound", "binary", 2, "resnet18", ["heart", "lung", "mixed"], "Heart/lung/mixed (ResNet-18)")
    add_cm(mc_out, "heart", "10class", 2, "resnet18", HEART_TYPES, "Heart sound types — 10 classes (ResNet-18)")
    add_cm(mc_out, "lung", "6class", 2, "resnet18", LUNG_TYPES, "Lung sound types — 6 classes (ResNet-18)")

    heart_df = load_predictions(mc_out, "heart", "10class", 2, "resnet18")
    lung_df = load_predictions(mc_out, "lung", "6class", 2, "resnet18")
    if heart_df is not None and lung_df is not None:
        hcm = confusion_from_predictions(heart_df, HEART_TYPES)
        lcm = confusion_from_predictions(lung_df, LUNG_TYPES)
        combined = np.zeros((16, 16), dtype=int)
        combined[:10, :10] = hcm
        combined[10:, 10:] = lcm
        slides.append({
            "title": "Combined heart + lung confusion (block-diagonal 16×16)",
            "fig": viz.confusion_matrix_fig(combined, HEART_TYPES + LUNG_TYPES,
                                            "combined heart + lung (block-diagonal)", normalize=True),
        })

    # --- post-softmax inference distributions from the heads
    champions = [
        (binary_out, "sound", "binary", 2, "resnet18", ["heart", "lung", "mixed"], "Sound — ResNet-18 heads"),
        (mc_out, "heart", "10class", 4, "transformer", HEART_TYPES, "Heart 10-class — Transformer heads"),
        (mc_out, "lung", "6class", 4, "transformer", LUNG_TYPES, "Lung 6-class — Transformer heads"),
    ]
    for out_dir, task, scheme, stage, model, names, label in champions:
        try:
            df = infer.infer_test_probs(data_dir, out_dir, task, scheme, stage, model, seed=0)
        except Exception as e:
            print(f"[demo] skip prob slides {task}/{scheme}/{model}: {e}")
            continue
        mean = infer.prob_matrix(df, names)
        slides.append({
            "title": f"{label} — mean post-softmax per actual class",
            "fig": viz.mean_probs_fig(mean, names,
                                      f"{label} · mean post-softmax distribution per actual class"),
        })
        # one confident correct + two most-confused misclassifications
        correct = df[df["true"] == df["pred"]].sort_values("conf", ascending=False).head(1)
        wrong = df[df["true"] != df["pred"]].sort_values("conf", ascending=True).head(2)
        for _, r in pd.concat([correct, wrong]).iterrows():
            probs = [float(v) for v in r["probs"].split()]
            slides.append({
                "title": f"{label} — inference distribution: {r['sample_id']}",
                "fig": viz.prob_dist_fig(probs, names, r["sample_id"], r["true"], r["pred"]),
            })

    # --- training curves (best-effort from WandB)
    for name in curves_names:
        hist = fetch_wandb_history(name)
        if hist is None or hist.empty:
            continue
        slides.append({"title": f"Training curves — {name}", "fig": curves_fig(hist)})

    # --- appendix: signal processing + audio clips
    m_heart = build_task_manifest(data_dir, "heart", "10class", os.path.join(demo_dir, "manifests"))
    m_lung = build_task_manifest(data_dir, "lung", "6class", os.path.join(demo_dir, "manifests"))
    ex = {}
    for label in ("heart", "lung", "mixed"):
        ex[label] = m_sound[m_sound["task_label"] == label].iloc[0]
    ex["heart_abnormal"] = m_heart[m_heart["task_label"] != "Normal"].iloc[0]
    ex["heart_normal"] = m_heart[m_heart["task_label"] == "Normal"].iloc[0]
    ex["lung_abnormal"] = m_lung[m_lung["task_label"] != "Normal"].iloc[0]
    ex["lung_normal"] = m_lung[m_lung["task_label"] == "Normal"].iloc[0]

    mel = MelSpec(cfg.sample_rate, cfg.n_mels, cfg.n_fft, cfg.hop_length)
    aug = Augment(cfg, cfg.seed)
    audio_rel = []
    for key, row in ex.items():
        raw, sr = load_wav(row["file_path"])
        raw = peak_normalize(raw)
        from cardia.data.transforms import preprocess

        filt = preprocess(row["file_path"], cfg)
        spec = mel(filt).numpy()
        wav_aug = aug.waveform(raw)
        spec_aug = aug.spec(mel(wav_aug)).numpy()
        stem = f"{key}_{row['sample_id']}"
        wav_raw = os.path.join(demo_dir, "audio", f"{stem}_raw.wav")
        wav_filt = os.path.join(demo_dir, "audio", f"{stem}_filtered.wav")
        import soundfile as sf

        sf.write(wav_raw, raw.numpy(), sr)
        sf.write(wav_filt, filt.numpy(), cfg.sample_rate)

        fig = viz.aug_fig(spec, spec_aug, f"{stem} · log-mel · {row['task_label']}")
        slides.append({
            "title": f"Appendix · signal processing: {stem}",
            "fig": fig,
            "audio": [(f"{stem}_raw.wav", f"{stem}_raw.png")],
        })
        audio_rel.append((wav_raw, wav_filt, stem))

    return slides, audio_rel


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the CardiaSense demo PDF + PPTX")
    p.add_argument("--binary-out", default="runs", help="binary campaign out dir (metrics.json, checkpoints/)")
    p.add_argument("--mc-out", default=DEFAULT_MC_OUT, help="multiclass campaign out dir")
    p.add_argument("--data-dir", default=Config().data_dir, help="HLS-CMDS dataset root")
    p.add_argument("--demo-dir", default="demo", help="output folder for pdf/pptx/figures/audio")
    p.add_argument("--curves", nargs="+", default=[
        "stage2-sound-resnet18-allfolds-s0",
        "stage4-heart-transformer-10class-allfolds-s0",
        "stage2-lung-resnet18-6class-allfolds-s0",
    ], help="WandB run names to pull training curves from (best-effort)")
    args = p.parse_args(argv)

    slides, audio_rel = build_slides(args.binary_out, args.mc_out, args.data_dir, args.demo_dir, args.curves)

    # PDF + slide PNGs in one pass, closing each figure after it is rendered
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["figure.max_open_warning"] = 100  # figures closed after each page
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(args.demo_dir, "cardiasense_demo.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, s in enumerate(slides):
            fig = s["fig"]
            fig_path = os.path.join(args.demo_dir, "figures", f"slide{i:02d}.png")
            fig.savefig(fig_path, dpi=110, bbox_inches="tight")
            s["fig_path"] = fig_path
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"[demo] slide {i}: {s['title']}")
    print(f"[demo] PDF -> {pdf_path}")

    # PPTX
    pptx_path = os.path.join(args.demo_dir, "cardiasense_demo.pptx")
    render_pptx(slides, os.path.join(args.demo_dir, "audio"), pptx_path)
    print(f"[demo] PPTX -> {pptx_path}")
    print(f"[demo] figures -> {args.demo_dir}/figures, audio -> {args.demo_dir}/audio ({len(audio_rel)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
