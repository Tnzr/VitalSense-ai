#!/usr/bin/env python3
"""Extract CLS embeddings from the heart-10class Transformer head and show how
the encoder organises class embeddings (cosine similarity between class means).
Also saves the per-clip embeddings (.npz) for downstream LLM experiments.

Usage:
  python scripts/extract_embeddings.py \
      --data-dir <dataset> --mc-out <multiclass-campaign-dir> \
      --out-dir runs/embeddings
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))

from cardia.config import DEFAULT_MC_OUT, HEART_TYPES  # noqa: E402
from cardia.features import embeddings as emb  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Extract audio CLS embeddings for LLM integration")
    p.add_argument("--data-dir", default="")
    p.add_argument("--mc-out", default=DEFAULT_MC_OUT)
    p.add_argument("--out-dir", default="runs/embeddings")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    data = emb.corpus_embeddings(args.data_dir or os.environ.get("ASCULTO_DATA_DIR", ""),
                                 args.mc_out, device=args.device)
    cls = np.stack(data["rows"]["cls"])
    np.savez(os.path.join(args.out_dir, "embeddings_heart_10class.npz"),
             cls=cls, sample_id=data["rows"]["sample_id"], label=data["rows"]["label"],
             classes=np.array(HEART_TYPES))
    print(f"[embeddings] {len(cls)} clips, shape {cls.shape} -> {args.out_dir}/embeddings_heart_10class.npz")

    M = emb.class_mean_matrix(data)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(HEART_TYPES)))
    ax.set_yticks(range(len(HEART_TYPES)))
    labels = [c if len(c) <= 14 else c[:12] + ".." for c in HEART_TYPES]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("class-mean CLS embedding")
    ax.set_ylabel("class-mean CLS embedding")
    ax.set_title("Heart-type class embeddings — cosine similarity (encoder-organized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("cosine similarity")
    fig.tight_layout()
    asset = os.path.join(ROOT, "docs", "assets", "report", "embeddings_heart.png")
    os.makedirs(os.path.dirname(asset), exist_ok=True)
    fig.savefig(asset, dpi=130, bbox_inches="tight")
    print(f"[embeddings] class-similarity figure -> {asset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())