"""Render per-class log-mel galleries (heart 10 types, lung 6 types) to docs/assets/.

Usage:  python scripts/make_mel_gallery.py   (needs data/HLS-CMDS; see ./setup.sh)
"""
from __future__ import annotations

import base64
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from webapp.app import matrix  # noqa: E402

OUT = os.path.join(ROOT, "docs", "assets")


def render(domain: dict, path: str, seconds: float = 6.0) -> None:
    tiles = [t for t in domain["tiles"] if t["sample_ids"]]
    cols = 2
    rows = int(np.ceil(len(tiles) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.2 * rows), squeeze=False)
    for ax, t in zip(axes.flat, tiles):
        p = matrix.mel_for(t["sample_ids"][0], domain["task"], domain["scheme"])
        mel = p["mel"]
        q = np.frombuffer(base64.b64decode(mel["data"]), dtype=np.uint8).reshape(mel["n_mels"], mel["frames"])
        n = min(int(seconds / mel["hop_s"]), mel["frames"])
        db = mel["db_min"] + q[:, :n] / 255.0 * (mel["db_max"] - mel["db_min"])
        ax.imshow(db, origin="lower", aspect="auto", cmap="inferno",
                  extent=[0, n * mel["hop_s"], 0, mel["n_mels"]])
        hz = np.asarray(mel["hz"])
        targets = (100, 250, 500, 1000, 1500)
        ax.set_yticks([int(np.abs(hz - f).argmin()) + 0.5 for f in targets], [str(f) for f in targets], fontsize=7)
        ax.set_title(f"{t['label']}  ({t['sample_ids'][0]})", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in list(axes.flat)[len(tiles):]:
        ax.axis("off")
    fig.suptitle(f"{domain['title']} — log-mel spectrograms (first {seconds:.0f} s, Hz on y-axis)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for d in matrix.tiles():
        if d.get("error"):
            sys.exit(f"{d['title']}: {d['error']}")
        render(d, os.path.join(OUT, f"mel_gallery_{d['task']}.png"))


if __name__ == "__main__":
    main()
