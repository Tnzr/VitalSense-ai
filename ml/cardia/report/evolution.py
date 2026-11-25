"""Training-evolution figure: input -> processing -> output probabilities at
spaced epochs, for a single sample, so model improvement is visible.

Trains one task/model/fold for a short schedule, capturing the post-softmax
head output on the test set at a set of milestone epochs, then renders a
labelled grid (rows = epochs, columns = input | processing | output).

Usage:
  python -m cardia.report.evolution --task heart --scheme 10class \
      --model transformer --stage 4 --epochs 60 --out-dir report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ML not in sys.path:
    sys.path.insert(0, ML)

from cardia.config import Config
from cardia.data.dataset import HLSDataset
from cardia.data.hls_cmds import build_task_manifest, class_names
from cardia.data.splits import load_splits, make_splits
from cardia.data.transforms import MelSpec, load_wav, peak_normalize, preprocess
from cardia.models.cnn import build_cnn
from cardia.models.temporal import build_temporal
from cardia.models.transformer import build_transformer
from cardia.train import evaluate as ev
from cardia.wandb_utils import viz

DEFAULT_MILESTONES = [1, 5, 20, 60]


def _collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "group": [b["group"] for b in batch],
    }


def _builder(stage: int):
    return {2: build_cnn, 3: build_temporal, 4: build_transformer}[stage]


def _probs_on(model, loader, device) -> dict:
    model.eval()
    out = {}
    with torch.no_grad():
        for batch in loader:
            p = torch.softmax(model(batch["x"].to(device)), dim=1).cpu().numpy()
            for sid, row in zip(batch["sample_id"], p):
                out[sid] = row
    return out


def run_evolution(cfg: Config, milestones: List[int], sample_id: str | None, out_dir: str) -> dict:
    device = torch.device(cfg.resolve_device())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    manifest = build_task_manifest(cfg.data_dir, cfg.task, cfg.class_scheme, os.path.join(out_dir, "manifests"))
    split_dir = os.path.join(out_dir, "splits", f"{cfg.task}_{cfg.class_scheme}_s{cfg.seed}")
    splits = load_splits(split_dir, n_folds=cfg.n_folds) if os.path.isdir(split_dir) else \
        make_splits(manifest, n_folds=cfg.n_folds, seed=cfg.seed, out_dir=out_dir, split_dir=split_dir)
    sp = splits[cfg.fold]
    names = class_names(cfg.task, cfg.class_scheme)

    def ds(ids, aug):
        m = manifest.set_index("sample_id").loc[ids].reset_index()
        return HLSDataset(m, cfg, augment=aug)

    train_ds, test_ds = ds(sp.train, True), ds(sp.test, False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                               collate_fn=_collate, generator=torch.Generator().manual_seed(cfg.seed))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)

    counts = np.bincount(train_ds.manifest["target"].to_numpy(), minlength=len(names))
    weights = ev.class_weights_from_counts(counts, len(names), device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = _builder(cfg.stage)(cfg, len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(milestones), eta_min=cfg.lr / 100.0)

    snaps = {}
    max_ep = max(milestones)
    for epoch in range(1, max_ep + 1):
        train_ds.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = criterion(model(batch["x"].to(device)), batch["y"].to(device))
            loss.backward()
            opt.step()
        sched.step()
        if epoch in milestones:
            snaps[epoch] = _probs_on(model, test_loader, device)

    # choose a sample that flips from wrong -> right across the schedule
    first, last = milestones[0], milestones[-1]
    ids = list(snaps[last].keys())
    chosen = sample_id
    if chosen is None:
        for sid in ids:
            if snaps[first][sid].argmax() != snaps[last][sid].argmax():
                chosen = sid
                break
        if chosen is None:
            chosen = ids[0]

    row = manifest.set_index("sample_id").loc[chosen]
    raw, sr = load_wav(row["file_path"])
    raw = peak_normalize(raw)
    filt = preprocess(row["file_path"], cfg)
    mel = MelSpec(cfg.sample_rate, cfg.n_mels, cfg.n_fft, cfg.hop_length)
    spec = mel(filt).numpy()

    true_idx = int(row["target"])
    fig = _evolution_fig(raw.numpy(), sr, spec, snaps, milestones, names, chosen,
                         names[true_idx], cfg)

    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    img = os.path.join(out_dir, "figures", f"evolution_{cfg.task}_{cfg.class_scheme}.png")
    fig.savefig(img, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)

    summary = {
        "task": cfg.task, "scheme": cfg.class_scheme, "model": cfg.model, "stage": cfg.stage,
        "sample_id": chosen, "true": names[true_idx], "image": img,
        "milestones": milestones,
        "predictions": {
            str(e): {
                "pred": names[int(snaps[e][chosen].argmax())],
                "conf": float(snaps[e][chosen].max()),
                "correct": bool(snaps[e][chosen].argmax() == true_idx),
            }
            for e in milestones
        },
    }
    with open(os.path.join(out_dir, "figures", f"evolution_{cfg.task}_{cfg.class_scheme}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _evolution_fig(raw, sr, spec, snaps, milestones, names, sample_id, true_label, cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(milestones)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.0 * n))
    t = np.arange(len(raw)) / sr
    for i, e in enumerate(milestones):
        # input
        ax = axes[i, 0]
        ax.plot(t, raw, lw=0.5, color="#4C72B0")
        ax.set_ylabel("amplitude")
        ax.set_xlabel("time (s)")
        ax.set_title(f"input · raw waveform · epoch {e}", fontsize=10)
        # processing
        ax = axes[i, 1]
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        ax.set_ylabel("mel bin")
        ax.set_xlabel("frame")
        ax.set_title(f"processing · band-pass + log-mel · epoch {e}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        # output
        ax = axes[i, 2]
        p = snaps[e][sample_id]
        pred = names[int(p.argmax())]
        correct = pred == true_label
        colors = ["#4C72B0"] * len(names)
        if true_label in names:
            colors[names.index(true_label)] = "#55A868"
        if pred in names and pred != true_label:
            colors[names.index(pred)] = "#C44E52"
        ax.bar(range(len(names)), p, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("post-softmax prob")
        ax.set_xlabel("class")
        labels = [c if len(c) <= 12 else c[:10] + ".." for c in names]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(
            f"output · epoch {e} · pred={pred} conf={p.max():.2f} "
            f"{'✓' if correct else '✗'} (actual={true_label})", fontsize=10,
        )
    fig.suptitle(
        f"Training evolution — {cfg.task}/{cfg.class_scheme} · {cfg.model} · sample {sample_id} · "
        f"actual={true_label}", fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the input/processing/output training-evolution figure")
    p.add_argument("--task", choices=("heart", "lung", "sound"), default="heart")
    p.add_argument("--scheme", default="10class")
    p.add_argument("--model", default="transformer")
    p.add_argument("--stage", type=int, default=4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--milestones", nargs="+", type=int, default=None)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default=Config().data_dir)
    p.add_argument("--out-dir", default="report")
    args = p.parse_args(argv)

    milestones = args.milestones or [m for m in DEFAULT_MILESTONES if m <= args.epochs]
    if milestones[-1] != args.epochs:
        milestones.append(args.epochs)
    cfg = Config(task=args.task, stage=args.stage, model=args.model, class_scheme=args.scheme,
                 fold=args.fold, epochs=args.epochs, seed=args.seed, device=args.device,
                 data_dir=args.data_dir, out_dir=args.out_dir)
    summary = run_evolution(cfg, milestones, args.sample_id, args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
