"""Train the mask separator on synthetic heart+lung mixtures.

    python -m cardia.separation.train --epochs 40

Validation: fixed synthetic mixes from held-out recordings. Test: the 36 exact-sum
HLS-CMDS mixtures (no shared recordings with train/val)."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..config import DEFAULT_DATA_DIR
from . import data, metrics
from .evaluate import DEFAULT_OUT, val_triplets
from .model import Separator, si_sdr_loss


def _sep_fn(model):
    def f(x, sr):
        with torch.no_grad():
            h, l = model(torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0))
        return h[0].numpy(), l[0].numpy()
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--items", type=int, default=1600, help="synthetic mixtures per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    pools = data.split_stems(a.data_dir, seed=a.seed)
    train = data.SyntheticMixes(pools["train"], a.items, seconds=a.seconds)
    loader = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=2,
                                         persistent_workers=True)
    val = val_triplets(a.data_dir)
    model = Separator()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(loader))
    ckpt = os.path.join(a.out, "separator.pt")
    best, history = -1e9, []
    print(f"train recordings: {len(pools['train'])}  val: {len(pools['val'])}  "
          f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    for ep in range(1, a.epochs + 1):
        t0, losses = time.time(), []
        model.train()
        for mix, h, l in loader:
            eh, el = model(mix)
            loss = 0.5 * (si_sdr_loss(eh, h) + si_sdr_loss(el, l))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        v = metrics.evaluate(_sep_fn(model), val)["summary"]
        score = 0.5 * (v["heart_si_sdri"] + v["lung_si_sdri"])
        history.append({"epoch": ep, "loss": float(np.mean(losses)), **v})
        tag = ""
        if score > best:
            best, tag = score, " *"
            torch.save({"state_dict": model.state_dict(), "model_kwargs": {}, "epoch": ep,
                        "val": v, "args": vars(a)}, ckpt)
        print(f"ep {ep:3d} loss {np.mean(losses):7.2f}  val SI-SDRi heart {v['heart_si_sdri']:6.2f} "
              f"lung {v['lung_si_sdri']:6.2f}  ({time.time() - t0:.0f}s){tag}", flush=True)

    model.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"])
    test = metrics.evaluate(_sep_fn(model.eval()), data.test_triplets(a.data_dir))
    json.dump({"history": history, "test": test, "best_val": best}, open(os.path.join(a.out, "train_metrics.json"), "w"),
              indent=1)
    s = test["summary"]
    print(f"TEST (36 exact mixes): heart SI-SDR {s['heart_si_sdr']:.2f} (SI-SDRi {s['heart_si_sdri']:.2f})  "
          f"lung SI-SDR {s['lung_si_sdr']:.2f} (SI-SDRi {s['lung_si_sdri']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
