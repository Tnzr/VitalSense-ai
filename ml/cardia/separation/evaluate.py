"""Score separators on the exact-sum test mixes or synthetic validation mixes.

    python -m cardia.separation.evaluate --methods identity,band,repet_band,repet_sim_band --split test
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from ..config import DEFAULT_DATA_DIR, _REPO
from . import classical, data, metrics

DEFAULT_OUT = os.path.join(_REPO, "ml", "runs", "sep")


def val_triplets(data_dir: str, n: int = 48, seconds: float = 15.0):
    ds = data.SyntheticMixes(data.split_stems(data_dir)["val"], n, seconds=seconds, fixed=True, seed=123)
    out = []
    for i in range(n):
        m, h, l = (t.numpy() for t in ds[i])
        out.append((SimpleNamespace(mix_id=f"val{i:03d}"), m, h, l))
    return out


def separator(method: str):
    if method == "identity":  # unprocessed mixture as both estimates
        return lambda x, sr: (x, x)
    if method == "model":
        from .model import load_separator
        return load_separator()
    return lambda x, sr: classical.separate(x, sr, method)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--methods", default="identity,band,repet,repet_band,repet_sim,repet_sim_band")
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    trip = data.test_triplets(a.data_dir) if a.split == "test" else val_triplets(a.data_dir)
    results = {}
    print(f"{a.split}: {len(trip)} mixtures")
    print(f"{'method':<16}{'heart SI-SDR':>14}{'lung SI-SDR':>13}{'heart SI-SDRi':>15}{'lung SI-SDRi':>14}")
    for m in a.methods.split(","):
        r = metrics.evaluate(separator(m), trip)
        results[m] = r
        s = r["summary"]
        print(f"{m:<16}{s['heart_si_sdr']:>14.2f}{s['lung_si_sdr']:>13.2f}{s['heart_si_sdri']:>15.2f}{s['lung_si_sdri']:>14.2f}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"eval_{a.split}.json")
    prev = json.load(open(path)) if os.path.isfile(path) else {}
    prev.update(results)
    json.dump(prev, open(path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
