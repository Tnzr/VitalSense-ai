"""Do separated stems help condition typing on HLS-CMDS mixtures?

For each Mix.csv row, run the served heart/lung typing heads on (a) the raw mixture,
(b) the separator's heart/lung estimate, (c) the true source recording, and report
balanced accuracy against the row's heart/lung type labels.

Caveat: the served heads were trained on folds that include the Mix sources, so (c) is
optimistic; compare (a) vs (b), which see the same models.

    python scripts/eval_separation_downstream.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "ml")]

from cardia.config import DEFAULT_DATA_DIR  # noqa: E402
from cardia.data import hls_cmds  # noqa: E402
from cardia.separation import data as sdata  # noqa: E402
from cardia.separation.model import load_separator  # noqa: E402
from webapp.app.inference import get_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "ml", "runs", "sep", "downstream.json"))
    a = ap.parse_args()
    reg = get_registry()
    sep = load_separator()
    rows = hls_cmds._read_csv(DEFAULT_DATA_DIR, "Mix.csv").to_dict("records")[: a.limit or None]
    exact = {e.mix_id for e in sdata.exact_mixes(DEFAULT_DATA_DIR)}
    res = {d: {"y": [], "mix": [], "sep": [], "true": [], "exact": []} for d in ("heart", "lung")}
    for r in rows:
        m = sdata.read(hls_cmds._resolve(DEFAULT_DATA_DIR, r["Mixed Sound ID"]))
        eh, el = sep(m, sdata.SR)
        for d, key, est, src in (("heart", "heart_types", eh, r["Heart Sound ID"]),
                                 ("lung", "lung_types", el, r["Lung Sound ID"])):
            head = reg.get(key)
            t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # noqa: E731
            res[d]["y"].append(r[f"{d.capitalize()} Sound Type"])
            res[d]["mix"].append(head.predict(t(m), sdata.SR)["pred"])
            res[d]["sep"].append(head.predict(t(est), sdata.SR)["pred"])
            res[d]["true"].append(head.predict(t(sdata.read(hls_cmds._resolve(DEFAULT_DATA_DIR, src))), sdata.SR)["pred"])
            res[d]["exact"].append(r["Mixed Sound ID"] in exact)
    summary = {}
    for d, v in res.items():
        y = np.array(v["y"])
        summary[d] = {k: float(balanced_accuracy_score(y, np.array(v[k]))) for k in ("mix", "sep", "true")}
        ex = np.array(v["exact"])
        summary[d]["exact_subset"] = {k: float(balanced_accuracy_score(y[ex], np.array(v[k])[ex]))
                                      for k in ("mix", "sep", "true")} if ex.any() else {}
        print(f"{d:5s} typing balanced acc  mixture {summary[d]['mix']:.3f}  separated {summary[d]['sep']:.3f}  "
              f"true source {summary[d]['true']:.3f}   (exact-mix subset: "
              + ", ".join(f"{k} {x:.3f}" for k, x in summary[d]["exact_subset"].items()) + ")")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "n": len(rows)}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
