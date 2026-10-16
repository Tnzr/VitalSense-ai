"""Leakage-safe grouped, stratified K-fold splits.

Fold k uses CV fold k as test and fold (k+1) % K as validation; the rest is
training. With K < 3 the validation set is carved out of the non-test groups.
Recordings sharing a ``group`` (e.g. a mixed clip and its heart/lung sources)
always land in the same partition; this is asserted on every split.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FoldSplit:
    fold: int
    train: List[str]
    val: List[str]
    test: List[str]


def _assert_disjoint(manifest: pd.DataFrame, sp: FoldSplit) -> None:
    g = manifest.set_index("sample_id")["group"].astype(str)
    parts = {p: set(g.loc[getattr(sp, p)]) for p in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = parts[a] & parts[b]
        assert not shared, f"fold {sp.fold}: groups leak between {a} and {b}: {sorted(shared)[:5]}"
    ids = sp.train + sp.val + sp.test
    assert len(ids) == len(set(ids)) == len(manifest), f"fold {sp.fold}: split does not partition the manifest"


def _group_folds(manifest: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    fold_of = np.full(len(manifest), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(manifest, manifest["target"], manifest["group"].astype(str))):
        fold_of[test_idx] = k
    return fold_of


def _carve_val(manifest: pd.DataFrame, idx: np.ndarray, seed: int, frac: float = 0.25) -> tuple:
    groups = np.array(sorted(manifest.iloc[idx]["group"].astype(str).unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_groups = set(groups[: max(1, int(round(frac * len(groups))))])
    in_val = manifest.iloc[idx]["group"].astype(str).isin(val_groups).to_numpy()
    return idx[~in_val], idx[in_val]


def make_splits(manifest: pd.DataFrame, n_folds: int = 5, seed: int = 0, out_dir: Optional[str] = None,
                split_dir: Optional[str] = None) -> List[FoldSplit]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    m = manifest.reset_index(drop=True)
    fold_of = _group_folds(m, n_folds, seed)
    ids = m["sample_id"].astype(str).to_numpy()
    splits: List[FoldSplit] = []
    for k in range(n_folds):
        test_idx = np.where(fold_of == k)[0]
        if n_folds >= 3:
            v = (k + 1) % n_folds
            val_idx = np.where(fold_of == v)[0]
            train_idx = np.where((fold_of != k) & (fold_of != v))[0]
        else:
            train_idx, val_idx = _carve_val(m, np.where(fold_of != k)[0], seed + k)
        sp = FoldSplit(k, ids[train_idx].tolist(), ids[val_idx].tolist(), ids[test_idx].tolist())
        _assert_disjoint(m, sp)
        splits.append(sp)
    if split_dir is None and out_dir is not None:
        split_dir = os.path.join(out_dir, "splits", f"n{n_folds}_s{seed}")
    if split_dir:
        save_splits(splits, split_dir)
    return splits


def save_splits(splits: List[FoldSplit], split_dir: str) -> None:
    os.makedirs(split_dir, exist_ok=True)
    for sp in splits:
        with open(os.path.join(split_dir, f"fold{sp.fold}.json"), "w") as f:
            json.dump(asdict(sp), f, indent=1)


def load_splits(split_dir: str, n_folds: int = 5) -> List[FoldSplit]:
    splits = []
    for k in range(n_folds):
        path = os.path.join(split_dir, f"fold{k}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing split file {path}")
        with open(path) as f:
            d = json.load(f)
        splits.append(FoldSplit(int(d["fold"]), list(d["train"]), list(d["val"]), list(d["test"])))
    return splits
