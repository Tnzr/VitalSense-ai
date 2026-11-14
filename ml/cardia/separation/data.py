"""Separation data: heart/lung stem pools, exact-sum test mixes, synthetic mixtures.

HLS-CMDS ships, for each of the 145 ``Mix.csv`` rows, the mixed recording ``M####``
plus its heart (``H####``) and lung (``L####``) sources. For 36 rows the mixture is an
exact scaled sum ``M = k * (H + L)`` (sample-aligned), which gives ground-truth stems for
scoring. The other rows were recorded separately and only carry labels.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

from ..data import hls_cmds

SR = 4000
EXACT_R2 = 0.999


@dataclass(frozen=True)
class Stem:
    kind: str  # "heart" | "lung"
    sample_id: str
    group: str
    path: str


@dataclass(frozen=True)
class ExactMix:
    mix_id: str
    group: str
    heart_id: str
    lung_id: str
    scale: float
    heart_type: str
    lung_type: str


def read(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x.copy()), sr, SR).numpy()
    return x


def stems(data_dir: str) -> List[Stem]:
    out = []
    for kind, rows in (("heart", hls_cmds._heart_rows(data_dir)), ("lung", hls_cmds._lung_rows(data_dir))):
        for r in rows:
            path = hls_cmds._resolve(data_dir, r["sample_id"], r)
            out.append(Stem(kind, r["sample_id"], str(r["group"]), path))
    return out


def _exact_scale(m: np.ndarray, h: np.ndarray, l: np.ndarray) -> Optional[float]:
    n = min(len(m), len(h), len(l))
    m, s = m[:n].astype(np.float64), (h[:n] + l[:n]).astype(np.float64)
    den = float(s @ s)
    if den <= 0:
        return None
    k = float(m @ s) / den
    r2 = 1.0 - float(((m - k * s) ** 2).sum()) / max(float(((m - m.mean()) ** 2).sum()), 1e-12)
    return k if r2 >= EXACT_R2 else None


@lru_cache(maxsize=4)
def exact_mixes(data_dir: str) -> Tuple[ExactMix, ...]:
    """Mix rows whose recording is a sample-aligned scaled sum of its two sources."""
    cache = os.path.join(data_dir, ".exact_mixes.json")
    if os.path.isfile(cache):
        return tuple(ExactMix(**d) for d in json.load(open(cache)))
    out = []
    for r in hls_cmds._read_csv(data_dir, "Mix.csv").to_dict("records"):
        mid, hid, lid = r["Mixed Sound ID"], r["Heart Sound ID"], r["Lung Sound ID"]
        k = _exact_scale(*(read(hls_cmds._resolve(data_dir, s)) for s in (mid, hid, lid)))
        if k is not None:
            out.append(ExactMix(mid, hls_cmds._mix_group(mid), hid, lid, k,
                                r["Heart Sound Type"], r["Lung Sound Type"]))
    try:
        json.dump([e.__dict__ for e in out], open(cache, "w"), indent=1)
    except OSError:
        pass
    return tuple(out)


DUP_CORR = 0.9


def _unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


@lru_cache(maxsize=4)
def recording_clusters(data_dir: str) -> Dict[str, str]:
    """sample_id -> cluster id, merging stems that are the same recording.

    HLS-CMDS reuses recordings: many ``Mix`` heart/lung sources are copies (possibly
    shifted) of each other or of standalone ``HS``/``LS`` clips, so ``Mix.csv`` row numbers
    are not independent groups. Two stems are merged when their peak normalized
    cross-correlation at any lag is >= ``DUP_CORR``."""
    cache = os.path.join(data_dir, ".recording_clusters.json")
    if os.path.isfile(cache):
        return json.load(open(cache))
    out: Dict[str, str] = {}
    for kind in ("heart", "lung"):
        pool = [s for s in stems(data_dir) if s.kind == kind]
        ids = list(dict.fromkeys(s.sample_id for s in pool))
        path = {s.sample_id: s.path for s in pool}
        wav = [_unit(read(path[i])) for i in ids]
        nfft = 1 << int(2 * max(len(w) for w in wav) - 1).bit_length()
        F = [np.fft.rfft(w, nfft) for w in wav]
        parent = list(range(len(ids)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if find(i) != find(j) and np.abs(np.fft.irfft(F[i] * np.conj(F[j]), nfft)).max() >= DUP_CORR:
                    parent[find(j)] = find(i)
        for i, sid in enumerate(ids):
            out[sid] = f"{kind}:{ids[find(i)]}"
    try:
        json.dump(out, open(cache, "w"), indent=1)
    except OSError:
        pass
    return out


def split_stems(data_dir: str, val_frac: float = 0.15, seed: int = 0) -> Dict[str, List[Stem]]:
    """Train/val stem pools that share no recording with each other or with the test mixes.

    Each stem is keyed by its duplicate-recording cluster; every cluster used by an
    exact-sum test mix is excluded, and val takes whole clusters."""
    cl = recording_clusters(data_dir)
    test_clusters = {cl[i] for e in exact_mixes(data_dir) for i in (e.heart_id, e.lung_id)}
    uniq = {}
    for s in stems(data_dir):
        c = cl[s.sample_id]
        if c not in test_clusters:
            uniq.setdefault(c, Stem(s.kind, s.sample_id, c, s.path))
    pool = list(uniq.values())
    groups = sorted({s.group for s in pool})
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_groups = set(groups[: max(1, int(len(groups) * val_frac))])
    return {
        "train": [s for s in pool if s.group not in val_groups],
        "val": [s for s in pool if s.group in val_groups],
    }


def test_triplets(data_dir: str) -> List[Tuple[ExactMix, np.ndarray, np.ndarray, np.ndarray]]:
    """(meta, mixture, heart_ref, lung_ref) with refs scaled so mixture == heart + lung."""
    out = []
    for e in exact_mixes(data_dir):
        m, h, l = (read(hls_cmds._resolve(data_dir, s)) for s in (e.mix_id, e.heart_id, e.lung_id))
        n = min(len(m), len(h), len(l))
        out.append((e, m[:n], e.scale * h[:n], e.scale * l[:n]))
    return out


class SyntheticMixes(torch.utils.data.Dataset):
    """Random heart + lung crops summed at a random heart-to-lung ratio.

    ``fixed=True`` makes item ``i`` deterministic (validation set)."""

    def __init__(self, pool: Sequence[Stem], n_items: int, seconds: float = 4.0,
                 snr_db: Tuple[float, float] = (-15.0, 20.0), fixed: bool = False, seed: int = 0):
        self.heart = [s for s in pool if s.kind == "heart"]
        self.lung = [s for s in pool if s.kind == "lung"]
        self.n, self.seg, self.snr_db, self.fixed, self.seed = n_items, int(seconds * SR), snr_db, fixed, seed
        self._cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.n

    def _wav(self, s: Stem) -> np.ndarray:
        if s.path not in self._cache:
            x = read(s.path)
            self._cache[s.path] = x / (np.sqrt((x ** 2).mean()) + 1e-8)
        return self._cache[s.path]

    def _crop(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if len(x) < self.seg:
            x = np.pad(x, (0, self.seg - len(x)))
        o = int(rng.integers(0, len(x) - self.seg + 1))
        return x[o:o + self.seg]

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i) if self.fixed else None)
        h = self._crop(self._wav(self.heart[int(rng.integers(len(self.heart)))]), rng)
        l = self._crop(self._wav(self.lung[int(rng.integers(len(self.lung)))]), rng)
        if not self.fixed and rng.random() < 0.5:
            h, l = h * rng.choice([-1, 1]), l * rng.choice([-1, 1])
        snr = rng.uniform(*self.snr_db)
        rh, rl = np.sqrt((h ** 2).mean()) + 1e-8, np.sqrt((l ** 2).mean()) + 1e-8
        h = h / rh * 10 ** (snr / 20)
        l = l / rl
        gain = 10 ** rng.uniform(-1.5, -0.5) / max(np.abs(h + l).max(), 1e-8)
        h, l = (h * gain).astype(np.float32), (l * gain).astype(np.float32)
        return torch.from_numpy(h + l), torch.from_numpy(h), torch.from_numpy(l)
