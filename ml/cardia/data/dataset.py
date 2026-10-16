"""Torch dataset over a task manifest: cached preprocessed waveforms -> log-mel."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from cardia.config import Config
from cardia.data.transforms import Augment, MelSpec, preprocess

# Shared across datasets/folds: preprocessing only depends on the file and DSP config.
_WAV_CACHE: Dict[Tuple, torch.Tensor] = {}


class HLSDataset(torch.utils.data.Dataset):
    """Items: ``{"x": (1, n_mels, frames), "y": long, "sample_id": str, "group": str}``."""

    def __init__(self, manifest: pd.DataFrame, cfg: Config, augment: bool = False, epoch: int = 0,
                 cache: bool = True) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.cache = cache
        self.mel = MelSpec(cfg.sample_rate, cfg.n_mels, cfg.n_fft, cfg.hop_length)
        self.aug = Augment(cfg, cfg.seed)
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.aug.set_epoch(self.epoch)

    def __len__(self) -> int:
        return len(self.manifest)

    def _waveform(self, path: str) -> torch.Tensor:
        c = self.cfg
        key = (path, c.sample_rate, c.band_low, c.band_high, c.max_samples)
        if self.cache and key in _WAV_CACHE:
            return _WAV_CACHE[key]
        wav = preprocess(path, c)
        if self.cache:
            _WAV_CACHE[key] = wav
        return wav

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        wav = self._waveform(row["file_path"])
        if self.augment:
            rng = np.random.default_rng([self.cfg.seed, self.epoch, idx])
            spec = self.aug.spec(self.mel(self.aug.waveform(wav, rng)), rng)
        else:
            spec = self.mel(wav)
        return {
            "x": spec.unsqueeze(0).float(),
            "y": torch.tensor(int(row["target"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "group": str(row["group"]),
        }
