"""Waveform DSP + log-mel features + train-time augmentation.

All waveform helpers take/return 1-D float32 ``torch.Tensor``s.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from cardia.config import Config


def _as_tensor(wav) -> torch.Tensor:
    if isinstance(wav, torch.Tensor):
        return wav.detach().to(torch.float32)
    return torch.as_tensor(np.asarray(wav), dtype=torch.float32)


def load_wav(path: str) -> Tuple[torch.Tensor, int]:
    """Read a WAV as mono float32 in [-1, 1]; returns (wav, sample_rate)."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.mean(axis=1).copy()), int(sr)


def resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    import torchaudio

    return torchaudio.functional.resample(wav, sr, target_sr)


def peak_normalize(wav, eps: float = 1e-9) -> torch.Tensor:
    w = _as_tensor(wav)
    peak = w.abs().max() if w.numel() else torch.tensor(0.0)
    return w / (peak + eps)


def band_pass(wav, sr: int, low: float, high: float, order: int = 4) -> torch.Tensor:
    """Zero-phase Butterworth band-pass (SOS, forward-backward)."""
    from scipy.signal import butter, sosfiltfilt

    w = _as_tensor(wav)
    nyq = sr / 2.0
    high = min(float(high), 0.99 * nyq)
    low = max(float(low), 1e-3)
    if w.numel() <= 3 * (2 * order + 1) or low >= high:
        return w
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return torch.from_numpy(sosfiltfilt(sos, w.numpy().astype(np.float64)).astype(np.float32).copy())


def _fixed_length(wav, n: int) -> torch.Tensor:
    """Center-crop or symmetrically zero-pad to exactly ``n`` samples."""
    w = _as_tensor(wav)
    length = w.shape[0]
    if length == n:
        return w
    if length > n:
        start = (length - n) // 2
        return w[start:start + n]
    pad = n - length
    return torch.nn.functional.pad(w, (pad // 2, pad - pad // 2))


def preprocess_waveform(wav, sr: int, cfg: Config) -> torch.Tensor:
    w = resample(_as_tensor(wav), sr, cfg.sample_rate)
    w = peak_normalize(w)
    w = band_pass(w, cfg.sample_rate, cfg.band_low, cfg.band_high)
    return _fixed_length(w, cfg.max_samples)


def preprocess(path: str, cfg: Config) -> torch.Tensor:
    """load -> resample -> peak-normalize -> task band-pass -> fixed length."""
    wav, sr = load_wav(path)
    return preprocess_waveform(wav, sr, cfg)


class MelSpec:
    """Per-clip standardized log-mel spectrogram: (T,) -> (n_mels, frames)."""

    def __init__(self, sample_rate: int, n_mels: int = 64, n_fft: int = 400, hop_length: int = 160,
                 eps: float = 1e-6) -> None:
        import torchaudio

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            f_min=0.0, f_max=sample_rate / 2.0, power=2.0, center=True,
        )

    def __call__(self, wav) -> torch.Tensor:
        w = _as_tensor(wav)
        if w.shape[-1] < self.n_fft:
            w = torch.nn.functional.pad(w, (0, self.n_fft - w.shape[-1]))
        with torch.no_grad():
            spec = torch.log(self._mel(w) + self.eps)
        return (spec - spec.mean()) / (spec.std() + self.eps)


class Augment:
    """Train-only augmentation: time shift, gain, noise (waveform) + SpecAugment.

    Deterministic given (seed, epoch, call order); ``set_epoch`` reseeds.
    """

    def __init__(self, cfg: Config, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.rng = np.random.default_rng([self.seed, self.epoch])

    def waveform(self, wav, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        rng = rng or self.rng
        w = _as_tensor(wav).clone()
        n = w.shape[0]
        max_shift = int(self.cfg.aug_time_shift * n)
        if max_shift > 0:
            w = torch.roll(w, int(rng.integers(-max_shift, max_shift + 1)))
        if self.cfg.aug_scale > 0:
            w = w * float(max(0.1, 1.0 + rng.normal(0.0, self.cfg.aug_scale)))
        if self.cfg.aug_noise > 0:
            w = w + torch.from_numpy(rng.normal(0.0, self.cfg.aug_noise, n).astype(np.float32))
        return w

    def spec(self, spec, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        rng = rng or self.rng
        s = _as_tensor(spec).clone()
        if not self.cfg.aug_specaug:
            return s
        n_freq, n_time = s.shape[-2], s.shape[-1]
        fill = float(s.mean())
        for _ in range(self.cfg.aug_time_masks):
            width = int(rng.integers(0, self.cfg.aug_time_mask_width + 1))
            if 0 < width < n_time:
                t0 = int(rng.integers(0, n_time - width))
                s[..., t0:t0 + width] = fill
        for _ in range(self.cfg.aug_freq_masks):
            width = int(rng.integers(0, self.cfg.aug_freq_mask_width + 1))
            if 0 < width < n_freq:
                f0 = int(rng.integers(0, n_freq - width))
                s[..., f0:f0 + width, :] = fill
        return s
