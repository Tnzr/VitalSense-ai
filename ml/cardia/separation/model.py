"""Mask-based heart/lung separator (STFT -> conv + BiGRU -> two softmax masks -> iSTFT).

The two masks sum to one, so ``heart + lung`` reconstructs the mixture exactly."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data import SR

N_FFT, HOP = 256, 64


class MaskNet(nn.Module):
    def __init__(self, n_freq: int = N_FFT // 2 + 1, ch: int = 32, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.GELU(), nn.BatchNorm2d(ch),
            nn.Conv2d(ch, ch, 3, padding=(1, 2), dilation=(1, 2)), nn.GELU(), nn.BatchNorm2d(ch),
            nn.Conv2d(ch, 8, 3, padding=(1, 4), dilation=(1, 4)), nn.GELU(),
        )
        self.proj = nn.Linear(8 * n_freq, 2 * hidden)
        self.rnn = nn.GRU(2 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 2 * n_freq)
        self.n_freq = n_freq

    def forward(self, logmag: torch.Tensor) -> torch.Tensor:  # (B, F, T) -> (B, 2, F, T)
        b, f, t = logmag.shape
        h = self.conv(logmag.unsqueeze(1))                       # (B, 8, F, T)
        h = self.proj(h.permute(0, 3, 1, 2).reshape(b, t, -1))   # (B, T, 2H)
        h, _ = self.rnn(h)
        m = self.head(h).view(b, t, 2, f).permute(0, 2, 3, 1)   # (B, 2, F, T)
        return torch.softmax(m, dim=1)


class Separator(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.net = MaskNet(**kw)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def forward(self, mix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # (B, N)
        n = mix.shape[-1]
        Z = torch.stft(mix, N_FFT, HOP, window=self.window, return_complex=True)
        scale = mix.pow(2).mean(-1, keepdim=True).sqrt().unsqueeze(-1) + 1e-8
        masks = self.net(torch.log1p(Z.abs() / scale))
        out = [torch.istft(Z * masks[:, i], N_FFT, HOP, window=self.window, length=n) for i in (0, 1)]
        return out[0], out[1]


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est, ref = est - est.mean(-1, keepdim=True), ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return -(10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + eps) + eps)).mean()


def default_checkpoint() -> str:
    from ..config import _REPO
    return os.environ.get("ASCULTO_SEP_CKPT", os.path.join(_REPO, "ml", "runs", "sep", "separator.pt"))


@lru_cache(maxsize=2)
def _load(path: str) -> Separator:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Separator(**ck.get("model_kwargs", {}))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def load_separator(path: Optional[str] = None):
    """Callable ``(x, sr) -> (heart, lung)`` numpy arrays at 4 kHz."""
    model = _load(path or default_checkpoint())

    def separate(x: np.ndarray, sr: int):
        w = torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1)
        if sr != SR:
            import torchaudio
            w = torchaudio.functional.resample(w, sr, SR)
        with torch.no_grad():
            h, l = model(w)
        return h[0].numpy(), l[0].numpy()

    return separate
