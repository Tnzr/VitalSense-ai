"""Central configuration: CLI args, hyperparameters, device auto-select, paths."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_REPO, ".env"))
except Exception:  # pragma: no cover - dotenv is optional
    pass

DEFAULT_DATA_DIR = os.environ.get("ASCULTO_DATA_DIR", os.path.join(_REPO, "data", "HLS-CMDS"))
# Multiclass (10/6-class Transformer) campaign output; defaults next to the binary runs.
DEFAULT_MC_OUT = os.environ.get("ASCULTO_MC_OUT", os.path.join(_REPO, "ml", "runs"))

TASKS = ("sound", "heart", "lung")
STAGES = (1, 2, 3, 4)
# binary = normal/abnormal; 6class = lung types; 10class = heart types
CLASS_SCHEMES = ("binary", "6class", "10class")
SCHEMES_BY_TASK = {
    "sound": ("binary",),
    "heart": ("binary", "10class"),
    "lung": ("binary", "6class"),
}

# Canonical label order matching the dataset's manifests (target encoding).
HEART_TYPES = [
    "Normal",
    "Late Diastolic Murmur",
    "Mid Systolic Murmur",
    "Late Systolic Murmur",
    "Atrial Fibrillation",
    "S3",
    "S4",
    "Early Systolic Murmur",
    "Tachycardia",
    "AV Block",
]
LUNG_TYPES = ["Normal", "Wheezing", "Fine Crackles", "Coarse Crackles", "Rhonchi", "Pleural Rub"]

# Band-pass bands (Hz) per task; Nyquist is 2000 Hz at 4 kHz.
BAND_BY_TASK = {"heart": (20.0, 600.0), "lung": (60.0, 1500.0), "sound": (20.0, 1500.0)}


@dataclass
class Config:
    task: str = "sound"
    stage: int = 1
    model: str = "resnet18"
    fold: int = 0
    n_folds: int = 5
    all_folds: bool = False
    seed: int = 0
    epochs: int = 60
    patience: int = 10
    batch_size: Optional[int] = None  # None -> auto (16 GPU / 8 CPU)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    class_scheme: str = "binary"
    data_dir: str = DEFAULT_DATA_DIR
    out_dir: str = "runs"
    device: str = "auto"  # auto|cuda|cpu
    num_workers: int = 2
    wandb_mode: str = "auto"  # auto|online|offline|disabled
    dry_run: bool = False
    smoke: bool = False
    cache_features: bool = True
    # DSP / feature config (logged to WandB)
    sample_rate: int = 4000
    max_samples: int = 60000  # 15 s at 4 kHz: center-pad/crop all clips to fixed length
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    mfcc_n: int = 13
    band_low: Optional[float] = None
    band_high: Optional[float] = None
    # Augmentation (train only)
    aug_time_shift: float = 0.15  # max fraction of clip length
    aug_scale: float = 0.15  # amplitude scale std
    aug_noise: float = 0.01  # gaussian noise std
    aug_specaug: bool = True
    aug_time_masks: int = 2
    aug_time_mask_width: int = 20
    aug_freq_masks: int = 2
    aug_freq_mask_width: int = 8
    # Temporal / transformer model sizes
    temporal_hidden: int = 128
    temporal_layers: int = 2
    transformer_d_model: int = 192
    transformer_nhead: int = 6
    transformer_layers: int = 4
    transformer_patch: tuple = (16, 16)
    grad_cam: bool = True

    def __post_init__(self) -> None:
        lo, hi = BAND_BY_TASK[self.task]
        self.band_low = self.band_low if self.band_low is not None else lo
        self.band_high = self.band_high if self.band_high is not None else hi
        if self.batch_size is None:
            self.batch_size = 16 if self.resolve_device() == "cuda" else 8
        if self.smoke:
            self.epochs = min(self.epochs, 2)
            self.n_folds = 2

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cardia-cli",
        description="CardiaSense HLS-CMDS benchmark training campaign (Stage 1 AI PoC)",
    )
    p.add_argument("--task", choices=TASKS, default="sound", help="classification task")
    p.add_argument("--stage", type=int, choices=STAGES, default=1, help="training stage 1..4")
    p.add_argument(
        "--model",
        default="resnet18",
        help="stage2: mobilenetv3|resnet18; stage3: gru|lstm; stage4: transformer",
    )
    p.add_argument("--fold", type=int, default=0, help="fold index 0..n_folds-1 (run one fold)")
    p.add_argument("--all-folds", action="store_true", help="run all folds for the stage/task")
    p.add_argument("--n-folds", type=int, default=5, help="number of CV folds")
    p.add_argument("--seed", type=int, default=0, help="global seed 0..2")
    p.add_argument("--epochs", type=int, default=60, help="max epochs")
    p.add_argument("--patience", type=int, default=10, help="early-stop patience")
    p.add_argument("--batch-size", type=int, default=None, help="override auto batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--class-scheme", choices=CLASS_SCHEMES, default="binary", help="lung task scheme")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="HLS-CMDS dataset root")
    p.add_argument("--out-dir", default="runs", help="output root (checkpoints, splits, metrics)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--wandb-mode",
        choices=("auto", "online", "offline", "disabled"),
        default="auto",
        help="auto: online if WANDB_API_KEY set, else offline",
    )
    p.add_argument("--dry-run", action="store_true", help="data-only run: stats + input figures")
    p.add_argument("--smoke", action="store_true", help="2 folds, 2 epochs smoke test")
    p.add_argument("--no-cache-features", action="store_true", help="disable in-memory feature cache")
    return p


def config_from_args(argv: Optional[list] = None) -> Config:
    p = build_parser()
    ns = p.parse_args(argv)
    kw = {
        "task": ns.task,
        "stage": ns.stage,
        "model": ns.model,
        "fold": ns.fold,
        "n_folds": ns.n_folds,
        "all_folds": ns.all_folds,
        "seed": ns.seed,
        "epochs": ns.epochs,
        "patience": ns.patience,
        "batch_size": ns.batch_size,
        "lr": ns.lr,
        "class_scheme": ns.class_scheme,
        "data_dir": ns.data_dir,
        "out_dir": ns.out_dir,
        "device": ns.device,
        "num_workers": ns.num_workers,
        "wandb_mode": ns.wandb_mode,
        "dry_run": ns.dry_run,
        "smoke": ns.smoke,
        "cache_features": not ns.no_cache_features,
    }
    return Config(**kw)
