import os

import numpy as np
import pandas as pd
import pytest
import torch

from cardia.config import HEART_TYPES, LUNG_TYPES, Config
from cardia.data import hls_cmds
from cardia.data.splits import load_splits, make_splits
from cardia.data.transforms import _fixed_length, band_pass, peak_normalize, preprocess_waveform

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED = os.path.join(REPO, "docs", "assets", "audio", "manifests")


def test_class_names():
    assert hls_cmds.class_names("sound") == ["heart", "lung", "mixed"]
    assert hls_cmds.class_names("heart") == ["normal", "abnormal"]
    assert hls_cmds.class_names("heart", "10class") == list(HEART_TYPES)
    assert hls_cmds.class_names("lung", "6class") == list(LUNG_TYPES)


@pytest.mark.parametrize("task,scheme,n", [
    ("sound", "binary", 535), ("heart", "binary", 195), ("heart", "10class", 195),
    ("lung", "binary", 195), ("lung", "6class", 195),
])
def test_manifest_counts_and_targets(data_dir, task, scheme, n):
    m = hls_cmds.build_task_manifest(data_dir, task, scheme)
    assert len(m) == n
    assert m["sample_id"].is_unique
    assert all(os.path.isfile(p) for p in m["file_path"])
    names = hls_cmds.class_names(task, scheme)
    assert (m["target"] == m["task_label"].map(names.index)).all()


@pytest.mark.parametrize("task,scheme", [("heart", "binary"), ("lung", "binary"), ("lung", "6class")])
def test_manifest_matches_committed_reference(data_dir, task, scheme):
    ref_path = os.path.join(COMMITTED, f"{task}_{scheme}_manifest.csv")
    if not os.path.isfile(ref_path):
        pytest.skip("no committed reference manifest")
    m = hls_cmds.build_task_manifest(data_dir, task, scheme).drop(columns="file_path").astype(str)
    ref = pd.read_csv(ref_path).drop(columns="file_path").astype(str)[m.columns]
    pd.testing.assert_frame_equal(m.reset_index(drop=True), ref)


def test_crackle_ids_resolve_to_renamed_files(data_dir):
    m = hls_cmds.build_task_manifest(data_dir, "lung", "6class")
    ids = set(m["sample_id"])
    assert {"M_FC_LUA", "M_CC_LUA"} <= ids
    assert "M_C_LUA" not in ids and "M_G_LUA" not in ids


@pytest.mark.parametrize("task,scheme", [("sound", "binary"), ("heart", "10class"), ("lung", "6class")])
def test_splits_are_group_disjoint_and_complete(data_dir, task, scheme, tmp_path):
    m = hls_cmds.build_task_manifest(data_dir, task, scheme)
    group = dict(zip(m["sample_id"], m["group"]))
    splits = make_splits(m, n_folds=5, seed=0, split_dir=str(tmp_path / "s"))
    tests = []
    for sp in splits:
        parts = [set(sp.train), set(sp.val), set(sp.test)]
        assert sum(len(p) for p in parts) == len(m)
        assert set().union(*parts) == set(m["sample_id"])
        gparts = [{group[s] for s in p} for p in parts]
        assert not (gparts[0] & gparts[1]) and not (gparts[0] & gparts[2]) and not (gparts[1] & gparts[2])
        tests.extend(sp.test)
    assert sorted(tests) == sorted(m["sample_id"])  # each sample tested exactly once
    reloaded = load_splits(str(tmp_path / "s"), n_folds=5)
    assert [r.test for r in reloaded] == [s.test for s in splits]


def test_waveform_transforms_shapes():
    cfg = Config(task="heart")
    sr = cfg.sample_rate
    t = torch.arange(sr * 3) / sr
    wav = 0.3 * torch.sin(2 * np.pi * 100 * t)
    assert peak_normalize(wav).abs().max() == pytest.approx(1.0, abs=1e-5)
    assert band_pass(wav, sr, cfg.band_low, cfg.band_high).shape == wav.shape
    n = cfg.max_samples
    assert _fixed_length(wav, n).shape[-1] == n
    out = preprocess_waveform(wav, sr, cfg)
    assert out.shape[-1] == n and torch.isfinite(out).all()
