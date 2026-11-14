import numpy as np
import torch

from cardia.separation import classical, metrics
from cardia.separation.model import Separator, si_sdr_loss

SR = 4000


def _toy(seconds=6.0):
    t = np.arange(int(seconds * SR)) / SR
    heart = np.zeros_like(t)
    for b in np.arange(0.2, seconds, 0.8):  # 75 bpm, 60 Hz thumps
        m = (t >= b) & (t < b + 0.08)
        heart[m] += np.sin(2 * np.pi * 60 * (t[m] - b)) * np.hanning(m.sum())
    lung = 0.3 * np.random.default_rng(0).standard_normal(t.size) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))
    return heart.astype(np.float32), lung.astype(np.float32)


def test_si_sdr_orders_estimates():
    h, l = _toy()
    assert metrics.si_sdr(h, h) > 60
    assert metrics.si_sdr(h + l, h) > metrics.si_sdr(l, h)


def test_classical_is_mixture_consistent():
    h, l = _toy()
    for method in ("band", "repet", "floor_repet_sim"):
        eh, el = classical.separate(h + l, SR, method)
        assert eh.shape == el.shape == h.shape
        np.testing.assert_allclose(eh + el, h + l, atol=1e-4)


def test_beat_period_on_toy_heart():
    h, _ = _toy(12.0)
    assert abs(classical.beat_period(h, SR) - 0.8) < 0.05


def test_mask_model_reconstructs_mixture_and_trains():
    torch.manual_seed(0)
    m = Separator()
    mix = torch.randn(2, 8000)
    h, l = m(mix)
    assert h.shape == mix.shape
    assert torch.allclose(h + l, mix, atol=1e-4)
    loss = si_sdr_loss(h, mix)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_synthetic_mixes_and_leakage_free_split(data_dir):
    from cardia.separation import data
    sp = data.split_stems(data_dir)
    cl = data.recording_clusters(data_dir)
    test = {cl[i] for e in data.exact_mixes(data_dir) for i in (e.heart_id, e.lung_id)}
    tr, va = {s.group for s in sp["train"]}, {s.group for s in sp["val"]}
    assert not (tr & va) and not (tr & test) and not (va & test)
    assert len(data.exact_mixes(data_dir)) >= 30
    ds = data.SyntheticMixes(sp["val"], 3, seconds=2.0, fixed=True)
    mix, h, l = ds[1]
    assert mix.shape == (8000,)
    assert torch.allclose(mix, h + l)
    assert torch.equal(ds[1][0], mix)  # fixed items are deterministic


def test_webapp_separation_without_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCULTO_SEP_CKPT", str(tmp_path / "missing.pt"))
    from webapp.app import separation
    out = separation.separate_report(torch.zeros(SR), SR, None, [])
    assert "error" in out
