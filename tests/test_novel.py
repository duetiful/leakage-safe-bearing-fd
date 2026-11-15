"""Leakage and mechanics checks for Phase 3 (lsbfd.envelope, lsbfd.novel). No data needed."""

import numpy as np
import pytest

from lsbfd import envelope, novel
from lsbfd.adapt import TEST_BEARINGS, label_of, manifest_bearings


def test_folds_hold_out_one_real_bearing_per_class():
    fs = novel.folds(10)
    assert fs[0] == sorted(TEST_BEARINGS)
    assert len({tuple(f) for f in fs}) == 10  # all distinct
    all_b = set(manifest_bearings())
    for f in fs:
        assert sorted(label_of(b) for b in f) == novel.CLASSES
        assert set(f) <= all_b
        for b in f:
            assert b in novel.REAL[label_of(b)]  # never an artificial-damage bearing
    assert novel.folds(10) == fs  # fixed seed: same folds every time


def test_rpm_from_recording_id():
    assert envelope.rpm_of("N15_M07_F10/KA01/3::HostService") == 1500
    assert envelope.rpm_of("N09_M07_F10/KA01/3::HostService") == 900


def _impulses(order, rpm, n=8192 * 3, seed=0):
    """Fault-like signal: resonance (7 kHz) struck at `order` x shaft speed, plus noise."""
    rng = np.random.default_rng(seed)
    fs, fr = envelope.FS, rpm / 60
    t = np.arange(n) / fs
    imp = np.zeros(n)
    imp[(np.arange(0, n / fs, 1 / (order * fr)) * fs).astype(int)] = 1.0
    ring = np.exp(-t[:400] * 3000) * np.sin(2 * np.pi * 7000 * t[:400])
    x = rng.normal(0, 0.3, n) + 3 * np.convolve(imp, ring)[:n]
    return np.stack([x[i:i + 8192] for i in range(0, n - 8192 + 1, 4096)])


@pytest.mark.parametrize("rpm", [900, 1500])
def test_envelope_features_find_the_fault_frequency(rpm):
    names = envelope.PHYS_NAMES
    for fault, other in (("bpfo", "bpfi"), ("bpfi", "bpfo")):
        p, spec = envelope.envelope_features(_impulses(envelope.FAULT_ORDERS[fault], rpm), rpm)
        assert p.shape[1] == len(names) and spec.shape[1:] == (len(envelope.BANDS),
                                                               len(envelope.ORDERS))
        d = dict(zip(names, p.mean(0)))
        assert d[f"b1_{fault}_h1"] > d[f"b1_{other}_h1"] + 0.5  # 4-10 kHz band holds 7 kHz


def test_generic_features_shape_and_rms():
    x = np.random.default_rng(0).normal(0, 2.0, (3, 8192))
    g = envelope.generic_features(x)
    assert g.shape == (3, len(envelope.GENERIC_NAMES))
    assert np.allclose(g[:, 0], 2.0, atol=0.05)


def test_bank_split_never_shares_a_bearing(tmp_path):
    rng = np.random.default_rng(0)
    for b in manifest_bearings():
        n = 20
        np.savez(tmp_path / f"env_{b}.npz", rec=np.repeat([0, 1], 10).astype(np.int32),
                 start=np.tile(np.arange(10) * 2048, 2),
                 rids=np.array([f"N15_M07_F10/{b}/1::HostService",
                                f"N09_M07_F10/{b}/2::HostService"]),
                 generic=rng.normal(size=(n, 11)).astype(np.float32),
                 phys=rng.normal(size=(n, len(envelope.PHYS_NAMES))).astype(np.float32),
                 spec=rng.normal(size=(n, 5, len(envelope.ORDERS))).astype(np.float16))
    bank = novel.Bank(tmp_path, manifest_bearings(), spec=True)
    for held in novel.folds(5):
        tr, te = bank.split(held, train_step=2048)
        assert set(bank.bearing[te]) == set(held)
        assert not set(bank.bearing[tr]) & set(held)
        assert np.all(bank.start[te] % novel.TEST_STEP == 0)
