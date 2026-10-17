"""Leakage and mechanics checks for Phase 2 (lsbfd.adapt). No data download needed."""

import numpy as np
import pytest
import torch

from lsbfd import adapt

PU_BEARINGS = sorted(
    {f"K00{i}" for i in range(1, 7)}
    | {f"KA{n:02d}" for n in (1, 3, 4, 5, 6, 7, 8, 9, 15, 16, 22, 30)}
    | {"KB23", "KB24", "KB27"}
    | {f"KI{n:02d}" for n in (1, 3, 4, 5, 7, 8, 14, 16, 17, 18, 21)}
)


def test_all_32_pu_bearings_listed():
    assert len(PU_BEARINGS) == 32


@pytest.mark.parametrize("stage", ["val", "test"])
def test_stage_bearings_are_disjoint(stage):
    train, scored = adapt.stage_bearings(stage, PU_BEARINGS)
    assert not set(train) & set(scored)
    assert {adapt.label_of(b) for b in scored} == {adapt.label_of(b) for b in train}
    if stage == "test":
        assert scored == sorted(adapt.TEST_BEARINGS) and len(train) == 28
    else:
        # The validation stage never touches the benchmark's test bearings.
        assert not set(adapt.TEST_BEARINGS) & (set(train) | set(scored))
        assert scored == sorted(adapt.VAL_BEARINGS) and len(train) == 24


def test_window_starts_match_pdm_bench():
    from pdm_bench.signals.recordings import Recording
    from pdm_bench.signals.dataset import Dataset

    n = 50_000
    sig = adapt.synthetic_signals(["K002", "KA01"], n_rec=2, n=n)
    for overlap in (adapt.TRAIN_OVERLAP, adapt.TEST_OVERLAP):
        starts, rec = sig.windows(overlap, {"KA01"})
        assert set(rec.tolist()) == {2, 3}  # only KA01's recordings
        r = Recording("x", sig.data[None, :n], 64000.0, "healthy", "PU", "g")
        ref = Dataset.from_recordings("t", [r]).window_dataset(adapt.WINDOW, overlap)[0].starts
        np.testing.assert_array_equal(starts[rec == 2] - sig.offsets[2], ref)
        assert (starts + adapt.WINDOW <= sig.offsets[rec] + sig.lengths[rec]).all()


def test_gradient_reversal_flips_and_scales():
    x = torch.ones(3, requires_grad=True)
    adapt.GradReverse.apply(x, 0.5).sum().backward()
    torch.testing.assert_close(x.grad, torch.full((3,), -0.5))


def test_windowset_gathers_the_right_samples():
    sig = adapt.synthetic_signals(["K002", "KA01", "KI01"], n_rec=1, n=20_000)
    data = torch.from_numpy(sig.data)
    classes = ["healthy", "inner_ring", "outer_ring"]
    ws = adapt.WindowSet(sig, data, ["KA01", "KI01"], 0.5, classes, ["KA01", "KI01"])
    i = torch.tensor([0, len(ws) - 1])
    x = ws.x(i)
    assert x.shape == (2, 1, adapt.WINDOW)
    s = ws.starts[-1].item()
    torch.testing.assert_close(x[1, 0], data[s:s + adapt.WINDOW])
    assert set(ws.bearing) == {"KA01", "KI01"} and ws.y.tolist()[0] == 2


def test_augment_keeps_shape_and_changes_signal():
    x = torch.randn(4, 1, adapt.WINDOW)
    v = adapt.augment(x)
    assert v.shape == x.shape and not torch.allclose(v, x)


def test_unreadable_files_are_skipped_like_pdm_bench(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pdm_bench.loaders import pu

    d = tmp_path / adapt.PU_DIR / "KA01"
    d.mkdir(parents=True)
    for m in (1, 2):
        (d / f"N15_M07_F10_KA01_{m}.mat").touch()

    def fake(f):
        if f.stem.endswith("_2"):
            raise TypeError("Expecting matrix here")
        rec = SimpleNamespace(data=np.zeros((1, 9000), np.float32), channels=[adapt.CHANNEL],
                              rid=f.stem, meta={"raster": "HostService", "bearing_code": "KA01"})
        return [rec]

    monkeypatch.setattr(pu, "_load_pu_file", fake)
    sig = adapt.load_signals(tmp_path, ["KA01"], cache=tmp_path / "cache")
    assert sig.rids == ["N15_M07_F10_KA01_1"] and sig.bearings == ["KA01"]
    # Second load comes from the per-bearing cache, with the raw files gone.
    for f in d.iterdir():
        f.unlink()
    again = adapt.load_signals(tmp_path, ["KA01"], cache=tmp_path / "cache")
    assert again.rids == sig.rids and (again.data == sig.data).all()


def test_manifest_lists_all_32_bearings():
    assert adapt.manifest_bearings() == PU_BEARINGS
