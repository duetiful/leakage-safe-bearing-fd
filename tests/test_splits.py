"""Leakage checks on the benchmark splits, run against the real file lists.

The file names come from data/checksums/*.sha256 (the committed download manifest), so
these tests need no data download and no signals.
"""

from pathlib import Path

import pytest

from lsbfd import splits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    for name in ("cwru", "pu"):
        for line in (ROOT / "data" / "checksums" / f"{name}.sha256").read_text().splitlines():
            rel = Path(line.split("  ", 1)[1]).relative_to("data/raw")
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).touch()
    return root


@pytest.mark.parametrize("scenario", splits.SCENARIOS)
def test_no_recording_in_both_splits(fake_root, scenario):
    df = splits.resolve(scenario, fake_root)
    assert (df.split == "train").any() and (df.split == "test").any()
    assert splits.check(df)["recording"] == []


def test_pu_cross_bearing_instance_holds_out_whole_bearings(fake_root):
    df = splits.resolve("pu_cross_bearing_instance", fake_root)
    test = df[df.split == "test"]
    assert sorted(test.bearing_code.unique()) == ["K001", "KA30", "KB23", "KI21"]
    assert len(test) == 4 * 80
    assert len(df[df.split == "train"]) == 28 * 80
    assert splits.check(df)["physical bearing"] == []
    assert splits.check(df)["measurement"] == []


def test_cwru_cross_load_reuses_the_same_bearings(fake_root):
    # Documented property, not a bug in our code: CWRU recorded each seeded fault at
    # every load, so cross-load tests new loads on bearings already seen in training.
    df = splits.resolve("cwru_cross_load", fake_root)
    assert len(splits.check(df)["physical bearing"]) == len(
        df[df.split == "test"].physical_bearing.unique()
    )
