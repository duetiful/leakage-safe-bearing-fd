"""Phase 3 inputs: physics-informed envelope-order features for PU vibration windows.

Bearing faults make the race or ball strike the damage once per pass, which modulates a
high-frequency structural resonance at a *fault characteristic frequency*. Demodulating the
signal (Hilbert envelope of a band-passed signal) and taking its spectrum shows peaks at those
frequencies, and dividing the frequency axis by the shaft speed (the "order" axis) makes the
peaks land at the same place for every bearing and speed. For the PU test rig's 6203 bearing
(8 balls, ball diameter 6.75 mm, pitch diameter 28.55 mm, contact angle 0):

    BPFO = 3.05 x shaft speed   (outer-ring fault)
    BPFI = 4.95 x shaft speed   (inner-ring fault)
    BSF  = 2.00 x shaft speed   (ball fault, ball spin)

These numbers depend on the bearing's geometry, not on which individual bearing is mounted,
which is why features read at them should transfer to unseen bearings better than generic
statistics (RMS, kurtosis, ...). The shaft speed comes from the PU file name (N09 = 900 rpm,
N15 = 1500 rpm), which is operating metadata, not the label.

For every window (8192 samples at 64 kHz, start step 2048 = 75 % overlap, the same starts as
Phase 2; 50 %-overlap windows are every second one), this module computes:

  generic  the 11 generic pdm-bench features the LR baseline uses (time: rms, std, kurt, cf,
           sf, clf, ii; frequency: mf, fc, rms_f, std_f), so feature sets can be compared in
           one pipeline
  phys     envelope-order features: in each of 5 bands, the envelope-spectrum level at
           harmonics 1-3 of BSF, BPFO and BPFI, at shaft orders 1-2, and at the inner-ring
           sidebands BPFI x h +/- 1 (h = 1, 2), each in log units relative to the median of
           that band's envelope spectrum; plus each band's share of signal energy and the
           kurtosis of its envelope
  spec     the whole log envelope-order spectrum (5 bands x 395 orders, 0.2 to 15.96), median-
           normalized per band, as float16: the input of the envelope CNN

Everything is per window and uses no statistics from other windows, so it cannot leak across
the split. Results are cached per bearing in <cache>/env_<bearing>.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy.stats import kurtosis

FS = 64_000
WINDOW = 8192
STEP = 2048  # 75 % overlap; 50 %-overlap windows are the ones with start % 4096 == 0
PAD = 4  # envelope FFT zero-padding factor (interpolation only; resolution is 7.8 Hz)
BANDS = ((500, 4_000), (4_000, 10_000), (10_000, 20_000), (20_000, 31_000), (500, 31_000))
ORDERS = np.round(np.arange(0.2, 16.0, 0.04), 4)  # 395 points (3 x BPFI = 14.8)
FAULT_ORDERS = {"bsf": 1.997, "bpfo": 3.054, "bpfi": 4.946}  # 6203: n=8, d=6.75, D=28.55
RPM = {"N09": 900, "N15": 1500}

GENERIC_NAMES = ["rms", "std", "kurt", "cf", "sf", "clf", "ii", "mf", "fc", "rms_f", "std_f"]


def _phys_names() -> list[str]:
    per_band = []
    for f in FAULT_ORDERS:
        per_band += [f"{f}_h{h}" for h in (1, 2, 3)]
    per_band += ["shaft_1x", "shaft_2x", "bpfi_h1_sb-", "bpfi_h1_sb+", "bpfi_h2_sb-",
                 "bpfi_h2_sb+"]
    names = [f"b{i}_{n}" for i in range(len(BANDS)) for n in per_band]
    names += [f"b{i}_energy_share" for i in range(len(BANDS))]
    names += [f"b{i}_env_kurt" for i in range(len(BANDS))]
    return names


PHYS_NAMES = _phys_names()


def _targets() -> list[tuple[float, float]]:
    """(order, tolerance) of every per-band peak feature, in PHYS_NAMES order."""
    t = []
    for o in FAULT_ORDERS.values():
        t += [(h * o, 0.03 * h * o + 0.15) for h in (1, 2, 3)]
    t += [(1.0, 0.12), (2.0, 0.15)]
    bpfi = FAULT_ORDERS["bpfi"]
    t += [(bpfi - 1, 0.2), (bpfi + 1, 0.2), (2 * bpfi - 1, 0.3), (2 * bpfi + 1, 0.3)]
    return t


_TARGET_MASKS = np.stack([np.abs(ORDERS - o) <= tol for o, tol in _targets()])  # (T, 395)


def rpm_of(rid: str) -> int:
    """Shaft speed from a PU recording id such as 'N15_M07_F10/KA01/3::HostService'."""
    return RPM[rid[:3]]


def generic_features(x: np.ndarray) -> np.ndarray:
    """pdm-bench's 11 generic features for a batch of windows (B, N) -> (B, 11)."""
    rms = np.sqrt(np.mean(x**2, axis=1))
    std = x.std(axis=1)
    kurt = kurtosis(x, axis=1, fisher=False)
    absx = np.abs(x)
    peak = absx.max(axis=1)
    mean_abs = absx.mean(axis=1)
    cf = x.max(axis=1) / rms
    sf = rms / mean_abs
    clf = peak / np.mean(np.sqrt(absx), axis=1) ** 2
    ii = peak / mean_abs
    mag = np.abs(sfft.rfft(x, axis=1))
    f = sfft.rfftfreq(x.shape[1], 1 / FS)
    s = mag.sum(axis=1)
    mf = (mag * f).sum(axis=1) / s
    fc = (mag * f**2).sum(axis=1) / (mag * f).sum(axis=1)
    rms_f = np.sqrt((mag * f**2).sum(axis=1) / s)
    std_f = np.sqrt((mag * (f[None, :] - fc[:, None]) ** 2).sum(axis=1) / s)
    return np.stack([rms, std, kurt, cf, sf, clf, ii, mf, fc, rms_f, std_f], axis=1)


def envelope_features(x: np.ndarray, rpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Envelope-order features and spectra for a batch of windows of one shaft speed.

    Returns (phys (B, len(PHYS_NAMES)), spec (B, len(BANDS), len(ORDERS)))."""
    b, n = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    spec = sfft.rfft(x, axis=1, workers=-1)
    f = sfft.rfftfreq(n, 1 / FS)
    fr = rpm / 60.0
    nfft = PAD * n
    fe = sfft.rfftfreq(nfft, 1 / FS)
    keep = fe <= (ORDERS[-1] + 0.5) * fr
    fe = fe[keep]
    hann = np.hanning(n)[None, :]
    total = np.sum(np.abs(spec) ** 2, axis=1) + 1e-30
    peaks, shares, kurts, out = [], [], [], []
    for lo, hi in BANDS:
        band = (f >= lo) & (f < hi)
        h = np.zeros((b, n), dtype=np.complex128)
        h[:, : len(f)][:, band] = 2.0 * spec[:, band]  # one-sided -> analytic signal
        env = np.abs(sfft.ifft(h, axis=1, workers=-1))
        shares.append(np.sum(np.abs(spec[:, band]) ** 2, axis=1) / total)
        kurts.append(kurtosis(env, axis=1, fisher=False))
        env = env - env.mean(axis=1, keepdims=True)
        es = np.abs(sfft.rfft(env * hann, n=nfft, axis=1, workers=-1))[:, keep]
        pos = np.interp(ORDERS * fr, fe, np.arange(len(fe)))  # fractional bin of each order
        i0 = np.floor(pos).astype(int)
        w = pos - i0
        on_orders = es[:, i0] * (1 - w) + es[:, np.minimum(i0 + 1, len(fe) - 1)] * w
        logs = np.log(on_orders + 1e-12)
        logs -= np.median(logs, axis=1, keepdims=True)
        out.append(logs)
        peaks.append(np.stack([logs[:, m].max(axis=1) for m in _TARGET_MASKS], axis=1))
    phys = np.concatenate(peaks + [np.log(np.stack(shares, 1) + 1e-12), np.log(np.stack(kurts, 1))],
                          axis=1)
    return phys.astype(np.float32), np.stack(out, axis=1).astype(np.float16)


def despike(x: np.ndarray, k: float) -> np.ndarray:
    """Clip each window at its median +/- k robust standard deviations (MAD / 0.6745).

    PU recordings carry large impulsive spikes that are not bearing-fault impacts (a spike
    train at 100 Hz at 1500 rpm and 52.5 Hz at 900 rpm, in most healthy and artificial-damage
    bearings; see results/explore.md). Clipping removes their amplitude while keeping the
    low-level modulation that real fault signatures produce."""
    med = np.median(x, axis=1, keepdims=True)
    sig = np.median(np.abs(x - med), axis=1, keepdims=True) / 0.6745
    return np.clip(x, med - k * sig, med + k * sig)


def cache_envelope(vib_cache: Path, bearing: str, out: Path, batch: int = 512,
                   clip: float | None = None, keep_spec: bool = True) -> Path:
    """Compute and store every window's features for one bearing (skips if present).
    `clip` despikes each window first (see `despike`)."""
    key = out / f"env_{bearing}.npz"
    if key.exists():
        return key
    z = np.load(vib_cache / f"pu_vibration_1_{bearing}.npz", allow_pickle=False)
    data, lengths, rids = z["data"], z["lengths"].astype(np.int64), list(z["rids"])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rec, start, gen, phys, spec = [], [], [], [], []
    for i, (off, n) in enumerate(zip(offsets, lengths)):
        starts = np.arange(0, n - WINDOW + 1, STEP, dtype=np.int64)
        if not len(starts):
            continue
        rpm = rpm_of(str(rids[i]))
        for j in range(0, len(starts), batch):
            s = starts[j:j + batch]
            xw = data[off + s[:, None] + np.arange(WINDOW)[None, :]].astype(np.float64)
            if clip:
                xw = despike(xw, clip)
            p, sp = envelope_features(xw, rpm)
            gen.append(generic_features(xw).astype(np.float32))
            phys.append(p)
            spec.append(sp if keep_spec else sp[:, :, :1])
        rec.append(np.full(len(starts), i, dtype=np.int32))
        start.append(starts)
    out.mkdir(parents=True, exist_ok=True)
    tmp = key.with_suffix(".tmp.npz")
    np.savez(tmp, rec=np.concatenate(rec), start=np.concatenate(start), rids=np.array(rids),
             generic=np.concatenate(gen), phys=np.concatenate(phys), spec=np.concatenate(spec))
    tmp.rename(key)
    return key
