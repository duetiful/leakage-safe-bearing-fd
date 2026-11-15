"""Phase 3: methods borrowed from neighbouring problems, on PU Cross-Bearing Instance.

    python -m lsbfd.novel features            # envelope-order cache for all 32 bearings (CPU)
    python -m lsbfd.novel ml                  # feature models, official + rotated folds (CPU)
    python -m lsbfd.novel cnn --seed 41       # envelope-spectrum CNN + test-time BN (GPU)
    python -m lsbfd.novel report              # tables -> <out>/phase3_tables.md

Methods (src/lsbfd/envelope.py explains the inputs):
  feature models   logistic regression (C chosen by leave-bearings-out CV on the training
                   bearings only) and gradient-boosted trees (fixed settings), on three
                   feature sets: `generic` (the 11 features of the 0.508 LR baseline),
                   `phys` (envelope-order features at the 6203 fault frequencies) and
                   `generic+phys`. Same windows (8192, 50 % overlap), same split, same
                   classifier code, so the feature set is the only change.
  env-CNN          a small 1D CNN on the 5-band log envelope-order spectrum, with band dropout
                   and order-jitter augmentation (DG: no target data in training). Scored three
                   ways: as trained; with AdaBN on the pooled unlabeled scored windows; and with
                   AdaBN per scored bearing. Both AdaBN variants use unlabeled test windows at
                   test time (test-time adaptation, transductive), so they are reported apart.

Folds. Fold 0 is the benchmark's split (test K001, KA30, KB23, KI21). Because one bearing per
class decides that score, folds 1..n-1 rotate the held-out set: each holds out one random
real-damage bearing per class (and one healthy bearing), drawn with a fixed seed, and trains on
the other 28. Whole bearings are held out in every fold; the overlap is asserted.

No hyperparameter is chosen on scored bearings: LR's C uses grouped CV inside the training
bearings; every other setting is fixed in this file before any PU result was seen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lsbfd.adapt import TEST_BEARINGS, git_sha, label_of, manifest_bearings
from lsbfd.envelope import GENERIC_NAMES, PHYS_NAMES, cache_envelope

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["healthy", "inner_ring", "multiple", "outer_ring"]  # sorted, as pdm-bench
REAL = {  # real (lifetime) damage bearings and healthy bearings, per class
    "healthy": ["K001", "K002", "K003", "K004", "K005", "K006"],
    "inner_ring": ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"],
    "multiple": ["KB23", "KB24", "KB27"],
    "outer_ring": ["KA04", "KA15", "KA16", "KA22", "KA30"],
}
FEATURE_SETS = ("generic", "phys", "generic+phys")
TEST_STEP = 4096  # 50 % overlap, as the benchmark scores (and trains its LR)


# ----------------------------------------------------------------------------- data


def folds(n: int, seed: int = 2026) -> list[list[str]]:
    """Fold 0 = benchmark split; then n-1 distinct random one-bearing-per-class hold-outs."""
    out = [sorted(TEST_BEARINGS)]
    rng = np.random.default_rng(seed)
    while len(out) < n:
        f = sorted(str(rng.choice(REAL[c])) for c in CLASSES)
        if f not in out:
            out.append(f)
    return out


class Bank:
    """Every cached window of every bearing: features, spectra, labels, bearing, recording."""

    def __init__(self, env_dir: Path, bearings: list[str], spec: bool = False):
        parts = {k: [] for k in ("generic", "phys", "spec", "start", "bearing", "rid")}
        for b in bearings:
            z = np.load(env_dir / f"env_{b}.npz", allow_pickle=False)
            n = len(z["start"])
            parts["generic"].append(z["generic"])
            parts["phys"].append(z["phys"])
            if spec:
                parts["spec"].append(z["spec"])
            parts["start"].append(z["start"])
            parts["bearing"].append(np.full(n, b))
            parts["rid"].append(z["rids"][z["rec"]])
        cat = {k: np.concatenate(v) for k, v in parts.items() if v}
        self.generic, self.phys = cat["generic"], cat["phys"]
        self.spec = cat.get("spec")
        self.start, self.bearing, self.rid = cat["start"], cat["bearing"], cat["rid"]
        self.y = np.array([CLASSES.index(label_of(b)) for b in self.bearing])

    def X(self, fs: str) -> np.ndarray:
        x = {"generic": self.generic, "phys": self.phys,
             "generic+phys": np.hstack([self.generic, self.phys])}[fs]
        return np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def split(self, held_out: list[str], train_step: int) -> tuple[np.ndarray, np.ndarray]:
        test = np.isin(self.bearing, held_out)
        tr = np.flatnonzero(~test & (self.start % train_step == 0))
        te = np.flatnonzero(test & (self.start % TEST_STEP == 0))
        assert not set(self.rid[tr]) & set(self.rid[te]), "a recording is on both sides"
        assert not set(self.bearing[tr]) & set(self.bearing[te]), "a bearing is on both sides"
        return tr, te


def metrics(true: np.ndarray, pred: np.ndarray, bearing: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score)

    labels = list(range(len(CLASSES)))
    return {
        "macro_f1": float(f1_score(true, pred, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "per_class_f1": dict(zip(CLASSES, map(float, f1_score(
            true, pred, average=None, labels=labels, zero_division=0)))),
        "per_bearing_accuracy": {str(b): float(np.mean(pred[bearing == b] == true[bearing == b]))
                                 for b in sorted(set(bearing))},
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
    }


def save_preds(path: Path, pred: np.ndarray, bank: "Bank", te: np.ndarray, **more) -> None:
    """Window predictions next to the run's json, for the recording-level analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), pred=pred.astype(np.int8), true=bank.y[te].astype(np.int8),
             rid=bank.rid[te], bearing=bank.bearing[te],
             **{k: v.astype(np.int8) for k, v in more.items()})


def vote(pred: np.ndarray, rid: np.ndarray) -> dict[str, int]:
    """Majority class of the window predictions of each recording (ties: lowest class id)."""
    out = {}
    for r in np.unique(rid):
        out[str(r)] = int(np.bincount(pred[rid == r], minlength=len(CLASSES)).argmax())
    return out


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ----------------------------------------------------------------------------- features


def cmd_features(a) -> None:
    from concurrent.futures import ProcessPoolExecutor

    bs = manifest_bearings()
    todo = [b for b in bs if not (Path(a.env) / f"env_{b}.npz").exists()]
    print(f"envelope cache: {len(bs) - len(todo)} of {len(bs)} bearings present", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {b: ex.submit(cache_envelope, Path(a.cache), b, Path(a.env)) for b in todo}
        for b, f in futs.items():
            f.result()
            print(f"{b}: done", flush=True)
    n = len(list(Path(a.env).glob("env_*.npz")))
    print(f"envelope cache in {a.env}: {n} of {len(bs)} bearings")


# ----------------------------------------------------------------------------- ML


def fit_predict(model: str, Xtr, ytr, gtr, Xte, seed: int) -> tuple[np.ndarray, dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "lr":
        def make(c):
            return make_pipeline(StandardScaler(), LogisticRegression(
                C=c, class_weight="balanced", max_iter=3000))
        # C by leave-bearings-out CV inside the training bearings (every 4th window for speed)
        sub = np.arange(0, len(ytr), 4)
        scores = {}
        for c in (0.01, 0.1, 1.0, 10.0):
            fs = []
            for tr, va in GroupKFold(n_splits=4).split(Xtr[sub], ytr[sub], gtr[sub]):
                m = make(c).fit(Xtr[sub][tr], ytr[sub][tr])
                fs.append(f1_score(ytr[sub][va], m.predict(Xtr[sub][va]), average="macro"))
            scores[c] = float(np.mean(fs))
        c = max(scores, key=lambda k: (scores[k], -k))  # ties go to stronger regularization
        return make(c).fit(Xtr, ytr).predict(Xte), {"C": c, "cv_macro_f1": scores}
    if model == "hgb":
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                           l2_regularization=1.0, class_weight="balanced",
                                           early_stopping=False, random_state=seed)
        return m.fit(Xtr, ytr).predict(Xte), {}
    raise ValueError(model)


def cmd_ml(a) -> None:
    bank = Bank(Path(a.env), manifest_bearings())
    for k, held in enumerate(folds(a.folds)):
        tr, te = bank.split(held, train_step=TEST_STEP)  # LR baseline trains at 50 % overlap
        for fs in FEATURE_SETS:
            for model in ("lr", "hgb"):
                path = Path(a.out) / "ml" / f"fold{k}" / f"{model}_{fs}.json"
                if path.exists() and path.with_suffix(".npz").exists() and not a.force:
                    continue
                t0 = time.time()
                X = bank.X(fs)
                pred, info = fit_predict(model, X[tr], bank.y[tr], bank.bearing[tr], X[te],
                                         a.seed)
                m = metrics(bank.y[te], pred, bank.bearing[te])
                save_preds(path, pred, bank, te)
                save(path, {"fold": k, "held_out": held, "feature_set": fs, "model": model,
                            "seed": a.seed, "metrics": m, "selection": info,
                            "n_train": int(len(tr)), "n_test": int(len(te)),
                            "n_features": int(X.shape[1]), "git_sha": git_sha(),
                            "seconds": round(time.time() - t0, 1)})
                print(f"fold{k} {held} {model:3s} {fs:13s} macro-F1 {m['macro_f1']:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- env-CNN


def env_cnn(n_bands: int, n_classes: int):
    import torch.nn as nn

    # Keeps coarse order position (AdaptiveAvgPool to 16 cells): where a peak sits on the
    # order axis is what separates BPFO from BPFI, so a global average would discard it.
    return nn.Sequential(
        nn.Conv1d(n_bands, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.AdaptiveAvgPool1d(16), nn.Flatten(),
        nn.Linear(64 * 16, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, n_classes))


def _augment(x, gen):
    import torch

    b, c, n = x.shape
    shift = int(torch.randint(-2, 3, (1,), generator=gen))  # order jitter (speed / slip)
    x = torch.roll(x, shift, dims=2)
    keep = (torch.rand(b, c, 1, generator=gen) > 0.2).to(x.device)  # band dropout
    return x * keep


def _predict(model, X, idx, bs=1024):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            out.append(model(X[idx[i:i + bs]].float()).argmax(1).cpu())
    return torch.cat(out).numpy()


def _adabn(model, X, idx, bs=1024):
    """Copy of the model whose BatchNorm statistics are re-estimated on X[idx] (no labels)."""
    import copy

    import torch
    import torch.nn as nn

    m = copy.deepcopy(model)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over all batches
    m.train()
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            m(X[idx[i:i + bs]].float())
    return m


def cmd_cnn(a) -> None:
    import torch
    import torch.nn.functional as F

    fold_ids = range(a.folds) if a.all_folds else [a.fold]
    bank = None
    for k in fold_ids:
        held = folds(max(a.folds, k + 1))[k]
        path = Path(a.out) / "cnn" / f"fold{k}" / f"envcnn_seed{a.seed}.json"
        if path.exists() and not a.force:
            print(f"skip: {path} exists")
            continue
        if bank is None:
            bank = Bank(Path(a.env), manifest_bearings(), spec=True)
            dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            X = torch.from_numpy(bank.spec).to(dev)  # float16, (N, bands, orders)
            Y = torch.from_numpy(bank.y).to(dev)
        tr, te = bank.split(held, train_step=2048)  # 75 % overlap, as the Phase 2 CNN
        torch.manual_seed(a.seed)
        gen = torch.Generator().manual_seed(a.seed)
        model = env_cnn(X.shape[1], len(CLASSES)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = a.epochs * int(np.ceil(len(tr) / a.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
        counts = np.bincount(bank.y[tr], minlength=len(CLASSES))
        w = torch.tensor(len(tr) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32,
                         device=dev)
        tr_t = torch.from_numpy(tr).to(dev)
        t0, hist = time.time(), []
        for ep in range(1, a.epochs + 1):
            model.train()
            perm = tr_t[torch.randperm(len(tr), generator=gen).to(dev)]
            tot = 0.0
            for i in range(0, len(perm), a.batch):
                idx = perm[i:i + a.batch]
                if len(idx) < 2:
                    continue
                logits = model(_augment(X[idx].float(), gen))
                loss = F.cross_entropy(logits, Y[idx], weight=w, label_smoothing=0.05)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(idx)
            hist.append({"epoch": ep, "loss": tot / len(tr)})
            print(f"fold{k} seed{a.seed} epoch {ep} loss {tot / len(tr):.4f}", flush=True)
        te_t = torch.from_numpy(te).to(dev)
        yt, bt = bank.y[te], bank.bearing[te]
        res = {"none": metrics(yt, _predict(model, X, te_t), bt),
               "adabn_pooled": metrics(yt, _predict(_adabn(model, X, te_t), X, te_t), bt)}
        pred = np.empty(len(te), dtype=int)
        for b in held:
            sel = np.flatnonzero(bt == b)
            ib = te_t[torch.from_numpy(sel).to(dev)]
            pred[sel] = _predict(_adabn(model, X, ib), X, ib)
        res["adabn_per_bearing"] = metrics(yt, pred, bt)
        save_preds(path, _predict(model, X, te_t), bank, te, adabn_per_bearing=pred,
                   adabn_pooled=_predict(_adabn(model, X, te_t), X, te_t))
        save(path, {"fold": k, "held_out": held, "seed": a.seed, "epochs": a.epochs,
                    "metrics": res, "history": hist, "n_train": int(len(tr)),
                    "n_test": int(len(te)), "git_sha": git_sha(), "device": str(dev),
                    "seconds": round(time.time() - t0, 1)})
        print(f"fold{k} seed{a.seed} env-CNN macro-F1: "
              + ", ".join(f"{v} {r['macro_f1']:.3f}" for v, r in res.items()), flush=True)


# ----------------------------------------------------------------------------- report


def cmd_report(a) -> None:
    from lsbfd.novel_report import report

    text = report(Path(a.out))
    (Path(a.out) / "phase3_tables.md").write_text(text)
    print(text)


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["features", "ml", "cnn", "report"])
    p.add_argument("--cache", default=str(ROOT / "data" / "processed"),
                   help="Phase 2 per-bearing vibration cache (pu_vibration_1_<B>.npz)")
    p.add_argument("--env", default="", help="envelope cache (default <cache>/env)")
    p.add_argument("--out", default=str(ROOT / "runs" / "phase3"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    a.env = a.env or str(Path(a.cache) / "env")
    return a


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    {"features": cmd_features, "ml": cmd_ml, "cnn": cmd_cnn, "report": cmd_report}[args.cmd](args)
