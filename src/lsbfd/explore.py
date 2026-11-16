"""Phase 4: looking for what limits cross-bearing transfer on PU, and testing fixes.

    python -m lsbfd.explore features --clip 4 --tag clip4   # despiked feature cache
    python -m lsbfd.explore ml --env <dir> --train real --labels 3 --tag <name>
    python -m lsbfd.explore table                            # results/explore_tables.md

Every variant is scored on the same 10 folds as Phase 3 (benchmark split + 9 rotated
one-bearing-per-class hold-outs), with whole bearings held out. Options:

  --train all    train on every non-held-out bearing (benchmark setting, 28 bearings)
  --train real   train only on healthy + real-damage bearings (drops the 12 artificial ones)
  --labels 4     benchmark labels: healthy / inner / multiple / outer
  --labels 3     the dataset authors' labels (Lessmeier et al., 2016, Sec. 7.4): KB23 and KB24
                 are inner-ring dominant, KB27 outer-ring dominant; healthy / inner / outer.
                 This changes the task, so it is reported separately, never as a benchmark score.

These were chosen from the data audit in results/explore.md, which looked at per-bearing
statistics of all 32 bearings (including held-out ones) but at no classifier score other than
the Phase 3 per-bearing accuracies already published. That is stated as a limitation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lsbfd.adapt import label_of, manifest_bearings
from lsbfd.envelope import cache_envelope
from lsbfd.novel import REAL, TEST_STEP, Bank, fit_predict, folds, git_sha, save, save_preds

ROOT = Path(__file__).resolve().parents[2]
REAL_SET = {b for bs in REAL.values() for b in bs}
DOMINANT = {"KB23": "inner_ring", "KB24": "inner_ring", "KB27": "outer_ring"}  # Lessmeier 7.4
CLASSES3 = ["healthy", "inner_ring", "outer_ring"]


def labels3(bearing: np.ndarray) -> np.ndarray:
    return np.array([CLASSES3.index(DOMINANT.get(b, label_of(b))) for b in bearing])


def metrics3(true, pred, bearing) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    lab = list(range(len(CLASSES3)))
    return {"macro_f1": float(f1_score(true, pred, average="macro", labels=lab, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "per_bearing_accuracy": {str(b): float(np.mean(pred[bearing == b] == true[bearing == b]))
                                     for b in sorted(set(bearing))},
            "confusion_matrix": confusion_matrix(true, pred, labels=lab).tolist()}


CONDS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10"]


def features(bank: Bank, fs: str) -> np.ndarray:
    """Feature set by name; a '+cond' suffix appends the operating condition (one-hot, from
    the file name: speed, load torque, radial force), which is metadata, not the label."""
    base, cond = (fs[:-5], True) if fs.endswith("+cond") else (fs, False)
    X = bank.X(base)
    if cond:
        c = np.array([str(r).split("/")[0] for r in bank.rid])
        X = np.hstack([X, np.stack([c == k for k in CONDS], 1).astype(np.float64)])
    return X


def fit_predict_x(model, Xtr, ytr, gtr, Xte, seed):
    """Phase 3 models plus `hgb-bag`: 15 trees ensembles, each trained on a random 70 % of
    the training bearings (bagging over bearings, so no single bearing's quirks dominate),
    probabilities averaged."""
    if model != "hgb-bag":
        return fit_predict(model, Xtr, ytr, gtr, Xte, seed)
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    bearings = np.unique(gtr)
    classes = np.unique(ytr)
    prob = np.zeros((len(Xte), len(classes)))
    for i in range(15):
        while True:  # every class must stay in the bag
            pick = rng.choice(bearings, size=int(round(0.7 * len(bearings))), replace=False)
            m = np.isin(gtr, pick)
            if set(np.unique(ytr[m])) == set(classes):
                break
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                             l2_regularization=1.0, class_weight="balanced",
                                             early_stopping=False, random_state=seed + i)
        clf.fit(Xtr[m], ytr[m])
        prob += clf.predict_proba(Xte)
    return classes[prob.argmax(1)], {"n_bags": 15, "bag_fraction": 0.7}


def cmd_features(a) -> None:
    out = Path(a.cache) / f"env_{a.tag}"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(cache_envelope, Path(a.cache), b, out, 512, a.clip, False)
                for b in manifest_bearings()]
        for f in futs:
            f.result()
    print(f"{out}: {len(list(out.glob('env_*.npz')))} bearings")


def cmd_ml(a) -> None:
    from lsbfd.novel import metrics

    bank = Bank(Path(a.env), manifest_bearings())
    y = labels3(bank.bearing) if a.labels == 3 else bank.y
    for k, held in enumerate(folds(a.folds)):
        tr, te = bank.split(held, train_step=TEST_STEP)
        if a.train == "real":
            tr = tr[np.isin(bank.bearing[tr], list(REAL_SET))]
        for fs in a.feature_sets.split(","):
            for model in a.models.split(","):
                path = Path(a.out) / a.tag / f"fold{k}" / f"{model}_{fs}.json"
                if path.exists() and not a.force:
                    continue
                t0 = time.time()
                X = features(bank, fs)
                pred, info = fit_predict_x(model, X[tr], y[tr], bank.bearing[tr], X[te], a.seed)
                m = (metrics3 if a.labels == 3 else metrics)(y[te], pred, bank.bearing[te])
                save_preds(path, pred, bank, te)
                save(path, {"fold": k, "held_out": held, "feature_set": fs, "model": model,
                            "tag": a.tag, "env": str(a.env), "train": a.train,
                            "labels": a.labels, "metrics": m, "selection": info,
                            "n_train": int(len(tr)), "git_sha": git_sha(),
                            "seconds": round(time.time() - t0, 1)})
                print(f"[{a.tag}] fold{k} {model} {fs} macro-F1 {m['macro_f1']:.3f}", flush=True)


def cmd_rec(a) -> None:
    """Recording-level diagnosis: average each 4 s recording's window envelope spectra
    (log, median-normalized) into one spectrum, read the same fault-order peaks from it, and
    classify recordings. Averaging over ~120 windows raises the fault-peak SNR that a single
    0.128 s window lacks. Unit of scoring: the recording (as in the recording-level vote)."""
    from lsbfd.envelope import _TARGET_MASKS
    from lsbfd.novel import CLASSES

    env = Path(a.env)
    feats, gens, ys, bs, rids = [], [], [], [], []
    for b in manifest_bearings():
        z = np.load(env / f"env_{b}.npz", allow_pickle=False)
        for r in np.unique(z["rec"]):
            m = z["rec"] == r
            spec = z["spec"][m].astype(np.float32).mean(0)  # (bands, orders)
            spec -= np.median(spec, axis=1, keepdims=True)
            peaks = np.stack([spec[:, t].max(axis=1) for t in _TARGET_MASKS], 1).ravel()
            feats.append(peaks)
            gens.append(np.median(z["generic"][m], axis=0))
            ys.append(CLASSES.index(label_of(b)))
            bs.append(b)
            rids.append(str(z["rids"][r]))
    P, G = np.nan_to_num(np.array(feats, dtype=np.float64)), np.nan_to_num(np.array(gens, dtype=np.float64))
    y, bs = np.array(ys), np.array(bs)
    if a.labels == 3:
        y = labels3(bs)
    sets = {"generic": G, "phys": P, "generic+phys": np.hstack([G, P])}
    for k, held in enumerate(folds(a.folds)):
        te = np.isin(bs, held)
        tr = ~te & (np.isin(bs, list(REAL_SET)) if a.train == "real" else True)
        tr, te = np.flatnonzero(tr), np.flatnonzero(te)
        for fs in a.feature_sets.split(","):
            for model in a.models.split(","):
                path = Path(a.out) / a.tag / f"fold{k}" / f"{model}_{fs}.json"
                if path.exists() and not a.force:
                    continue
                X = sets[fs]
                pred, info = fit_predict_x(model, X[tr], y[tr], bs[tr], X[te], a.seed)
                from lsbfd.novel import metrics
                m = (metrics3 if a.labels == 3 else metrics)(y[te], pred, bs[te])
                save(path, {"fold": k, "held_out": held, "feature_set": fs, "model": model,
                            "tag": a.tag, "unit": "recording", "train": a.train,
                            "labels": a.labels, "metrics": m, "selection": info,
                            "n_train": int(len(tr)), "git_sha": git_sha()})
                print(f"[{a.tag}] fold{k} {model} {fs} recording macro-F1 {m['macro_f1']:.3f}",
                      flush=True)


def cmd_table(a) -> None:
    import pandas as pd
    from scipy.stats import wilcoxon

    rows = []
    for p in Path(a.out).glob("*/fold*/*.json"):
        r = json.loads(p.read_text())
        rows.append({"variant": r["tag"], "method": f"{r['model']} / {r['feature_set']}",
                     "labels": r["labels"], "fold": r["fold"], "f1": r["metrics"]["macro_f1"]})
    for p in (ROOT / "runs/phase3/ml").glob("fold*/*.json"):  # Phase 3 = baseline variant
        r = json.loads(p.read_text())
        rows.append({"variant": "phase3", "method": f"{r['model']} / {r['feature_set']}",
                     "labels": 4, "fold": r["fold"], "f1": r["metrics"]["macro_f1"]})
    df = pd.DataFrame(rows)
    pv = df.pivot_table(index=["labels", "variant", "method"], columns="fold", values="f1")
    lines = ["| Labels | Variant | Method | Folds | Macro-F1 mean ± std | Fold 0 | Δ vs Phase 3 same method | Better folds | p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (lab, var, meth), row in pv.iterrows():
        v = row.dropna()
        ref = pv.loc[(4, "phase3", meth)] if (4, "phase3", meth) in pv.index else None
        cells = [str(lab), var, meth, str(len(v)), f"{v.mean():.3f} ± {v.std(ddof=1):.3f}",
                 f"{row.get(0, float('nan')):.3f}"]
        if ref is not None and var != "phase3" and lab == 4:
            d = (v - ref[v.index]).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
            cells += [f"{d.mean():+.3f}", f"{int((d > 0).sum())} of {len(d)}", f"{p:.3f}"]
        else:
            cells += ["", "", ""]
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "explore_tables.md").write_text(text)
    print(text)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["features", "ml", "rec", "table"])
    p.add_argument("--cache", default=str(ROOT / "data" / "processed"))
    p.add_argument("--env", default=str(ROOT / "data" / "processed" / "env"))
    p.add_argument("--out", default=str(ROOT / "runs" / "explore"))
    p.add_argument("--tag", default="base")
    p.add_argument("--clip", type=float, default=None)
    p.add_argument("--train", choices=["all", "real"], default="all")
    p.add_argument("--labels", type=int, choices=[3, 4], default=4)
    p.add_argument("--models", default="hgb")
    p.add_argument("--feature-sets", default="generic,phys,generic+phys")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    {"features": cmd_features, "ml": cmd_ml, "rec": cmd_rec, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
