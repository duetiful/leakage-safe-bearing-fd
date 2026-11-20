"""Feature-space analysis: how compact and bearing-invariant is each representation?

    python -m lsbfd.featurespace features              # generic + envelope features, raw and despiked
    python -m lsbfd.featurespace embed --method erm    # CPU retrain of the CNN, save embeddings
    python -m lsbfd.featurespace embed --method dann   # same with the within-class DANN head
    python -m lsbfd.featurespace analyze               # metrics + 2-D maps -> <out>/

Every representation is read on the same sample of windows: 4 windows per recording (starts
0, 2, 4 and 6 x 30,720 samples), all 32 bearings, so about 10,200 points.

Metrics (computed in the full-dimensional space, standardized with statistics of the 28
training bearings of the benchmark split, never of the held-out four):

  silhouette, class      how well the four fault classes form separate, compact clusters
                         (-1..1, higher is better)
  silhouette, bearing    inside each class, how well individual bearings form their own
                         clusters, averaged over classes (lower is better: a representation
                         that separates bearings of one class carries bearing identity)
  (the row "real-damage training" uses healthy + real-damage bearings only, as Phase 4 does)
  kNN transfer F1        macro-F1 of a 15-nearest-neighbour vote from training-bearing windows
                         to the windows of the held-out bearings (does an unseen bearing land
                         among training windows of its own class?). Feature spaces: mean over
                         the 10 folds of Phase 3. CNN/DANN: fold 0 only (one trained model).

The 2-D maps are UMAP (or t-SNE if umap-learn is missing) of the same points, fitted without
labels; they are for looking, and every number above is computed before projection.

CNN / DANN embeddings. The Phase 2 checkpoints were trained on Kaggle GPUs and not kept, so
`embed` retrains both on CPU with the Phase 2 code (`lsbfd.adapt`: same CNN1D, same recipe,
DANN with the within-class bearing adversary and lambda = 1, chosen in Phase 2), but with a
reduced budget: non-overlapping training windows and fewer epochs (see --epochs). The 128-d
penultimate layer (global average pool output) is the embedding. The reduced runs' own
macro-F1 is saved next to them, so they can be compared with the reported 3-seed numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lsbfd.adapt import TEST_BEARINGS, WINDOW, label_of, manifest_bearings

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["healthy", "inner_ring", "multiple", "outer_ring"]
SAMPLE_STARTS = np.array([0, 2, 4, 6]) * 30_720  # 4 windows per 4 s recording
FOLD0 = sorted(TEST_BEARINGS)


# ----------------------------------------------------------------------------- sample


def sample_index(env_dir: Path) -> dict:
    """(bearing, rid, start) of the sampled windows, from the envelope cache."""
    out = {"bearing": [], "rid": [], "start": [], "row": []}
    for b in manifest_bearings():
        z = np.load(env_dir / f"env_{b}.npz", allow_pickle=False)
        rows = np.flatnonzero(np.isin(z["start"], SAMPLE_STARTS))
        out["row"].append(rows)
        out["bearing"].append(np.full(len(rows), b))
        out["rid"].append(z["rids"][z["rec"][rows]])
        out["start"].append(z["start"][rows])
    return {k: np.concatenate(v) for k, v in out.items()}


def load_feature_space(env_dir: Path, name: str) -> tuple[np.ndarray, dict]:
    idx = sample_index(env_dir)
    xs, i0 = [], 0
    for b in manifest_bearings():
        z = np.load(env_dir / f"env_{b}.npz", allow_pickle=False)
        n = int((idx["bearing"] == b).sum())
        xs.append(z[name][idx["row"][i0:i0 + n]])
        i0 += n
    X = np.nan_to_num(np.concatenate(xs).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return X, idx


# ----------------------------------------------------------------------------- features


def cmd_features(a) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from lsbfd.envelope import cache_envelope

    for tag, clip in (("raw", None), ("clip4", 4.0)):
        out = Path(a.cache) / f"fs_env_{tag}"
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {b: ex.submit(cache_envelope, Path(a.cache), b, out, 512, clip, False)
                    for b in manifest_bearings()}
            for b, f in futs.items():
                f.result()
        print(f"{out}: {len(list(out.glob('env_*.npz')))} bearings", flush=True)


# ----------------------------------------------------------------------------- embed


def cmd_embed(a) -> None:
    import torch

    from lsbfd import adapt

    torch.set_num_threads(a.threads)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    out = Path(a.out) / f"cnn_{a.method}"
    out.mkdir(parents=True, exist_ok=True)

    all_b = manifest_bearings()
    train_b, scored_b = adapt.stage_bearings("test", all_b)
    sig = adapt.load_signals(Path(a.cache), train_b + scored_b, Path(a.cache))
    data = torch.from_numpy(sig.data).to(dev)
    classes = sorted({label_of(b) for b in train_b})
    assert classes == CLASSES
    domains = train_b if a.method == "dann" else []
    src = adapt.WindowSet(sig, data, train_b, a.train_overlap, classes, domains)
    if a.train_fraction < 1:  # fixed random subset of training windows (same for both methods)
        keep = np.sort(np.random.default_rng(0).choice(
            len(src), int(len(src) * a.train_fraction), replace=False))
        k = torch.from_numpy(keep).to(dev)
        src.starts, src.y, src.d, src.bearing = src.starts[k], src.y[k], src.d[k], src.bearing[keep]
    tgt = adapt.WindowSet(sig, data, scored_b, adapt.TEST_OVERLAP, classes, [])
    assert not src.rids & tgt.rids
    ns = argparse.Namespace(method=a.method, domain="bearing_in_class", lam=a.lam,
                            epochs=a.epochs, batch=64, lr=1e-3, wd=1e-4)
    ns.same_class = (torch.tensor([[label_of(b) == c for b in domains] for c in classes],
                                  dtype=torch.bool, device=dev) if domains else None)
    model = adapt.Net(len(classes), len(domains) if a.method == "dann" else 0).to(dev)
    hist = []

    def log(row):
        row["t"] = round(time.time() - t0)
        hist.append(row)
        print(json.dumps(row), flush=True)

    print(f"[{a.method}] {len(src)} training windows from {len(train_b)} bearings, "
          f"{a.epochs} epochs, device {dev}", flush=True)
    t0 = time.time()
    adapt.train(model, src, None, ns, gen, amp, log)
    torch.save(model.state_dict(), out / "model.pt")

    from sklearn.metrics import f1_score
    pred = adapt.predict(model, tgt, amp)
    true = tgt.y.cpu().numpy()
    f1 = float(f1_score(true, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0))
    per_b = {b: float(np.mean(pred[tgt.bearing == b] == true[tgt.bearing == b])) for b in scored_b}
    print(f"[{a.method}] fold-0 macro-F1 {f1:.3f} {per_b}", flush=True)

    # embeddings of the analysis sample (same windows as the feature spaces)
    idx = sample_index(Path(a.env))
    rid_pos = {(sig.bearings[i], str(r)): i for i, r in enumerate(sig.rids)}
    starts = np.array([sig.offsets[rid_pos[(b, str(r))]] + s
                       for b, r, s in zip(idx["bearing"], idx["rid"], idx["start"])])
    model.eval()
    emb = []
    ar = torch.arange(WINDOW, device=dev)
    with torch.no_grad():
        for i in range(0, len(starts), 256):
            st = torch.from_numpy(starts[i:i + 256]).to(dev)
            emb.append(model.embed(data[st[:, None] + ar].unsqueeze(1)).float().cpu().numpy())
    np.savez(out / "embedding.npz", X=np.concatenate(emb), bearing=idx["bearing"],
             rid=idx["rid"], start=idx["start"])
    (out / "summary.json").write_text(json.dumps({
        "method": a.method, "lam": a.lam if a.method == "dann" else None, "seed": a.seed,
        "epochs": a.epochs, "train_overlap": a.train_overlap, "train_fraction": a.train_fraction,
        "n_train_windows": len(src), "fold0_macro_f1": f1, "per_bearing_accuracy": per_b,
        "device": str(dev), "amp": amp, "git_sha": adapt.git_sha(),
        "seconds": round(time.time() - t0), "history": hist}, indent=2))


# ----------------------------------------------------------------------------- analyze


def silhouettes(X: np.ndarray, y: np.ndarray, bearing: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import silhouette_score

    s_class = float(silhouette_score(X, y))
    per, w = [], []
    for c in np.unique(y):
        m = y == c
        if len(np.unique(bearing[m])) > 1:
            per.append(silhouette_score(X[m], bearing[m]))
            w.append(m.sum())
    return s_class, float(np.average(per, weights=w))


def knn_transfer(X, y, bearing, held, use=None) -> float:
    """use: optional mask of points allowed in training (e.g. real-damage bearings only)."""
    from sklearn.metrics import f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    te = np.isin(bearing, held)
    tr = ~te if use is None else (~te & use)
    sc = StandardScaler().fit(X[tr])
    knn = KNeighborsClassifier(15).fit(sc.transform(X[tr]), y[tr])
    return float(f1_score(y[te], knn.predict(sc.transform(X[te])), average="macro",
                          labels=[0, 1, 2, 3], zero_division=0))


def cmd_analyze(a) -> None:
    from sklearn.preprocessing import StandardScaler

    from lsbfd.novel import REAL, folds

    out = Path(a.out)
    spaces = {}
    raw, clip = Path(a.cache) / "fs_env_raw", Path(a.cache) / "fs_env_clip4"
    ref = None
    for key, d, name, title in [
        ("generic", raw, "generic", "Generic statistics (11)"),
        ("phys", raw, "phys", "Envelope-order features (85)"),
        ("phys_clip4", clip, "phys", "Envelope-order, despiked (85)"),
        ("generic_clip4", clip, "generic", "Generic, despiked (11)"),
    ]:
        X, idx = load_feature_space(d, name)
        if ref is None:
            ref = idx
        assert (idx["bearing"] == ref["bearing"]).all() and (idx["start"] == ref["start"]).all()
        spaces[key] = (title, X, None)
    spaces["phys_clip4_real"] = ("Envelope, despiked, real-damage training (85)",
                                 spaces["phys_clip4"][1], "real")
    for m, title in (("erm", "CNN embedding (128)"), ("dann", "DANN embedding (128)")):
        p = out / f"cnn_{m}" / "embedding.npz"
        if p.exists():
            z = np.load(p, allow_pickle=False)
            assert (z["bearing"] == ref["bearing"]).all() and (z["start"] == ref["start"]).all()
            spaces[m] = (title, z["X"].astype(np.float64), None)
        else:
            print(f"missing {p}: {title} left out", flush=True)
    bearing = ref["bearing"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features_sample.npz", bearing=bearing, rid=ref["rid"],
                        start=ref["start"],
                        **{k: v[1].astype(np.float32) for k, v in spaces.items()})
    y = np.array([CLASSES.index(label_of(b)) for b in bearing])
    train = ~np.isin(bearing, FOLD0)
    fold_list = folds(10)

    rows, proj = [], {}
    real = np.isin(bearing, [b for bs in REAL.values() for b in bs])
    for key, (title, X, subset) in spaces.items():
        use = real if subset == "real" else np.ones(len(y), bool)
        # healthy + real-damage bearings only: metrics on that subset, scaler on its training part
        Z = StandardScaler().fit(X[train & use]).transform(X[use])
        s_c, s_b = silhouettes(Z, y[use], bearing[use])
        u = None if subset is None else use
        k0 = knn_transfer(X, y, bearing, FOLD0, u)
        k10 = ([knn_transfer(X, y, bearing, f, u) for f in fold_list]
               if key not in ("erm", "dann") else None)
        rows.append({"space": key, "title": title, "dim": X.shape[1], "points": subset or "all",
                     "silhouette_class": s_c,
                     "silhouette_bearing_in_class": s_b, "knn_f1_fold0": k0,
                     "knn_f1_10fold_mean": None if k10 is None else float(np.mean(k10)),
                     "knn_f1_10fold_std": None if k10 is None else float(np.std(k10, ddof=1))})
        print(json.dumps(rows[-1]), flush=True)
        if subset is not None:
            continue  # same coordinates as its parent space; no separate map
        try:
            import umap
            emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z)
            method = "UMAP"
        except ImportError:
            from sklearn.manifold import TSNE
            emb = TSNE(perplexity=40, random_state=0, init="pca").fit_transform(Z)
            method = "t-SNE"
        proj[key] = emb
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"n_points": int(len(y)), "projection": method,
                                                 "rows": rows}, indent=2))
    np.savez(out / "projections.npz", bearing=bearing, y=y,
             **{k: v.astype(np.float32) for k, v in proj.items()})
    lines = ["| Representation | Dim | Silhouette, class (↑) | Silhouette, bearing within class (↓) "
             "| kNN transfer F1, fold 0 | kNN transfer F1, 10 folds |", "|---|---|---|---|---|---|"]
    for r in rows:
        k10 = ("—" if r["knn_f1_10fold_mean"] is None
               else f"{r['knn_f1_10fold_mean']:.3f} ± {r['knn_f1_10fold_std']:.3f}")
        lines.append(f"| {r['title']} | {r['dim']} | {r['silhouette_class']:.3f} | "
                     f"{r['silhouette_bearing_in_class']:.3f} | {r['knn_f1_fold0']:.3f} | {k10} |")
    (out / "featurespace_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["features", "embed", "analyze"])
    p.add_argument("--cache", default=str(ROOT / "data" / "processed"))
    p.add_argument("--env", default=str(ROOT / "data" / "processed" / "fs_env_raw"))
    p.add_argument("--out", default=str(ROOT / "runs" / "featurespace"))
    p.add_argument("--method", choices=["erm", "dann"], default="erm")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--train-overlap", type=float, default=0.0)
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    {"features": cmd_features, "embed": cmd_embed, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
