"""Phase 2: domain adaptation on PU Cross-Bearing Instance, vibration channel only.

    python -m lsbfd.adapt --stage val  --method dann --domain bearing_in_class --lam 0.3 --seed 41
    python -m lsbfd.adapt --stage test --method simclr --seed 42

Stages (both leakage-safe; whole bearings are held out, never windows):
  val   train on 24 source bearings, score on 4 *source* bearings held out from them
        (K006, KA22, KB27, KI18: one per class, real damage like the test bearings).
        The test bearings are never loaded. Used to choose hyperparameters.
  test  train on all 28 source bearings, score on the benchmark's test bearings
        (K001, KA30, KB23, KI21).

Methods (all share pdm-bench's CNN1D and its Phase 1 recipe: AdamW 1e-3, weight decay 1e-4,
ReduceLROnPlateau on train loss, label smoothing 0.05, gradient clip 1.0, batch 64, no
validation-based early stopping):
  erm     plain cross-entropy; the control for this trainer
  dann    cross-entropy + gradient-reversal domain head (Ganin & Lempitsky, 2015). --domain:
            bearing           predict which source bearing a window came from (DG)
            bearing_in_class  same, but only among bearings of the window's own class, so
                              the adversary removes bearing identity, not fault class (DG)
            target            source vs. unlabeled target windows (UDA, transductive: the
                              target is the scored split's windows, without labels)
  simclr  SimCLR pretraining of the feature extractor on augmented source windows, then
          full fine-tuning with cross-entropy (DG; --pretrain-target adds unlabeled target
          windows, making it UDA)

DG = pure domain generalization (no target data of any kind in training).
Outputs go to <out>/<stage>/<tag>/seed<seed>/ in the same layout as pdm-bench runs, so
`python -m lsbfd.summarize <out>` collects them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
PU_DIR = "datasets/paderborn-university-bearing-dataset"
TEST_BEARINGS = ("K001", "KA30", "KB23", "KI21")  # pdm-bench pu_cross_bearing_instance
VAL_BEARINGS = ("K006", "KA22", "KB27", "KI18")  # our source-only validation hold-out
CHANNEL = "vibration_1"
WINDOW = 8192
TRAIN_OVERLAP, TEST_OVERLAP = 0.75, 0.5  # paper's "long" profile


def label_of(bearing: str) -> str:
    from pdm_bench.loaders.pu import _infer_label_from_bearing_code

    return _infer_label_from_bearing_code(bearing)


# ----------------------------------------------------------------------------- data


def stage_bearings(stage: str, all_bearings: list[str]) -> tuple[list[str], list[str]]:
    """(train bearings, scored bearings) for a stage. Never mixes the two."""
    source = sorted(b for b in all_bearings if b not in TEST_BEARINGS)
    if stage == "test":
        train, scored = source, sorted(TEST_BEARINGS)
    elif stage == "val":
        train = [b for b in source if b not in VAL_BEARINGS]
        scored = sorted(VAL_BEARINGS)
    else:
        raise ValueError(f"unknown stage {stage}")
    assert not set(train) & set(scored), "a bearing is on both sides of the split"
    assert stage == "test" or not set(TEST_BEARINGS) & (set(train) | set(scored))
    return train, scored


class Signals:
    """Vibration recordings of a set of bearings, concatenated into one 1D array."""

    def __init__(self, data: np.ndarray, offsets: np.ndarray, lengths: np.ndarray,
                 bearings: list[str], rids: list[str]):
        self.data, self.offsets, self.lengths = data, offsets, lengths
        self.bearings, self.rids = bearings, rids

    def windows(self, overlap: float, keep: set[str]) -> tuple[np.ndarray, np.ndarray]:
        """Global start index and recording of every window of the kept bearings, with the
        same starts as pdm-bench's window_dataset. Windows never cross recordings."""
        step = min(WINDOW, max(1, round(WINDOW * (1.0 - overlap))))
        starts, rec = [], []
        for i, (off, n) in enumerate(zip(self.offsets, self.lengths)):
            if self.bearings[i] not in keep:
                continue
            s = np.arange(0, n - WINDOW + 1, step, dtype=np.int64) if n >= WINDOW else []
            starts.append(off + s)
            rec.append(np.full(len(s), i, dtype=np.int64))
        return np.concatenate(starts), np.concatenate(rec)


def load_signals(root: Path, bearings: list[str], cache: Path | None) -> Signals:
    """Load the HostService vibration channel of every PU file of the given bearings."""
    from pdm_bench.loaders.pu import _load_pu_file

    files = sorted(f for f in (root / PU_DIR).rglob("*.mat") if f.parent.name in bearings)
    if not files:
        raise FileNotFoundError(f"No PU .mat files for {bearings} under {root / PU_DIR}")
    key = cache / f"pu_{CHANNEL}_{'-'.join(sorted(bearings))}.npz" if cache else None
    if key and len(key.name) > 120:
        digest = hashlib.sha1(",".join(sorted(bearings)).encode()).hexdigest()[:12]
        key = cache / f"pu_{CHANNEL}_{len(bearings)}bearings_{digest}.npz"
    if key and key.exists():
        z = np.load(key, allow_pickle=False)
        return Signals(z["data"], z["offsets"], z["lengths"], list(z["bearings"]), list(z["rids"]))
    parts, bear, rids = [], [], []
    for i, f in enumerate(files):
        recs = [r for r in _load_pu_file(f) if r.meta["raster"] == "HostService"]
        (r,) = recs
        parts.append(r.data[r.channels.index(CHANNEL)].astype(np.float32))
        bear.append(r.meta["bearing_code"])
        rids.append(r.rid)
        if (i + 1) % 200 == 0:
            print(f"loaded {i + 1}/{len(files)} files", flush=True)
    lengths = np.array([len(p) for p in parts], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    sig = Signals(np.concatenate(parts), offsets, lengths, bear, rids)
    if key:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.savez(key, data=sig.data, offsets=offsets, lengths=lengths,
                 bearings=np.array(bear), rids=np.array(rids))
    return sig


def synthetic_signals(bearings: list[str], n_rec: int = 4, n: int = 40_000,
                      seed: int = 0) -> Signals:
    """Small fake PU set for smoke tests: a class tone plus a bearing-specific tone."""
    rng = np.random.default_rng(seed)
    freq = {"healthy": 0.0, "outer_ring": 0.011, "inner_ring": 0.023, "multiple": 0.037}
    t = np.arange(n)
    parts, bear, rids = [], [], []
    for b in bearings:
        nuisance = 0.005 + 0.04 * (sum(map(ord, b)) % 97) / 97
        for m in range(n_rec):
            x = (np.sin(2 * np.pi * freq[label_of(b)] * t) + 0.7 * np.sin(2 * np.pi * nuisance * t)
                 + rng.normal(0, 0.5, n))
            parts.append(x.astype(np.float32))
            bear.append(b)
            rids.append(f"SYN/{b}/{m}::HostService")
    lengths = np.full(len(parts), n, dtype=np.int64)
    offsets = np.arange(len(parts), dtype=np.int64) * n
    return Signals(np.concatenate(parts), offsets, lengths, bear, rids)


class WindowSet:
    """Windows of one split, gathered on the training device in one indexing op per batch."""

    def __init__(self, sig: Signals, data: torch.Tensor, keep: list[str], overlap: float,
                 classes: list[str], domains: list[str]):
        device = data.device
        starts, rec = sig.windows(overlap, set(keep))
        self.data = data  # shared by every split; windows only index their own recordings
        self.rids = {sig.rids[i] for i in set(rec.tolist())}
        self.starts = torch.from_numpy(starts).to(device)
        bearing = np.array(sig.bearings)[rec]
        self.y = torch.tensor([classes.index(label_of(b)) for b in bearing], device=device)
        dom = {d: i for i, d in enumerate(domains)}
        self.d = torch.tensor([dom.get(b, -1) for b in bearing], device=device)
        self.bearing = bearing
        self._ar = torch.arange(WINDOW, device=device)

    def __len__(self) -> int:
        return len(self.starts)

    def x(self, idx: torch.Tensor) -> torch.Tensor:
        return self.data[self.starts[idx, None] + self._ar].unsqueeze(1)  # (B, 1, W)


# ----------------------------------------------------------------------------- models


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class Net(nn.Module):
    """pdm-bench's CNN1D (unchanged) plus optional domain and projection heads."""

    def __init__(self, n_classes: int, n_domains: int = 0):
        super().__init__()
        from pdm_bench.training.dl.models import CNN1D

        cnn = CNN1D(in_ch=1, n_classes=n_classes)
        self.features, self.classifier = cnn.features, cnn.classifier
        self.domain = (nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_domains))
                       if n_domains else None)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))

    def embed(self, x):
        return self.features(x).flatten(1)  # (B, 128)

    def forward(self, x):
        return self.classifier(self.embed(x))


def augment(x: torch.Tensor) -> torch.Tensor:
    """SimCLR views for vibration windows: gain, polarity, circular shift, noise, mask."""
    b, _, n = x.shape
    dev = x.device
    gain = torch.empty(b, 1, 1, device=dev).uniform_(0.7, 1.3)
    sign = torch.where(torch.rand(b, 1, 1, device=dev) < 0.5, -1.0, 1.0)
    x = x * gain * sign
    shift = torch.randint(0, n, (b,), device=dev)
    idx = (torch.arange(n, device=dev)[None, :] + shift[:, None]) % n
    x = torch.gather(x, 2, idx[:, None, :])
    x = x + 0.1 * x.std(dim=2, keepdim=True) * torch.randn_like(x)
    m_len = n // 10
    m0 = torch.randint(0, n - m_len, (b,), device=dev)
    pos = torch.arange(n, device=dev)[None, :]
    mask = (pos >= m0[:, None]) & (pos < m0[:, None] + m_len)
    return x.masked_fill(mask[:, None, :], 0.0)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    b = len(z1)
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, target)


# ----------------------------------------------------------------------------- training


def batches(n: int, size: int, gen: torch.Generator, device) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=gen).to(device)
    return [perm[i:i + size] for i in range(0, n, size)]


@torch.no_grad()
def predict(model: Net, ws: WindowSet, amp: bool, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ws), bs):
        idx = torch.arange(i, min(i + bs, len(ws)), device=ws.starts.device)
        with torch.autocast(ws.data.device.type, enabled=amp):
            out.append(model(ws.x(idx)).argmax(1).cpu())
    return torch.cat(out).numpy()


def pretrain_simclr(model: Net, pool: WindowSet, a, gen, amp: bool, log) -> None:
    """Labels and domains of the pool are never read: only its windows."""
    params = list(model.features.parameters()) + list(model.proj.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler(enabled=amp)
    for ep in range(1, a.pretrain_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for idx in batches(len(pool), a.pretrain_batch, gen, pool.starts.device):
            if len(idx) < 2:
                continue
            xs = pool.x(idx)
            with torch.autocast(xs.device.type, enabled=amp):
                z1 = model.proj(model.embed(augment(xs)))
                z2 = model.proj(model.embed(augment(xs)))
                loss = nt_xent(z1, z2, a.tau)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            tot, nb = tot + loss.item(), nb + 1
        log({"phase": "pretrain", "epoch": ep, "nt_xent": tot / max(nb, 1)})


def train(model: Net, src: WindowSet, tgt: WindowSet | None, a, gen, amp: bool, log) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=amp)
    dev = src.starts.device
    steps_per_epoch = math.ceil(len(src) / a.batch)
    total, step = a.epochs * steps_per_epoch, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        ce_sum = dom_sum = 0.0
        correct = seen = 0
        for idx in batches(len(src), a.batch, gen, dev):
            x, y = src.x(idx), src.y[idx]
            lam = a.lam * (2.0 / (1.0 + math.exp(-10.0 * step / total)) - 1.0)
            with torch.autocast(dev.type, enabled=amp):
                if a.method == "dann" and a.domain == "target":
                    tidx = torch.randint(0, len(tgt), (len(idx),), generator=gen).to(dev)
                    h = model.embed(torch.cat([x, tgt.x(tidx)]))  # target labels never read
                    hs = h[: len(idx)]
                    logits = model.classifier(hs)
                    dlog = model.domain(GradReverse.apply(h, lam))
                    dy = torch.cat([torch.zeros(len(idx)), torch.ones(len(idx))]).long().to(dev)
                    dom = F.cross_entropy(dlog.float(), dy)
                else:
                    hs = model.embed(x)
                    logits = model.classifier(hs)
                    dom = torch.zeros((), device=dev)
                    if a.method == "dann":
                        dlog = model.domain(GradReverse.apply(hs, lam)).float()
                        if a.domain == "bearing_in_class":
                            dlog = dlog.masked_fill(~a.same_class[y], float("-inf"))
                        dom = F.cross_entropy(dlog, src.d[idx])
                ce = F.cross_entropy(logits.float(), y, label_smoothing=0.05)
                loss = ce + dom
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            ce_sum += ce.item() * len(idx)
            dom_sum += dom.item() * len(idx)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(idx)
        sched.step(ce_sum / seen)  # train loss, as pdm-bench does without a val split
        row = {"phase": "train", "epoch": ep, "ce": ce_sum / seen, "train_acc": correct / seen,
               "lr": opt.param_groups[0]["lr"]}
        if a.method == "dann":
            row |= {"domain_loss": dom_sum / seen, "lam": lam}
        log(row)


# ----------------------------------------------------------------------------- run


def tag_of(a) -> str:
    if a.method == "dann":
        return f"dann-{a.domain}-lam{a.lam:g}"
    if a.method == "simclr":
        return f"simclr{'-tgt' if a.pretrain_target else ''}-pre{a.pretrain_epochs}"
    return "erm"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def setting_of(a) -> str:
    if (a.method == "dann" and a.domain == "target") or (a.method == "simclr" and a.pretrain_target):
        return "UDA (unlabeled scored-split windows used in training)"
    return "DG (no target data)"


def run(a) -> dict:
    tag = tag_of(a)
    run_dir = Path(a.out) / a.stage / tag / f"seed{a.seed}"
    summary_path = run_dir / "results" / f"{tag}_test_summary.json"
    if summary_path.exists() and not a.force:
        print(f"skip: {summary_path} exists (use --force to rerun)")
        return json.loads(summary_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = a.amp and dev.type == "cuda"
    torch.backends.cudnn.benchmark = True

    if a.synthetic:
        all_bearings = sorted({l.split()[1].split("/")[-2]
                               for l in (ROOT / "data/checksums/pu.sha256").read_text().splitlines()})
    else:
        root = Path(a.workspace)
        all_bearings = sorted(p.name for p in (root / PU_DIR).iterdir() if p.is_dir()
                              and any(p.glob("*.mat")))
    train_b, scored_b = stage_bearings(a.stage, all_bearings)
    if a.synthetic:
        sig = synthetic_signals(train_b + scored_b, seed=0)
    else:
        sig = load_signals(root, train_b + scored_b, Path(a.cache) if a.cache else None)
    data = torch.from_numpy(sig.data).to(dev)
    classes = sorted({label_of(b) for b in train_b})  # same order as pdm-bench label_to_id
    domains = train_b if a.method == "dann" and a.domain != "target" else []
    src = WindowSet(sig, data, train_b, TRAIN_OVERLAP, classes, domains)
    tgt_train = WindowSet(sig, data, scored_b, TRAIN_OVERLAP, classes, [])
    tgt_eval = WindowSet(sig, data, scored_b, TEST_OVERLAP, classes, [])
    assert not src.rids & tgt_eval.rids, "a recording is on both sides of the split"
    a.same_class = torch.tensor([[label_of(b) == c for b in domains] for c in classes],
                                dtype=torch.bool, device=dev) if domains else None

    n_dom = 2 if a.domain == "target" else len(domains)
    model = Net(len(classes), n_dom if a.method == "dann" else 0).to(dev)

    cfg = {k: v for k, v in vars(a).items() if k != "same_class"}
    cfg |= {"run": {"name": f"pu_cross_bearing_instance/{a.stage}"},
            "train": {"random_state": a.seed}, "variant": f"phase2-{a.stage}", "model": tag,
            "setting": setting_of(a), "git_sha": git_sha(), "device": str(dev), "amp": amp,
            "split": {"train_bearings": train_b, "scored_bearings": scored_b,
                      "window": WINDOW, "train_overlap": TRAIN_OVERLAP,
                      "test_overlap": TEST_OVERLAP, "channel": CHANNEL},
            "n_train_windows": len(src), "n_scored_windows": len(tgt_eval), "classes": classes}
    (run_dir / "pipeline_config.json").write_text(json.dumps(cfg, indent=2))
    history = []

    def log(row):
        history.append(row)
        print(json.dumps(row), flush=True)

    print(f"[{tag}] stage={a.stage} seed={a.seed} device={dev} amp={amp} "
          f"train={len(src)} windows from {len(train_b)} bearings, scored={len(tgt_eval)} "
          f"windows from {scored_b}; {cfg['setting']}", flush=True)
    t0 = time.time()
    if a.method == "simclr":
        pool = WindowSet(sig, data, train_b + scored_b, TRAIN_OVERLAP, classes, []) \
            if a.pretrain_target else src
        pretrain_simclr(model, pool, a, gen, amp, log)
    train(model, src, tgt_train, a, gen, amp, log)

    pred = predict(model, tgt_eval, amp)
    true = tgt_eval.y.cpu().numpy()
    labels = list(range(len(classes)))
    per_bearing = {b: float(accuracy_score(true[tgt_eval.bearing == b], pred[tgt_eval.bearing == b]))
                   for b in scored_b}
    metrics = {
        "macro_f1": float(f1_score(true, pred, average="macro", labels=labels, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "per_class_f1": dict(zip(classes, map(float, f1_score(true, pred, average=None,
                                                             labels=labels, zero_division=0)))),
        "per_bearing_accuracy": per_bearing,
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
    }
    out = {"model": tag, "split": a.stage, "seed": a.seed, "metrics": metrics,
           "seconds": round(time.time() - t0, 1)}
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"[{tag}] {a.stage} macro-F1 {metrics['macro_f1']:.4f}  "
          f"balanced acc {metrics['balanced_accuracy']:.4f}  ({out['seconds']}s)", flush=True)
    return out


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=["val", "test"], required=True)
    p.add_argument("--method", choices=["erm", "dann", "simclr"], required=True)
    p.add_argument("--domain", choices=["bearing", "bearing_in_class", "target"],
                   default="bearing_in_class")
    p.add_argument("--lam", type=float, default=0.3, help="max DANN weight (ramped up)")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--pretrain-batch", type=int, default=128)
    p.add_argument("--pretrain-target", action="store_true")
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--device", default="")
    p.add_argument("--workspace", default=str(ROOT / "data" / "raw"))
    p.add_argument("--cache", default=str(ROOT / "data" / "processed"))
    p.add_argument("--out", default=str(ROOT / "runs" / "phase2"))
    p.add_argument("--synthetic", action="store_true", help="fake signals, for smoke tests")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    import os

    args = parse(sys.argv[1:])
    if "WORKSPACE" in os.environ and args.workspace == str(ROOT / "data" / "raw"):
        args.workspace = os.environ["WORKSPACE"]
    run(args)
