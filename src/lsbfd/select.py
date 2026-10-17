"""Pick each Phase 2 method's hyperparameters from the source-only validation runs.

    python -m lsbfd.select runs/phase2        # prints one line of lsbfd.adapt args per method

The rule is fixed in advance: within each family, the setting with the highest validation
macro-F1 (seed 41) wins; ties go to the weaker setting (smaller lambda). The test bearings
play no part. Families: erm, dann-dg (bearing or bearing_in_class domains), dann-uda
(target domain), simclr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def family(cfg: dict) -> str:
    if cfg["method"] == "dann":
        return "dann-uda" if cfg["domain"] == "target" else "dann-dg"
    return cfg["method"]


def args_of(cfg: dict) -> str:
    out = f"--method {cfg['method']}"
    if cfg["method"] == "dann":
        out += f" --domain {cfg['domain']} --lam {cfg['lam']:g}"
    if cfg["method"] == "simclr":
        out += f" --pretrain-epochs {cfg['pretrain_epochs']} --tau {cfg['tau']:g}"
        out += " --pretrain-target" if cfg["pretrain_target"] else ""
    return out


def select(base: Path) -> dict[str, tuple[float, str, str]]:
    """Family -> (validation macro-F1, run tag, lsbfd.adapt args) of the chosen setting."""
    best: dict[str, tuple[tuple[float, float], str, str]] = {}
    for summary in sorted((base / "val").glob("*/seed41/results/*_test_summary.json")):
        cfg = json.loads((summary.parent.parent / "pipeline_config.json").read_text())
        key = (json.loads(summary.read_text())["metrics"]["macro_f1"], -cfg.get("lam", 0.0))
        fam = family(cfg)
        if fam not in best or key > best[fam][0]:
            best[fam] = (key, cfg["model"], args_of(cfg))
    return {fam: (key[0], tag, args) for fam, (key, tag, args) in best.items()}


def main(argv: list[str]) -> int:
    base = Path(argv[0] if argv else "runs/phase2")
    best = select(base)
    if not best:
        print(f"No validation runs under {base / 'val'}", file=sys.stderr)
        return 1
    for fam in ("erm", "dann-dg", "simclr", "dann-uda"):
        if fam in best:
            f1, tag, args = best[fam]
            print(f"{fam}\t{tag}\t{f1:.4f}\t{args}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
