"""Collect macro-F1 from pdm-bench run folders into one table.

    python -m lsbfd.summarize runs/phase1
Prints a per-run table plus mean ± std per model, and writes <dir>/summary.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def collect(base: Path) -> pd.DataFrame:
    rows = []
    for summary in sorted(base.rglob("*_test_summary.json")):
        run_dir = summary.parent.parent
        cfg_path = run_dir / "pipeline_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        metrics = json.loads(summary.read_text())["metrics"]
        rel = run_dir.relative_to(base)
        variant = rel.parts[0] if rel.parts[0] in {"vib", "leaky"} else "as_released"
        rows.append(
            {
                "variant": variant,
                "model": summary.name.removesuffix("_test_summary.json"),
                "task": cfg.get("run", {}).get("name", ""),
                "seed": cfg.get("train", {}).get("random_state", ""),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "accuracy": metrics.get("accuracy"),
                "run_dir": str(rel),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    base = Path(argv[0] if argv else "runs/phase1")
    df = collect(base)
    if df.empty:
        print(f"No *_test_summary.json under {base}")
        return 1
    df.to_csv(base / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.drop(columns="run_dir").to_string(index=False))
        agg = df.groupby(["variant", "model"])["macro_f1"].agg(["count", "mean", "std"]).round(4)
        print("\nmacro-F1 by variant and model:\n" + agg.to_string())
    print(f"\nWrote {base / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
