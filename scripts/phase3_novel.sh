#!/usr/bin/env bash
# Phase 3: methods that worked on neighbouring problems, on PU Cross-Bearing Instance
# (src/lsbfd/novel.py has the details). Reads the Phase 2 vibration cache ($CACHE, built by
# scripts/prepare_pu_cache.sh); every finished run is saved under $OUT and skipped next time.
#
#   bash scripts/phase3_novel.sh features   # 1. envelope-order cache, CPU (~10-30 min, once)
#   bash scripts/phase3_novel.sh ml         # 2. LR + boosted trees, 3 feature sets, 10 folds, CPU
#   bash scripts/phase3_novel.sh cnn        # 3. envelope CNN + AdaBN, seeds 41-43, benchmark split, GPU
#   bash scripts/phase3_novel.sh cnn-folds  # 4. envelope CNN, seed 41, all 10 folds, GPU
#   bash scripts/phase3_novel.sh raw-adabn  # 5. optional: raw-signal ERM CNN + AdaBN, seeds 41-43
#                                           #    (~50 min per seed, Phase 2 trainer, 20 epochs)
#   bash scripts/phase3_novel.sh report     # tables -> $OUT/phase3_tables.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$(realpath -m "${CACHE:-$ROOT/data/processed}")"
OUT="$(realpath -m "${OUT:-$ROOT/runs/phase3}")"
FOLDS="${FOLDS:-10}"
cd "$ROOT"

n="$(ls "$CACHE"/pu_vibration_1_*.npz 2>/dev/null | wc -l)"
[[ "$n" -eq 32 ]] || { echo "need the 32-bearing vibration cache in $CACHE (have $n); run scripts/prepare_pu_cache.sh" >&2; exit 1; }

novel() { python -m lsbfd.novel "$1" --cache "$CACHE" --out "$OUT" --folds "$FOLDS" "${@:2}"; }

case "${1:-}" in
  features)  novel features --workers "${WORKERS:-4}" ;;
  ml)        novel ml; novel report ;;
  cnn)       for s in 41 42 43; do novel cnn --seed "$s"; done; novel report ;;
  cnn-folds) novel cnn --seed 41 --all-folds; novel report ;;
  raw-adabn)
    for s in 41 42 43; do
      python -m lsbfd.adapt --stage test --method erm --adabn --seed "$s" \
        --epochs "${TEST_EPOCHS:-20}" --cache "$CACHE" --out "$OUT/raw"
    done
    python -m lsbfd.summarize "$OUT/raw"
    python - "$OUT/raw" <<'PY'
import json, sys, pathlib, statistics as st
rows = [json.loads(p.read_text())["metrics"] for p in pathlib.Path(sys.argv[1]).rglob("erm-adabn_test_summary.json")]
for k in ("macro_f1",):
    for name, get in (("raw CNN (no adaptation)", lambda m: m[k]),
                      ("raw CNN + AdaBN pooled", lambda m: m["adabn_pooled"][k]),
                      ("raw CNN + AdaBN per bearing", lambda m: m["adabn_per_bearing"][k])):
        v = [get(m) for m in rows]
        print(f"{name}: {st.mean(v):.3f} ± {st.stdev(v) if len(v) > 1 else 0:.3f} ({len(v)} seeds)")
PY
    ;;
  report)    novel report ;;
  *) echo "usage: $0 features | ml | cnn | cnn-folds | raw-adabn | report" >&2; exit 1 ;;
esac
