#!/usr/bin/env bash
# Feature-space analysis (lsbfd.featurespace) on the PU vibration cache.
#   CACHE=/kaggle/working/cache OUT=/kaggle/working/featurespace bash scripts/featurespace.sh all
# Steps: features (CPU, ~20 min), erm and dann (GPU, ~50 min each at 20 epochs), analyze (CPU).
# Every step skips work already done, so rerun the same line after a timeout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CACHE="${CACHE:-$PWD/data/processed}"
OUT="${OUT:-$PWD/runs/featurespace}"
EPOCHS="${EPOCHS:-20}"
step="${1:-all}"
run() { python -m lsbfd.featurespace "$@" --cache "$CACHE" --out "$OUT"; }
if [[ $step == features || $step == all ]]; then run features; fi
for m in erm dann; do
  if [[ $step == "$m" || $step == all ]]; then
    if [[ -f "$OUT/cnn_$m/embedding.npz" ]]; then echo "$m: done"; else
      run embed --method "$m" --env "$CACHE/fs_env_raw" --epochs "$EPOCHS" --train-overlap 0.75
    fi
  fi
done
if [[ $step == analyze || $step == all ]]; then run analyze; fi
if [[ $step == zip || $step == all ]]; then
  (cd "$(dirname "$OUT")" && zip -qr featurespace.zip "$(basename "$OUT")" -x '*/model.pt')
  echo "send back: $(dirname "$OUT")/featurespace.zip"
fi
