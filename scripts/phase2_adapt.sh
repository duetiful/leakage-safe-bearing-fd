#!/usr/bin/env bash
# Phase 2: domain adaptation on PU Cross-Bearing Instance, vibration channel only
# (src/lsbfd/adapt.py has the details). Every run is skipped if its result already exists,
# so a stage can be rerun after a Colab disconnect and picks up where it stopped.
#
#   bash scripts/phase2_adapt.sh val          # 1. hyperparameter search on source-only
#                                             #    validation bearings (seed 41, 20 epochs)
#   bash scripts/phase2_adapt.sh select       # 2. print the setting each method keeps
#   bash scripts/phase2_adapt.sh test 41      # 3. chosen settings on the test bearings,
#                                             #    one seed, 20 epochs (repeat for 42, 43)
#   bash scripts/phase2_adapt.sh uda 41       # 4. optional: the UDA variant, same way
#
# OUT (default runs/phase2) can point at Google Drive so results survive a Colab reset.
# CACHE (default data/processed) holds the per-bearing vibration cache; on Kaggle, build it
# first with scripts/prepare_pu_cache.sh and the raw .mat files are not needed.
# VAL_EPOCHS and TEST_EPOCHS (default 20 each; Phase 1 used 50, too slow for Colab at
# ~50 min per 20-epoch run) and EXTRA (extra lsbfd.adapt args, smoke tests only) are optional.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WORKSPACE="${WORKSPACE:-$ROOT/data/raw}"
OUT="$(realpath -m "${OUT:-$ROOT/runs/phase2}")"
cd "$ROOT"

CACHE="$(realpath -m "${CACHE:-$ROOT/data/processed}")"

adapt() { python -m lsbfd.adapt --out "$OUT" --cache "$CACHE" "$@" ${EXTRA:-}; }

chosen() {  # chosen lsbfd.adapt args for one family, from the validation runs
  python -m lsbfd.select "$OUT" | awk -F'\t' -v f="$1" '$1 == f {print $4}'
}

if [[ "${1:-}" != select && "${EXTRA:-}" != *--synthetic* ]]; then
  # Refuses to continue if a recording leaks. Only file names are read, so without the raw
  # data (cache-only machines) it checks the committed download manifest instead.
  if find "$WORKSPACE/datasets/paderborn-university-bearing-dataset" -name '*.mat' -print -quit \
       2>/dev/null | grep -q .; then
    python -m lsbfd.splits pu_cross_bearing_instance
  else
    fake="$(mktemp -d)"
    while read -r _ path; do mkdir -p "$fake/$(dirname "${path#data/raw/}")"
      touch "$fake/${path#data/raw/}"; done < data/checksums/pu.sha256
    WORKSPACE="$fake" python -m lsbfd.splits pu_cross_bearing_instance
    rm -rf "$fake"
  fi
fi
case "${1:-}" in
  val)
    E="--stage val --seed 41 --epochs ${VAL_EPOCHS:-20}"
    adapt $E --method erm
    for lam in 0.1 1; do
      adapt $E --method dann --domain bearing_in_class --lam "$lam"
      adapt $E --method dann --domain bearing --lam "$lam"
    done
    adapt $E --method simclr
    for lam in 0.1 1; do adapt $E --method dann --domain target --lam "$lam"; done
    python -m lsbfd.summarize "$OUT" ;;
  select)
    python -m lsbfd.select "$OUT" ;;
  test)
    seed="${2:?seed required, e.g. 41}"
    for fam in erm dann-dg simclr; do
      args="$(chosen "$fam")"
      [[ -n "$args" ]] || { echo "no validation run for $fam; run the val stage first" >&2; exit 1; }
      adapt --stage test --seed "$seed" --epochs "${TEST_EPOCHS:-20}" $args
    done
    python -m lsbfd.summarize "$OUT" ;;
  uda)
    seed="${2:?seed required, e.g. 41}"
    args="$(chosen dann-uda)"
    [[ -n "$args" ]] || { echo "no validation run for dann-uda; run the val stage first" >&2; exit 1; }
    adapt --stage test --seed "$seed" --epochs "${TEST_EPOCHS:-20}" $args
    python -m lsbfd.summarize "$OUT" ;;
  *) echo "usage: $0 val | select | test <seed> | uda <seed>" >&2; exit 1 ;;
esac
