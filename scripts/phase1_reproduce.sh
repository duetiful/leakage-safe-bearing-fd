#!/usr/bin/env bash
# Phase 1: reproduce the two target numbers on PU Cross-Bearing Instance with the
# benchmark's own code and exactly the settings behind the paper's tables
# (see docs/benchmark-protocol.md section 5).
#
#   bash scripts/phase1_reproduce.sh lr           # logistic regression (CPU, ~minutes)
#   bash scripts/phase1_reproduce.sh cnn 41       # 1D CNN, one seed (GPU, longest step)
#   bash scripts/phase1_reproduce.sh all          # lr + cnn seeds 41 42 43
#
# Variants for the rest of M1 (same settings, one thing changed each):
#   vib-lr, vib-cnn <seed>      vibration channel only (drops the two motor currents)
#   leaky-lr, leaky-cnn <seed>  bearing-leaky split: measurements 1, 6, 11, 16 of every
#                               bearing are the test set, so every test bearing is also
#                               in training (no recording is shared)
#
# OUT (default runs/phase1) can point at Google Drive so results survive a Colab reset.
# EXTRA appends Hydra overrides to every run (used only for smoke tests).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WORKSPACE="${WORKSPACE:-$ROOT/data/raw}"
OUT="$(realpath -m "${OUT:-$ROOT/runs/phase1}")"
TASK=pu_cross_bearing_instance
DEVICE="$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
cd "$ROOT/third_party/pdm-bench"

ML=(python -m pdm_bench.pipelines.ml.pipeline)
DL=(python -m pdm_bench.pipelines.dl.pipeline)
SUB=""
SPLIT=()
case "${1:-all}" in
  vib-*)
    ML=(python -m lsbfd.run_variant --channels vibration_1 ml)
    DL=(python -m lsbfd.run_variant --channels vibration_1 dl)
    SUB=vib/ ;;
  leaky-*)
    SUB=leaky/
    SPLIT=("dataset.train_query=\"raster == 'HostService' and measurement_no not in [1, 6, 11, 16]\""
           "dataset.test_query=\"raster == 'HostService' and measurement_no in [1, 6, 11, 16]\"") ;;
esac

run_lr() {
  # "long" ML profile from run_phm_benchmark.sh, LR only (other classifiers do not
  # affect LR's result; each gets its own search with a fixed seed).
  "${ML[@]}" \
    "task=$TASK" "run.output_dir=$OUT/${SUB}lr" "${SPLIT[@]}" \
    "train.classifier_names=[LogisticRegression]" \
    train.bayes_n_iter=30 train.bayes_cv=5 train.bayes_n_points=2 \
    windowing.train_overlap=0.75 windowing.test_overlap=0.5 \
    "features.time_features=[pp,zp,rms,cf,std,kurt,sf,eo,FM4,FM6,FM8,clf,ii,FN4]" \
    "features.freq_features=[mf,fc,rms_f,std_f]" ${EXTRA:-}
}

run_cnn() {
  # "winners" entry pu_cross_bearing_instance_long_cnn1d from run_phm_followup_dl.sh.
  local seed="$1"
  "${DL[@]}" \
    "task=$TASK/cnn1d" "run.output_dir=$OUT/${SUB}cnn/seed$seed" "${SPLIT[@]}" \
    train.epochs=50 train.batch_size=64 train.num_workers=0 "train.device=$DEVICE" \
    windowing.train_overlap=0.75 windowing.test_overlap=0.5 \
    "+train.random_state=$seed" ${EXTRA:-}
}

python -m lsbfd.splits "$TASK"   # refuses to continue if any recording leaks
case "${1:-all}" in
  lr|vib-lr|leaky-lr)    run_lr ;;
  cnn|vib-cnn|leaky-cnn) run_cnn "${2:?seed required, e.g. 41}" ;;
  all) run_lr; for s in 41 42 43; do run_cnn "$s"; done ;;
  *) echo "usage: $0 [vib-|leaky-]lr | [vib-|leaky-]cnn <seed> | all" >&2; exit 1 ;;
esac
python -m lsbfd.summarize "$OUT"
