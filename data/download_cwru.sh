#!/usr/bin/env bash
# Download CWRU into data/raw/ using the same Kaggle mirror as the pdm-bench benchmark
# (javadseraj/cwru-bearing-fault-data-set), so the benchmark's CWRU loader and queries
# work unchanged. Needs the kaggle CLI (in requirements.txt) and an API token in
# ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/datasets/javadseraj-cwru-bearing-fault-data-set"
DATASET_ID="javadseraj/cwru-bearing-fault-data-set"

if [[ -d "$DEST/Datasets/CWRU" ]]; then
  echo "CWRU already present at $DEST"
  exit 0
fi
command -v kaggle >/dev/null || { echo "kaggle CLI not found; pip install -r requirements.txt" >&2; exit 1; }
if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]] && [[ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]]; then
  echo "Kaggle credentials not found (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)." >&2
  exit 1
fi

mkdir -p "$DEST"
kaggle datasets download -d "$DATASET_ID" -p "$DEST" --unzip
echo "CWRU ready: $DEST"
"$ROOT/data/checksums.sh" cwru
