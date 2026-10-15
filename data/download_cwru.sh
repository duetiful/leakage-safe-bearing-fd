#!/usr/bin/env bash
# Download CWRU into data/raw/ using the same Kaggle mirror as the pdm-bench benchmark
# (javadseraj/cwru-bearing-fault-data-set), so the benchmark's CWRU loader and queries
# work unchanged. Needs the kaggle CLI (in requirements.txt) and Kaggle credentials:
# a new-style API token (KAGGLE_API_TOKEN or ~/.kaggle/access_token), or a legacy
# ~/.kaggle/kaggle.json / KAGGLE_USERNAME + KAGGLE_KEY.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/datasets/javadseraj-cwru-bearing-fault-data-set"
DATASET_ID="javadseraj/cwru-bearing-fault-data-set"

if [[ -d "$DEST/Datasets/CWRU" ]]; then
  echo "CWRU already present at $DEST"
  exit 0
fi
command -v kaggle >/dev/null || { echo "kaggle CLI not found; pip install -r requirements.txt" >&2; exit 1; }
KDIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "$KDIR/access_token" && ! -f "$KDIR/kaggle.json" ]] \
   && [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "Kaggle credentials not found. Set KAGGLE_API_TOKEN, or save the token to $KDIR/access_token." >&2
  exit 1
fi

mkdir -p "$DEST"
kaggle datasets download -d "$DATASET_ID" -p "$DEST" --unzip
echo "CWRU ready: $DEST"
"$ROOT/data/checksums.sh" cwru
