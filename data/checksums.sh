#!/usr/bin/env bash
# Record or verify SHA-256 checksums of the raw .mat files.
#   data/checksums.sh cwru|pu           write data/checksums/<name>.sha256 if missing, else verify
# Commit the .sha256 files after the first download so every later download is checked
# against the same bytes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  cwru) DIR="data/raw/datasets/javadseraj-cwru-bearing-fault-data-set" ;;
  pu)   DIR="data/raw/datasets/paderborn-university-bearing-dataset" ;;
  *) echo "usage: $0 cwru|pu" >&2; exit 1 ;;
esac
SUMS="$ROOT/data/checksums/$1.sha256"
mkdir -p "$(dirname "$SUMS")"
cd "$ROOT"

if [[ -f "$SUMS" ]]; then
  sha256sum --quiet -c "$SUMS" && echo "$1: all checksums match $SUMS"
else
  find "$DIR" -type f -name '*.mat' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$SUMS"
  echo "$1: wrote $(wc -l < "$SUMS") checksums to $SUMS (commit this file)"
fi
