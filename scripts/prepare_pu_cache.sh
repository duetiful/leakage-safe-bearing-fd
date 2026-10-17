#!/usr/bin/env bash
# Build the vibration-only PU cache used by Phase 2 without ever holding the whole raw
# dataset on disk: one bearing at a time, download its archive, extract, verify every file
# against data/checksums/pu.sha256, write <CACHE>/pu_vibration_1_<bearing>.npz, then delete
# the raw files. Peak disk is one bearing (well under 1 GB); the finished cache is ~2.6 GB.
# Made for machines with little disk (Kaggle). Already-cached bearings are skipped.
#   CACHE=/kaggle/working/cache bash scripts/prepare_pu_cache.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$(realpath -m "${CACHE:-$ROOT/data/processed}")"
DEST="$ROOT/data/raw/datasets/paderborn-university-bearing-dataset"
cd "$ROOT"

for url in $(data/download_pu.sh --list); do
  b="$(basename "$url" .rar)"
  grep -q "/$b/" data/checksums/pu.sha256 || { echo "skip $url: not a PU bearing"; continue; }
  if [[ -f "$CACHE/pu_vibration_1_$b.npz" ]]; then echo "$b: cached"; continue; fi
  mkdir -p "$DEST"
  curl -fsSL --retry 3 -o "$DEST/$b.rar" "$url"
  if command -v unrar >/dev/null; then unrar x -o+ -inul "$DEST/$b.rar" "$DEST/"
  else 7z x -y "-o$DEST" "$DEST/$b.rar" >/dev/null; fi
  rm "$DEST/$b.rar"
  grep "/$b/" data/checksums/pu.sha256 | sha256sum --quiet -c -
  python -c "from pathlib import Path; from lsbfd.adapt import cache_bearing; \
print(cache_bearing(Path('$ROOT/data/raw'), '$b', Path('$CACHE')))"
  rm -rf "${DEST:?}/$b"
done
n="$(ls "$CACHE"/pu_vibration_1_*.npz | wc -l)"
echo "PU cache in $CACHE: $n of 32 bearings"
[[ "$n" -eq 32 ]]
