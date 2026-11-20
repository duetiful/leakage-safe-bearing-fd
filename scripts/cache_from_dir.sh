#!/usr/bin/env bash
# Build the vibration cache from a local copy of PU (e.g. a Kaggle input dataset) instead of
# the Paderborn server. Finds each bearing's .mat files (or its <bearing>.rar) anywhere under
# SRC, checks every file against data/checksums/pu.sha256, and refuses any mismatch.
#   CACHE=/kaggle/working/cache bash scripts/cache_from_dir.sh /kaggle/input
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(realpath "$1")"
CACHE="$(realpath -m "${CACHE:-$ROOT/data/processed}")"
DEST="$ROOT/data/raw/datasets/paderborn-university-bearing-dataset"
cd "$ROOT"
mkdir -p "$CACHE"
for b in $(awk '{print $2}' data/checksums/pu.sha256 | awk -F/ '{print $(NF-1)}' | sort -u); do
  if [[ -f "$CACHE/pu_vibration_1_$b.npz" ]]; then echo "$b: cached"; continue; fi
  rm -rf "${DEST:?}/$b"; mkdir -p "$DEST/$b"
  mats="$(find -L "$SRC" -type f -name "*_${b}_*.mat" 2>/dev/null)"
  if [[ -n "$mats" ]]; then
    while read -r f; do ln -sf "$f" "$DEST/$b/$(basename "$f")"; done <<<"$mats"
  else
    rar="$(find -L "$SRC" -type f -iname "$b.rar" 2>/dev/null | head -1)"
    [[ -n "$rar" ]] || { echo "$b: no .mat files or $b.rar under $SRC" >&2; exit 1; }
    if command -v unrar >/dev/null; then unrar x -o+ -inul "$rar" "$DEST/"
    else 7z x -y "-o$DEST" "$rar" >/dev/null; fi
  fi
  grep "/$b/" data/checksums/pu.sha256 | sha256sum --quiet -c - \
    || { echo "$b: checksum mismatch, this copy differs from the official files" >&2; exit 1; }
  python -c "from pathlib import Path; from lsbfd.adapt import cache_bearing; \
print(cache_bearing(Path('$ROOT/data/raw'), '$b', Path('$CACHE')))"
  rm -rf "${DEST:?}/$b"
done
echo "PU cache in $CACHE: $(ls "$CACHE"/pu_vibration_1_*.npz | wc -l) of 32 bearings"
