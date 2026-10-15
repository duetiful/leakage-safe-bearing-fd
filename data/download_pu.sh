#!/usr/bin/env bash
# Download the Paderborn University (PU) bearing dataset into data/raw/ from the official
# BearingDataCenter, as pdm-bench does. Needs curl (or wget), and unrar or 7z.
# aria2c is used for parallel downloads when available, otherwise curl one by one.
# Usage: data/download_pu.sh [--list]   (--list only prints the archive URLs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="https://groups.uni-paderborn.de/kat/BearingDataCenter/"
DEST="$ROOT/data/raw/datasets/paderborn-university-bearing-dataset"
RARS="$DEST/rars"

fetch() { if command -v curl >/dev/null; then curl -fsSL "$1"; else wget -qO- "$1"; fi; }

list_urls() {
  fetch "$BASE_URL" | grep -oE 'href="[^"]+\.rar"' | sed -E 's/href="([^"]+)"/\1/' \
    | sed -E "s#^#${BASE_URL}#"
}

if [[ "${1:-}" == "--list" ]]; then
  list_urls
  exit 0
fi

if find "$DEST" -name '*.mat' -print -quit 2>/dev/null | grep -q .; then
  echo "PU already present at $DEST"
  exit 0
fi

urls="$(list_urls)"
[[ -n "$urls" ]] || { echo "No .rar links found at $BASE_URL" >&2; exit 1; }
mkdir -p "$RARS"
echo "Downloading $(wc -l <<<"$urls") PU archives into $RARS ..."
if command -v aria2c >/dev/null; then
  aria2c -c -x 8 -s 8 -j 8 -d "$RARS" --input-file=- <<<"$urls"
else
  while read -r url; do curl -fSL -C - -o "$RARS/$(basename "$url")" "$url"; done <<<"$urls"
fi

echo "Extracting into $DEST ..."
for archive in "$RARS"/*.rar; do
  if command -v unrar >/dev/null; then
    unrar x -o+ -inul "$archive" "$DEST/"
  elif command -v 7z >/dev/null; then
    7z x -y "-o$DEST" "$archive" >/dev/null
  else
    echo "unrar or 7z is required" >&2; exit 1
  fi
done
rm -rf "$RARS"
echo "PU ready: $DEST"
"$ROOT/data/checksums.sh" pu
