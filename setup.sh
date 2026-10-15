#!/usr/bin/env bash
# Install pinned libraries, then the benchmark package. Run from the repo root
# inside your venv / conda env / Colab:  bash setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python -m pip install -r requirements.txt
python -m pip install --no-deps --ignore-requires-python -r requirements-bench.txt
python -c "import pdm_bench, torch, sklearn; print('pdm-bench ready; torch', torch.__version__)"
