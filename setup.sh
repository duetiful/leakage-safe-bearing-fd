#!/usr/bin/env bash
# Install pinned libraries, then the benchmark code (pdm-bench) as an editable clone in
# third_party/, so its Hydra configs and run scripts are available. Run from anywhere:
#   bash setup.sh
# pdm-bench declares Python <3.13, but its 278 tests pass on 3.13 (Colab) with these pins,
# so it is installed with --ignore-requires-python.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PDM_COMMIT=ca524087219fd47bb7fb51ce63563c05b34cd6ee

python -m pip install -r requirements.txt
if [[ ! -d third_party/pdm-bench/.git ]]; then
  git clone -q https://github.com/1Sensor/pdm-bench third_party/pdm-bench
fi
git -C third_party/pdm-bench fetch -q origin "$PDM_COMMIT" 2>/dev/null || true
git -C third_party/pdm-bench checkout -q "$PDM_COMMIT"
python -m pip install --no-deps --ignore-requires-python -e third_party/pdm-bench
python -m pip install --no-deps -e .
python -c "import pdm_bench, lsbfd, torch; print('ready; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
