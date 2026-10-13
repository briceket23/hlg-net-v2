#!/usr/bin/env bash
# Editable install of the four packages. Pass --cpu for the CPU-only PyTorch wheel.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--cpu" ]]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
fi
python -m pip install -r requirements/dev.txt
python -m pip install -e packages/malcore -e packages/malprep -e packages/malexp -e packages/malanalysis
python -m pytest -q packages/malcore/tests
