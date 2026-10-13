#!/usr/bin/env bash
# Steps 8-12 in protocol order on the GPU machine. Every command is resumable; rerun the script after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/data.yaml}
python -m malexp preflight --data-config "$CONFIG"                                   # Step 8: device/precision check
python -m malexp run --data-config "$CONFIG" --unit slide --tiers 1 2 3              # Tiers 1-3, slide-grouped (primary)
python -m malanalysis check --config configs/analysis.yaml                           # gate G5/G6 before any test metric
python -m malexp run --data-config "$CONFIG" --unit image --tiers 1 2                # image-grouped sensitivity
python -m malexp run --data-config "$CONFIG" --unit slide --tiers 1 2 \
       --backbone efficientnet_b0.ra_in1k --input-size 224 --runs runs/slide_effb0  # Tier 4 replication
python -m malexp complexity --data-config "$CONFIG" --out analysis                   # Step 13 params / MACs / latency
python -m malanalysis run --config configs/analysis.yaml                             # Steps 13-14, 17
