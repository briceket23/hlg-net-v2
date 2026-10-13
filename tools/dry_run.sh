#!/usr/bin/env bash
# Dry run of the data pipeline on the real label files with synthetic full-size images (pipeline test only).
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "from malprep.synthetic import render_mpidb_images as r; r('raw_data/mp_idb/labels','raw_data/mp_idb/data.yaml','raw_data/mp_idb_synthetic')"
sed -e 's#raw_data/mp_idb/images#raw_data/mp_idb_synthetic#' -e 's#^label_source:.*#label_source: "DRY RUN"#' \
    -e 's#out:          data #out:          dryrun #' -e 's#splits:       splits#splits:       dryrun/splits#' configs/data.yaml > configs/dry.yaml
python -m malprep all --config configs/dry.yaml --allow-incomplete-qc
python -m malprep gate --config configs/dry.yaml || true
