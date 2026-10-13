# Dry run of the data pipeline on the real label files with synthetic images (pipeline test only).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -c "from malprep.synthetic import render_mpidb_images as r; r('raw_data/mp_idb/labels','raw_data/mp_idb/data.yaml','raw_data/mp_idb_synthetic')"
(Get-Content configs\data.yaml) -replace 'raw_data/mp_idb/images', 'raw_data/mp_idb_synthetic' `
    -replace '^label_source:.*', 'label_source: "DRY RUN"' -replace 'out:          data ', 'out:          dryrun ' `
    -replace 'splits:       splits', 'splits:       dryrun/splits' | Set-Content -Encoding utf8 configs\dry.yaml
python -m malprep all --config configs\dry.yaml --allow-incomplete-qc
python -m malprep gate --config configs\dry.yaml
