# Steps 8-12 in protocol order on the Windows GPU machine. Every command is resumable; rerun after a crash.
#   powershell -ExecutionPolicy Bypass -File tools\run_matrix.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$cfg = "configs\data.yaml"
python -m malexp preflight --data-config $cfg;                                   if ($LASTEXITCODE) { exit 1 }
python -m malexp run --data-config $cfg --unit slide --tiers 1 2 3;              if ($LASTEXITCODE) { exit 1 }
python -m malanalysis check --config configs\analysis.yaml;                      if ($LASTEXITCODE) { exit 1 }
python -m malexp run --data-config $cfg --unit image --tiers 1 2;                if ($LASTEXITCODE) { exit 1 }
python -m malexp run --data-config $cfg --unit slide --tiers 1 2 --backbone efficientnet_b0.ra_in1k --input-size 224 --runs runs\slide_effb0
if ($LASTEXITCODE) { exit 1 }
python -m malexp complexity --data-config $cfg --out analysis;                   if ($LASTEXITCODE) { exit 1 }
python -m malanalysis run --config configs\analysis.yaml
