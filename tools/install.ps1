# Editable install of the four packages on Windows (PowerShell). Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 [-Cuda cu124 | -Cpu]
param([string]$Cuda = "", [switch]$Cpu)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if ($Cpu) { python -m pip install --index-url https://download.pytorch.org/whl/cpu torch }
elseif ($Cuda -ne "") { python -m pip install --index-url "https://download.pytorch.org/whl/$Cuda" torch }
python -m pip install -r requirements\dev.txt
python -m pip install -e packages\malcore -e packages\malprep -e packages\malexp -e packages\malanalysis
python -m pytest -q packages\malcore\tests
