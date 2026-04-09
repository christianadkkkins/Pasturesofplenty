Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$PythonExe = "python"
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "solar.py"

& $PythonExe $scriptPath `
    --write-feature-csv `
    --write-alignment-csv `
    --beta-short 0.10 `
    --beta-long 0.01 `
    --projective-min-state-energy 1e-8 `
    --hmm-rolling-window-minutes 15 `
    --hmm-merge-gap-minutes 15 `
    --hmm-alert-active-tail-minutes 30 `
    --hmm-alert-max-minutes 180 `
    --hmm-alert-cooldown-minutes 60
