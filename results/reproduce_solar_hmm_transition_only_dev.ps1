Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$PythonExe = "python"
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "solar.py"

& $PythonExe $scriptPath `
    --start-time 2007-01-01 `
    --end-time 2008-12-31 `
    --write-feature-csv `
    --beta-short 0.20 `
    --beta-long 0.003 `
    --projective-min-state-energy 1e-8 `
    --hmm-rolling-window-minutes 15 `
    --hmm-merge-gap-minutes 5 `
    --hmm-alert-active-tail-minutes 0 `
    --hmm-alert-max-minutes 180 `
    --hmm-alert-cooldown-minutes 180
