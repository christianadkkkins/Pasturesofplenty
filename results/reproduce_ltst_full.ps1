Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$PythonExe = "python",
    [string]$Records = "all"
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "artifact\ltst_full_86_study.py"

& $PythonExe $scriptPath `
    --records $Records `
    --db ltstdb `
    --st-ext stc `
    --beat-ext atr `
    --top-n 1102 `
    --checkpoint-every 5 `
    --save-beat-level
