Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$PythonExe = "python",
    [string]$RunDir = ""
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runsRoot = Join-Path $repoRoot "artifact\runs"

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $latest = Get-ChildItem $runsRoot -Directory |
        Where-Object { $_.Name -like "ltst_full_86_*" } |
        Sort-Object Name |
        Select-Object -Last 1
    if ($null -eq $latest) {
        throw "No ltst_full_86_* run directory found under $runsRoot"
    }
    $RunDir = $latest.FullName
}

$script = Join-Path $repoRoot "ESP32\ltst_esp32_contract_sim.py"

& $PythonExe $script `
    --run-dir $RunDir
