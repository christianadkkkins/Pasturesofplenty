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

$hmmScript = Join-Path $repoRoot "artifact\ltst_transition_hmm.py"
$deriveScript = Join-Path $repoRoot "results\derive_ltst_transition_hmm_metrics.py"
$outputDir = Join-Path $RunDir "transition_hmm_onset_cooldown32"

& $PythonExe $hmmScript `
    --run-dir $RunDir `
    --out-dir-name transition_hmm_onset_cooldown32 `
    --alert-active-tail-beats 16 `
    --alert-max-beats 96 `
    --alert-cooldown-beats 32

& $PythonExe $deriveScript `
    --run-dir $outputDir `
    --pre-event-beats 500
