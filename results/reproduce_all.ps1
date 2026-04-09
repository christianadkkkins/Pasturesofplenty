Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$PythonExe = "python"
)

$ltstFullScript = Join-Path $PSScriptRoot "reproduce_ltst_full.ps1"
$ltstHmmScript = Join-Path $PSScriptRoot "reproduce_ltst_transition_hmm.ps1"
$esp32Script = Join-Path $PSScriptRoot "reproduce_esp32_contract.ps1"
$solarScript = Join-Path $PSScriptRoot "reproduce_solar_hmm.ps1"

& $ltstFullScript -PythonExe $PythonExe
& $ltstHmmScript -PythonExe $PythonExe
& $esp32Script -PythonExe $PythonExe
& $solarScript -PythonExe $PythonExe
