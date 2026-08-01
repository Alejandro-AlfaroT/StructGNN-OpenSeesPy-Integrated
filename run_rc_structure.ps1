$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
if (-not $env:RC_SHOW_PLOTS) {
    $env:RC_SHOW_PLOTS = "0"
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $env:USERPROFILE "anaconda3\envs\OpPy\python.exe"
$main = Join-Path $projectRoot "RC Structure\Main.py"

if (-not (Test-Path $python)) {
    throw "OpPy Python not found at $python"
}

if (-not (Test-Path $main)) {
    throw "RC Structure Main.py not found at $main"
}

& $python $main
