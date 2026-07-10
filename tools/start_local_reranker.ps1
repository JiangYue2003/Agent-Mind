param(
    [int]$Port = 18080
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv-reranker\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Local reranker is not installed. Run .\tools\install_local_reranker.ps1 first."
}

Set-Location $ProjectRoot
& $VenvPython -m uvicorn local_reranker.app:app --host 0.0.0.0 --port $Port --workers 1
