param(
    [string]$PythonExecutable = "python",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv-reranker"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

& $PythonExecutable -m venv $VenvPath
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install torch --index-url $TorchIndexUrl
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "local_reranker\requirements.txt")
& $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
