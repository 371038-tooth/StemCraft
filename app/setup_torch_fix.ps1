param(
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

Write-Host "Torch environment setup starting..." -ForegroundColor Cyan
Write-Host ""

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
    Write-Host "venv detected. Installing Torch into venv." -ForegroundColor Green
} else {
    $pythonCmd = "python"
    Write-Host "venv not found. Installing Torch into current Python environment." -ForegroundColor Yellow
}

$pytorchIndexUrl = "https://download.pytorch.org/whl/cpu"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $pytorchIndexUrl = "https://download.pytorch.org/whl/cu128"
    Write-Host "NVIDIA GPU detected. Installing CUDA Torch (cu128)." -ForegroundColor Green
} else {
    Write-Host "NVIDIA GPU not detected. Installing CPU Torch." -ForegroundColor Yellow
}

$vcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
$vcRedistPath = Join-Path $env:TEMP "vc_redist.x64.exe"

try {
    Write-Host "Checking Visual C++ Redistributable..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $vcRedistUrl -OutFile $vcRedistPath -UseBasicParsing
    Start-Process -FilePath $vcRedistPath -ArgumentList @("/install", "/quiet", "/norestart") -Wait
    Write-Host "Visual C++ Redistributable installation completed." -ForegroundColor Green
} catch {
    Write-Host "Visual C++ Redistributable installation failed." -ForegroundColor Red
    Write-Host "Manual download: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Reinstalling Torch..." -ForegroundColor Yellow

& $pythonCmd -m pip uninstall torch torchvision torchaudio -y
& $pythonCmd -m pip install torch torchvision torchaudio --index-url $pytorchIndexUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host "Torch installation failed." -ForegroundColor Red
    if (-not $NoPause) {
        Read-Host "Press Enter to exit"
    }
    exit 1
}

Write-Host ""
Write-Host "Testing Torch..." -ForegroundColor Yellow

$testScript = @'
import torch
print('Torch ' + torch.__version__ + ' OK')
print('CUDA available: ' + str(torch.cuda.is_available()))
'@
& $pythonCmd -c $testScript

if ($LASTEXITCODE -ne 0) {
    Write-Host "Torch test failed." -ForegroundColor Red
    if (-not $NoPause) {
        Read-Host "Press Enter to exit"
    }
    exit 1
}

Write-Host ""
Write-Host "Torch setup completed." -ForegroundColor Green

if (-not $NoPause) {
    Write-Host ""
    Read-Host "Press Enter to exit"
}