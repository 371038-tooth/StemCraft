# StemCraft 起動スクリプト (PowerShell版)
# 使用方法: PowerShellで実行
# powershell -ExecutionPolicy Bypass -File run.ps1

$ErrorActionPreference = "Stop"

function Get-SystemPythonCommand {
    $candidates = @(
        @{ Type = "launcher"; Command = "py"; Args = @("-3.11") },
        @{ Type = "launcher"; Command = "py"; Args = @("-3") },
        @{ Type = "command"; Command = "python"; Args = @() },
        @{ Type = "path"; Command = "$env:LocalAppData\Programs\Python\Python311\python.exe"; Args = @() },
        @{ Type = "path"; Command = "$env:ProgramFiles\Python311\python.exe"; Args = @() },
        @{ Type = "path"; Command = "$env:ProgramFiles(x86)\Python311\python.exe"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        try {
            if ($candidate.Type -eq "path" -and -not (Test-Path $candidate.Command)) {
                continue
            }

            $testArgs = @($candidate.Args + "-c" + "import sys")
            & $candidate.Command @testArgs *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }

    return $null
}

function Install-BundledPython {
    param(
        [string]$InstallerPath
    )

    if (-not (Test-Path $InstallerPath)) {
        throw "同梱 Python インストーラが見つかりません: $InstallerPath"
    }

    Write-Host "Python が見つかりません。同梱インストーラを実行します..." -ForegroundColor Yellow
    Start-Process -FilePath $InstallerPath -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_pip=1",
        "Include_test=0",
        "Include_launcher=1",
        "Shortcuts=0"
    ) -Wait

    $env:PATH = "$env:LocalAppData\Programs\Python\Python311;$env:LocalAppData\Programs\Python\Python311\Scripts;$env:PATH"
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║            StemCraft - ボーカル分離プレイヤー             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 仮想環境のパス
$venv_activate = "venv\Scripts\Activate.ps1"
$venvPython = "venv\Scripts\python.exe"
$pythonInstaller = Join-Path $PSScriptRoot "installers\python-3.11.9-amd64.exe"

if (-not (Test-Path $venvPython)) {
    $pythonCmd = Get-SystemPythonCommand
    if ($null -eq $pythonCmd) {
        Install-BundledPython -InstallerPath $pythonInstaller
        $pythonCmd = Get-SystemPythonCommand
    }

    if ($null -eq $pythonCmd) {
        Write-Host ""
        Write-Host "❌ エラー: Python の検出に失敗しました" -ForegroundColor Red
        Read-Host "何かキーを押して終了します"
        exit 1
    }

    Write-Host "Python を検出しました: $($pythonCmd.Command) $($pythonCmd.Args -join ' ')" -ForegroundColor Green
    Write-Host "仮想環境を作成中..." -ForegroundColor Yellow
    & $pythonCmd.Command @($pythonCmd.Args + "-m" + "venv" + "venv")
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ エラー: 仮想環境の作成に失敗しました" -ForegroundColor Red
        Read-Host "何かキーを押して終了します"
        exit 1
    }

    Write-Host "依存ライブラリをインストール中..." -ForegroundColor Yellow
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ エラー: 依存ライブラリのインストールに失敗しました" -ForegroundColor Red
        Read-Host "何かキーを押して終了します"
        exit 1
    }
}

if (Test-Path $venv_activate) {
    Write-Host "✓ 仮想環境を検出しました" -ForegroundColor Green
    Write-Host "  仮想環境をアクティベート中..." -ForegroundColor Yellow

    $originalPolicy = $null
    try {
        $originalPolicy = Get-ExecutionPolicy
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
        & $venv_activate

        Write-Host ""
        Write-Host "✓ 仮想環境がアクティベートされました" -ForegroundColor Green
        Write-Host ""
        Write-Host "アプリケーションを起動しています..." -ForegroundColor Yellow
        Write-Host ""

        Write-Host "使用Python: " -NoNewline -ForegroundColor Cyan
        & $venvPython -c "import sys; print(sys.executable)"

        $hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
        if ($hasNvidia) { $env:HAS_NVIDIA = "1" } else { $env:HAS_NVIDIA = "0" }

        & $venvPython "preflight_torch.py"
        $needTorchSetup = ($LASTEXITCODE -ne 0)
        if ($needTorchSetup) {
            Write-Host ""
            Write-Host "Torch/CUDA の環境が未設定または非対応です。AI除去(Demucs)用の依存関係をセットアップします..." -ForegroundColor Yellow
            powershell -NoProfile -ExecutionPolicy Bypass -File "setup_torch_fix.ps1" -NoPause
            & $venvPython -m pip install julius demucs
        }

        & $venvPython main.py

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "アプリケーションがエラーで終了しました (終了コード: $LASTEXITCODE)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host ""
        Write-Host "❌ エラー: 起動処理に失敗しました" -ForegroundColor Red
        Write-Host "  エラー詳細: $_" -ForegroundColor Red
        Write-Host ""
        Read-Host "何かキーを押して終了します"
        exit 1
    }
    finally {
        if ($null -ne $originalPolicy) {
            try {
                Set-ExecutionPolicy -ExecutionPolicy $originalPolicy -Scope Process -Force
            } catch {
            }
        }
    }
} else {
    Write-Host ""
    Write-Host "❌ エラー: 仮想環境 (venv) が見つかりません" -ForegroundColor Red
    Write-Host ""
    Read-Host "何かキーを押して終了します"
    exit 1
}

Write-Host ""
Write-Host "👋 アプリケーションを終了しました" -ForegroundColor Green
Write-Host ""
