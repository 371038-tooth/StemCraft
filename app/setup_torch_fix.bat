@echo off
echo 🔧 Torch 環境設定を開始します...
echo.

REM venv があればそちらを優先
set "ROOT=%~dp0"
set "PYTHON=python"
set "PIP=pip"
if exist "%ROOT%venv\Scripts\python.exe" (
	set "PYTHON=%ROOT%venv\Scripts\python.exe"
	set "PIP=%ROOT%venv\Scripts\pip.exe"
	echo ✓ venv を検出しました。venv に Torch をインストールします
) else (
	echo ⚠ venv が見つかりません。現在の Python 環境に Torch をインストールします
)

REM NVIDIA GPU があれば cu128、無ければ cpu
set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
	set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
	echo 🖥️  NVIDIA GPU を検出しました。CUDA版 Torch をインストールします (cu128)
) else (
	echo 🖥️  NVIDIA GPU が見つかりません。CPU版 Torch をインストールします
)

echo 📦 Visual C++ Redistributable を確認中...
echo   ダウンロード中...
powershell -Command "& {Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vc_redist.x64.exe'}"

echo   インストール中...
"%TEMP%\vc_redist.x64.exe" /install /quiet /norestart

echo ✓ Visual C++ Redistributable のインストールが完了しました
echo.

echo 🔄 Torch を再インストール中...
"%PIP%" uninstall torch torchvision torchaudio -y -q
"%PIP%" install torch torchvision torchaudio --index-url %PYTORCH_INDEX_URL% -q

echo ✓ Torch の再インストールが完了しました
echo.

echo 🧪 Torch の動作確認中...
"%PYTHON%" -c "import torch; print('Torch ' + torch.__version__ + ' が正常に動作しています'); print('CUDA 利用可能: ' + str(torch.cuda.is_available()))"

echo.
echo 🎉 設定完了！
echo   アプリケーションを再起動してください
echo.

pause