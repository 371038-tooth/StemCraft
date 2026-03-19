@echo off
chcp 65001 >nul
setlocal

REM StemCraft 起動スクリプト
cd /d "%~dp0"

set "BASE=%~dp0"
set "ROOT=%BASE%app\"
set "APP_MAIN=%ROOT%main.py"
set "PYTHON_INSTALLER=%ROOT%installers\python-3.11.9-amd64.exe"
set "SYSTEM_PYTHON="
set "VENV_PY=%ROOT%venv\Scripts\python.exe"

REM 旧構成の残骸があれば可能な範囲で削除（ロック中は無視）
if exist "%BASE%venv\Scripts\python.exe" (
    del /f /q "%BASE%venv\Scripts\python.exe" >nul 2>&1
    rmdir /s /q "%BASE%venv" >nul 2>&1
)
if exist "%BASE%ffmpeg\bin\ffmpeg.exe" (
    del /f /q "%BASE%ffmpeg\bin\ffmpeg.exe" >nul 2>&1
    rmdir /s /q "%BASE%ffmpeg" >nul 2>&1
)

if not exist "%APP_MAIN%" (
    echo.
    echo エラー: app フォルダが見つかりません
    echo 想定パス: %APP_MAIN%
    pause
    exit /b 1
)

cd /d "%ROOT%"

echo.
echo ============================================================
echo             StemCraft - ボーカル分離プレイヤー
echo ============================================================
echo.

if not exist "%VENV_PY%" goto PREPARE_PYTHON

goto ACTIVATE_VENV

:PREPARE_PYTHON
call :RESOLVE_SYSTEM_PYTHON

if not defined SYSTEM_PYTHON (
    call :INSTALL_PYTHON
    if errorlevel 1 exit /b 1
    call :RESOLVE_SYSTEM_PYTHON
)

if not defined SYSTEM_PYTHON (
    echo.
    echo エラー: Python の検出に失敗しました
    echo 同梱インストーラの実行後も Python が見つかりませんでした
    pause
    exit /b 1
)

echo Python を検出しました: %SYSTEM_PYTHON%
echo.
echo 仮想環境を作成中...
%SYSTEM_PYTHON% -m venv venv

if errorlevel 1 (
    echo.
    echo エラー: 仮想環境の作成に失敗しました
    pause
    exit /b 1
)

echo 仮想環境を作成しました
echo.
echo 依存ライブラリをインストール中...
echo （初回は3～10分かかります...）
echo.
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo エラー: 仮想環境のアクティベーションに失敗しました
    pause
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo エラー: ライブラリのインストールに失敗しました
    pause
    exit /b 1
)

:ACTIVATE_VENV
echo 仮想環境を検出しました
echo 仮想環境をアクティベート中...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo.
    echo エラー: 仮想環境のアクティベーションに失敗しました
    pause
    exit /b 1
)

echo 仮想環境がアクティベートされました
echo.

REM FFmpeg チェック・セットアップ
echo FFmpeg をチェック中...
if exist "%ROOT%ffmpeg\bin\ffmpeg.exe" (
    echo FFmpeg detected (local)
    echo.
) else (
    ffmpeg -version >nul 2>&1
    if not errorlevel 1 (
        echo FFmpeg detected (system)
        echo.
        goto RUN_APP
    )

    echo.
    echo FFmpeg がインストールされていません
    echo FFmpeg を自動ダウンロード・セットアップしています...
    echo.
    
    REM PowerShell でセットアップスクリプトを実行
    powershell -NoProfile -ExecutionPolicy Bypass -File "setup_ffmpeg_new.ps1"
    
    if errorlevel 1 (
        echo.
        echo WARNING FFmpeg setup failed
        echo Please manually download from: https://ffmpeg.org/download.html
        echo.
    ) else (
        echo FFmpeg setup completed
        echo.
    )
)

:RUN_APP
echo アプリケーションを起動しています...
echo.

REM ローカル FFmpeg を PATH に追加（末尾に追加して DLL 衝突を避ける）
set "PATH=%PATH%;%ROOT%ffmpeg\bin"

REM venv の Python を明示的に使用（PATHの食い違い対策）
set "VENV_PY=%ROOT%venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    echo エラー: venv の python.exe が見つかりません: %VENV_PY%
    echo venv を作り直してから再実行してください
    pause
    exit /b 1
)

echo 使用Python: 
"%VENV_PY%" -c "import sys; print(sys.executable)"

REM NVIDIA GPU の有無を検出
set "HAS_NVIDIA=0"
where nvidia-smi >nul 2>&1
if %errorlevel%==0 set "HAS_NVIDIA=1"

REM Torch が無い/壊れている/GPU非対応の場合は自動でセットアップ（GPUがあればCUDA版を選択）
set "NEED_TORCH_SETUP=0"
"%VENV_PY%" "%ROOT%preflight_torch.py"
if not errorlevel 1 (
    set "NEED_TORCH_SETUP=0"
) else (
    set "NEED_TORCH_SETUP=1"
)

if %NEED_TORCH_SETUP%==1 (
    echo.
    echo Torch/CUDA の環境が未設定または非対応です。AI除去 Demucs 用の依存関係をセットアップします...
    powershell -NoProfile -ExecutionPolicy Bypass -File "setup_torch_fix.ps1" -NoPause
    "%VENV_PY%" -m pip install julius demucs
)

"%VENV_PY%" "%ROOT%main.py"
goto END

:RESOLVE_SYSTEM_PYTHON
set "SYSTEM_PYTHON="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "SYSTEM_PYTHON=py -3.11"
        goto :eof
    )

    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "SYSTEM_PYTHON=py -3"
        goto :eof
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "SYSTEM_PYTHON=python"
        goto :eof
    )
)

for %%P in (
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles(x86)%\Python311\python.exe"
) do (
    if exist "%%~P" (
        set "SYSTEM_PYTHON=\"%%~P\""
        goto :eof
    )
)

goto :eof

:INSTALL_PYTHON
echo Python が見つかりませんでした
echo 同梱の Python インストーラを実行します...
echo.

if not exist "%PYTHON_INSTALLER%" (
    echo エラー: 同梱インストーラが見つかりません
    echo 想定パス: %PYTHON_INSTALLER%
    pause
    exit /b 1
)

start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0 Include_launcher=1 Shortcuts=0

if errorlevel 1 (
    echo.
    echo エラー: Python のインストールに失敗しました
    pause
    exit /b 1
)

set "PATH=%LocalAppData%\Programs\Python\Python311;%LocalAppData%\Programs\Python\Python311\Scripts;%PATH%"
echo Python のインストールが完了しました
echo.
goto :eof

:END
echo.
echo アプリケーションを終了しました
echo.
pause
