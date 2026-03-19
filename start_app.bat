@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Fixing run.bat line endings (LF -^> CRLF)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='run.bat'; if(Test-Path $p){ $lines=[IO.File]::ReadAllLines($p); $enc=New-Object System.Text.UTF8Encoding $false; [IO.File]::WriteAllLines($p, $lines, $enc) }"

echo [2/2] Starting StemCraft...
call run.bat
