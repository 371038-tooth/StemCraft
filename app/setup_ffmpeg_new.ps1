param()

$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

$ffmpegDir = Join-Path $PSScriptRoot "ffmpeg"
$ffmpegBin = Join-Path $ffmpegDir "bin"
$ffmpegExe = Join-Path $ffmpegBin "ffmpeg.exe"

if (Test-Path $ffmpegExe) {
    Write-Host "[OK] FFmpeg is already available" -ForegroundColor Green
    exit 0
}

$downloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$tempZip = Join-Path $env:TEMP "ffmpeg_temp.zip"

try {
    if (-not (Test-Path $ffmpegDir)) {
        New-Item -ItemType Directory -Path $ffmpegDir | Out-Null
        Write-Host "[OK] Created ffmpeg directory" -ForegroundColor Green
    }
    
    Write-Host "[*] Downloading FFmpeg..." -ForegroundColor Yellow
    
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $downloadUrl -OutFile $tempZip -ErrorAction Stop
    
    Write-Host "[OK] Download completed" -ForegroundColor Green
    
    Write-Host "[*] Extracting FFmpeg..." -ForegroundColor Yellow
    Expand-Archive -Path $tempZip -DestinationPath $ffmpegDir -Force
    
    $extractedBin = Get-ChildItem -Path $ffmpegDir -Filter "bin" -Directory -Recurse | Select-Object -First 1
    if ($null -ne $extractedBin) {
        Copy-Item -Path $extractedBin.FullName -Destination $ffmpegBin -Container -Force -Recurse
        Get-ChildItem -Path $ffmpegDir -Directory | Where-Object { $_.Name -ne "bin" } | Remove-Item -Recurse -Force
    }
    
    Write-Host "[OK] Extraction completed" -ForegroundColor Green
    
    if (Test-Path $ffmpegExe) {
        $env:PATH = "$ffmpegBin;$env:PATH"
        Write-Host "[OK] PATH updated" -ForegroundColor Green
        Write-Host "[OK] FFmpeg setup completed" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "[ERROR] ffmpeg.exe not found" -ForegroundColor Red
        exit 1
    }
    
} catch {
    Write-Host "[ERROR] $_" -ForegroundColor Red
    exit 1
} finally {
    if (Test-Path $tempZip) {
        Remove-Item $tempZip -Force -ErrorAction SilentlyContinue
    }
}
