# StemCraft

Windows 向けのボーカル除去ミュージックプレイヤーです。

## 機能

- 音声ファイル読込: MP3、WAV、FLAC、OGG、M4A、MP4
- AI 音源分離: Demucs による 4 パート / 6 パート分離
- パート別ミックス: vocals、drums、bass、piano、guitar、other
- 再生機能: 再生、一時停止、停止、シーク
- 保存機能: WAV、MP3、FLAC、OGG

## Windows での使い方

最も簡単な起動方法は [run.bat](run.bat) の実行です。

[run.bat](run.bat) は以下を自動で行います。

- Python の有無を確認
- Python が無ければ同梱インストーラを実行
- 仮想環境 venv を作成
- 依存ライブラリをインストール
- FFmpeg を確認し、必要ならセットアップ
- Torch / Demucs を確認して不足時にセットアップ
- アプリを起動

同梱している Python インストーラ:

- [installers/python-3.11.9-amd64.exe](installers/python-3.11.9-amd64.exe)

## PowerShell での起動

PowerShell から起動する場合は [run.ps1](run.ps1) を使えます。

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

## 手動セットアップ

Python が既に入っている環境では、手動でも起動できます。

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

AI 分離を使う場合に Torch 周りで問題が出たら、以下を実行してください。

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_torch_fix.ps1
```

## トラブルシューティング

### Python が見つからない場合

- [run.bat](run.bat) は同梱した Python インストーラを自動実行します
- インストール後も検出されない場合は、いったんウィンドウを閉じて再度 [run.bat](run.bat) を実行してください

### Torch の初期化に失敗する場合

- [setup_torch_fix.ps1](setup_torch_fix.ps1) または [setup_torch_fix.bat](setup_torch_fix.bat) を実行してください
- NVIDIA GPU がある場合は CUDA 版 Torch、無い場合は CPU 版 Torch を導入します

### MP3 / OGG 保存が動かない場合

- FFmpeg が必要です
- [run.bat](run.bat) は起動時に FFmpeg を確認し、未導入なら [setup_ffmpeg_new.ps1](setup_ffmpeg_new.ps1) を実行します

## ファイル構成

```text
StemCraft/
├── installers/
│   └── python-3.11.9-amd64.exe
├── main.py
├── run.bat
├── run.ps1
├── run.sh
├── requirements.txt
├── requirements_extended.txt
├── setup_ffmpeg_new.ps1
├── setup_torch_fix.bat
├── setup_torch_fix.ps1
├── setup_torch_fix.sh
└── src/
```

## 補足

- exe 配布関連のファイルは削除しています
- Windows では [run.bat](run.bat) を正式な起動手段として想定しています
