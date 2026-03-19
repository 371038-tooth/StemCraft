#!/bin/bash

echo "🔧 Torch 環境設定を開始します..."
echo ""

# venv があればそちらを優先
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"
PIP_CMD="pip"

if [ -x "$ROOT_DIR/venv/bin/python" ]; then
	PYTHON_CMD="$ROOT_DIR/venv/bin/python"
	PIP_CMD="$ROOT_DIR/venv/bin/pip"
	echo "✓ venv を検出しました。venv に Torch をインストールします"
elif [ -x "$ROOT_DIR/venv/Scripts/python.exe" ]; then
	# Git Bash / MSYS 環境で Windows venv を使っているケース
	PYTHON_CMD="$ROOT_DIR/venv/Scripts/python.exe"
	PIP_CMD="$ROOT_DIR/venv/Scripts/pip.exe"
	echo "✓ venv を検出しました。venv に Torch をインストールします"
else
	echo "⚠ venv が見つかりません。現在の Python 環境に Torch をインストールします"
fi

# NVIDIA GPU があれば cu128、無ければ cpu
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
	PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
	echo "🖥️  NVIDIA GPU を検出しました。CUDA版 Torch をインストールします (cu128)"
else
	echo "🖥️  NVIDIA GPU が見つかりません。CPU版 Torch をインストールします"
fi

echo "🔄 Torch を再インストール中..."

# 既存の Torch をアンインストール
"$PIP_CMD" uninstall torch torchvision torchaudio -y -q

# CPU 版 Torch をインストール
"$PIP_CMD" install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL" -q

echo "✓ Torch の再インストールが完了しました"
echo ""

echo "🧪 Torch の動作確認中..."

# テスト
"$PYTHON_CMD" -c "import torch; print('Torch ' + torch.__version__ + ' が正常に動作しています'); print('CUDA 利用可能: ' + str(torch.cuda.is_available()))"

echo ""
echo "🎉 設定完了！"
echo "  アプリケーションを再起動してください"
echo ""

read -p "Enter キーを押して終了します"