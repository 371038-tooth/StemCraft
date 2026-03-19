#!/bin/bash
# StemCraft 起動スクリプト (Linux/Mac用)

# 仮想環境をアクティベート
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: 仮想環境 (venv) が見つかりません"
    echo "以下のコマンドで仮想環境を作成してください:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# メインアプリケーションを実行
python main.py
