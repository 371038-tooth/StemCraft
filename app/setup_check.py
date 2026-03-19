"""
セットアップガイドと環境確認スクリプト
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Pythonバージョンを確認"""
    print("=" * 60)
    print("📋 Python バージョン確認")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8以上が必要です")
        return False
    
    print("✓ Pythonバージョン OK")
    return True


def check_dependencies():
    """依存ライブラリをチェック"""
    print("\n" + "=" * 60)
    print("📦 依存ライブラリ確認")
    print("=" * 60)
    
    required_packages = [
        'PyQt5',
        'librosa',
        'numpy',
        'scipy',
        'soundfile',
        'sounddevice'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.lower().replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} (未インストール)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ 以下のパッケージをインストール: {', '.join(missing)}")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ 全ての依存ライブラリがインストールされています")
    return True


def check_project_structure():
    """プロジェクト構造を確認"""
    print("\n" + "=" * 60)
    print("🗂️  プロジェクト構造確認")
    print("=" * 60)
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'src/audio_processor.py',
        'src/audio_player.py',
        'src/__init__.py',
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    for file in required_files:
        file_path = base_path / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"❌ {file} (見つかりません)")
            all_exist = False
    
    if all_exist:
        print("\n✓ プロジェクト構造 OK")
    else:
        print("\n❌ 一部のファイルが見つかりません")
    
    return all_exist


def print_setup_instructions():
    """セットアップ手順を表示"""
    print("\n" + "=" * 60)
    print("🚀 セットアップ手順")
    print("=" * 60)
    
    print("""
1️⃣  仮想環境を作成:
   python -m venv venv

2️⃣  仮想環境をアクティベート:
   Windows: venv\Scripts\activate
   Linux/Mac: source venv/bin/activate

3️⃣  依存ライブラリをインストール:
   pip install -r requirements.txt

4️⃣  アプリケーションを起動:
   Windows: run.bat または python main.py
   Linux/Mac: ./run.sh または python main.py
""")


def main():
    """メイン関数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║    StemCraft - セットアップ確認スクリプト    " + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # チェック実行
    python_ok = check_python_version()
    project_ok = check_project_structure()
    
    print("\n" + "=" * 60)
    print("📌 次のステップ")
    print("=" * 60)
    
    if python_ok and project_ok:
        print("""
✓ プロジェクト準備 OK

次のコマンドで依存ライブラリをインストール:
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt

その後、以下で起動:
  python main.py
""")
    else:
        print_setup_instructions()
    
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
