"""
メインGUIアプリケーション
ボーカル除去機能付きミュージックプレイヤー
"""

import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Torch/Demucsを先に読み込んでDLL依存関係を確定させ、後続のQtロードによるDLL競合を避ける
from src.advanced_vocal_remover import get_advanced_remover

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QFileDialog, QProgressBar,
    QGroupBox, QSpinBox, QDialog, QLineEdit, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QColor

from src.audio_processor import AudioProcessor
from src.audio_player import AudioPlayer

_CONFIG_PATH = Path(__file__).parent / 'app_config.json'


def _load_config():
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def _save_config(config):
    _CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


class ModelInitializationWorker(QThread):
    """AI模型初期化専用スレッド"""
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, advanced_remover):
        super().__init__()
        self.advanced_remover = advanced_remover
    
    def run(self):
        try:
            self.progress.emit("🤖 Demucsモデルを初期化中...")
            if self.advanced_remover.initialize_model():
                self.progress.emit("✓ Demucsモデルの初期化が完了しました")
                self.finished.emit()
            else:
                self.error.emit("Demucsモデルの初期化に失敗しました")
        except Exception as e:
            self.error.emit(f"モデル初期化エラー: {str(e)}")


class VocalRemovalWorker(QThread):
    """音源分離処理専用スレッド（AIのみ）"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    
    def __init__(self, audio_file, model_name='htdemucs'):
        super().__init__()
        self.audio_file = audio_file
        self.model_name = model_name
        self.result = None
    
    def run(self):
        try:
            # AI除去を使用
            advanced_remover = get_advanced_remover()
            if advanced_remover is None:
                raise Exception("Demucsモデルが利用不可です。")
            
            # モデルのロード/切り替え
            self.progress.emit(f"🤖 モデル初期化中... ({self.model_name})")
            if not advanced_remover.initialize_model(self.model_name):
                raise Exception("モデルの初期化に失敗しました")

            if advanced_remover and advanced_remover.model is not None:
                self.progress.emit("🎤 AIモデルで音源分離中...")
                
                def progress_callback(current, total):
                    if total > 0:
                        pct = int((current / total) * 100)
                        self.progress_percent.emit(pct)
                
                # result is now a dict of stems
                self.result = advanced_remover.separate_audio(
                    self.audio_file, 
                    progress_callback=progress_callback
                )
            else:
                raise Exception("Demucsモデルが利用不可です。")

            self.finished.emit()
        except Exception as e:
            self.error.emit(f"音源分離エラー: {str(e)}")


class AutoDetectWorker(QThread):
    """BPM・キー自動検出ワーカー"""
    finished = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, audio_processor):
        super().__init__()
        self.audio_processor = audio_processor

    def run(self):
        try:
            y = self.audio_processor.get_audio_data()
            sr = self.audio_processor.sr_loaded
            if y is None or sr is None:
                self.error.emit("音声データがありません")
                return

            # モノラルに変換して検出
            mono = y[:, 0] if y.ndim == 2 else y

            bpm = self.audio_processor.detect_tempo(mono, sr)
            key = self.audio_processor.detect_key(mono, sr)

            self.audio_processor.detected_bpm = bpm
            self.audio_processor.detected_key = key

            self.finished.emit(bpm, key)
        except Exception as e:
            self.error.emit(f"自動検出エラー: {str(e)}")


class PitchTempoWorker(QThread):
    """ピッチ・テンポ変換ワーカー"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)

    def __init__(self, audio_processor, stems=None, n_steps=0, rate=1.0):
        """
        Args:
            audio_processor: AudioProcessor インスタンス
            stems: dict | None  ステム分離済み音声
            n_steps: ピッチシフト量（半音単位、0 で変換なし）
            rate: タイムストレッチ倍率（1.0 で変換なし）
        """
        super().__init__()
        self.audio_processor = audio_processor
        self.stems = stems
        self.n_steps = n_steps
        self.rate = rate

    def run(self):
        try:
            sr = self.audio_processor.sr_loaded
            if sr is None:
                self.error.emit("音声データがありません")
                return

            msg_parts = []
            if self.n_steps != 0:
                msg_parts.append(f"ピッチ {self.n_steps:+d} 半音")
            if abs(self.rate - 1.0) > 0.001:
                msg_parts.append(f"テンポ ×{self.rate:.3f}")
            self._base_msg = f"🎵 変換中 ({', '.join(msg_parts) or '変更なし'})"
            self.progress.emit(f"{self._base_msg}...")

            n_steps, rate = self.n_steps, self.rate

            def transform(audio, progress_callback=None):
                return self.audio_processor.apply_pitch_and_tempo(
                    audio, n_steps, rate, sr, progress_callback
                )

            self.finished.emit(self._apply_to_all(transform))
        except Exception as e:
            self.error.emit(f"変換エラー: {str(e)}")

    def _apply_to_all(self, func):
        """stems があれば各ステムに並列処理 + 細粒度進捗、なければ元音声に func を適用"""
        base_msg = getattr(self, '_base_msg', '🎵 変換中')
        if self.stems:
            stem_names = list(self.stems.keys())
            n_stems = len(stem_names)
            # 各ステムの進捗(0-100)を保持。CPython GIL により dict への代入はスレッドセーフ
            stem_prog: dict[str, int] = {name: 0 for name in stem_names}

            def make_cb(name: str):
                def cb(pct: int):
                    stem_prog[name] = pct
                    aggregate = sum(stem_prog.values()) // n_stems
                    self.progress_percent.emit(aggregate)
                    self.progress.emit(f"{base_msg} ({name} 変換中 {pct}%)...")
                return cb

            results = {}
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(func, audio, make_cb(name)): name
                    for name, audio in self.stems.items()
                }
                for future in as_completed(futures):
                    name = futures[future]
                    results[name] = future.result()
            return results
        else:
            def cb(pct: int):
                self.progress_percent.emit(pct)
                if pct < 100:
                    self.progress.emit(f"{base_msg} ({pct}% 処理中)...")
            return func(self.audio_processor.get_audio_data(), progress_callback=cb)


class FfmpegSettingsDialog(QDialog):
    def __init__(self, current_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定 — ffmpeg パス")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "M4A / AAC ファイルの読み込みに使用する ffmpeg.exe のパスを指定してください。\n"
            "空白のままにすると PATH から自動検索します。"
        ))

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(current_path or "")
        self.path_edit.setPlaceholderText("例: C:/tools/ffmpeg/bin/ffmpeg.exe")
        path_row.addWidget(self.path_edit)
        browse_btn = QPushButton("参照...")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ffmpeg.exe を選択", "", "実行ファイル (*.exe);;全てのファイル (*.*)"
        )
        if path:
            self.path_edit.setText(path)

    def get_path(self):
        return self.path_edit.text().strip()


class StemCraftApp(QMainWindow):
    """メインアプリケーションウィンドウ"""

    def __init__(self):
        super().__init__()
        self._config = _load_config()
        ffmpeg_path = self._config.get('ffmpeg_path') or None
        self.audio_processor = AudioProcessor(ffmpeg_path=ffmpeg_path)
        self.audio_player = AudioPlayer()
        self.current_file = None
        self.is_vocal_removed = False
        self.vocal_removal_worker = None
        self.model_init_worker = None
        self.is_seeking = False
        self.use_ai_removal = False
        
        # AI分離データの管理
        self.stems = {}
        self.stem_sliders = {}
        self._converted_audio = None  # 変換済み音声データ（ピッチ/テンポ変換後）
        self._current_pitch_steps: int = 0  # 現在適用中のピッチシフト量

        # ピッチ・テンポ変換ワーカー
        self.auto_detect_worker = None
        self.pitch_tempo_worker = None
        self._original_stems: dict = {}  # 音源分離直後の元ステムを保持

        # AI除去の初期化状態
        self.advanced_remover = get_advanced_remover()
        self.ai_available = False
        self.ai_initializing = True  # 初期化中フラグ
        
        self.init_ui()
        self.setup_timers()
        self.start_ai_initialization()  # バックグラウンドでAI初期化を開始
        self.setup_theme()
        
    def setup_theme(self):
        """アプリ全体にモダンダークテーマ（QSS）を適用"""
        style = """
        QMainWindow {
            background-color: #12141A;
        }
        QWidget {
            color: #E0E0E0;
            font-family: "Yu Gothic UI", "Meiryo", sans-serif;
            font-size: 17px;
        }
        /* パネル(カード)のような見た目を作る */
        QGroupBox, #stems_widget {
            background-color: #1E2028;
            border-radius: 12px;
            border: 1px solid #2A2C35;
            margin-top: 2ex;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
            color: #90A0B0;
            font-weight: bold;
        }
        /* 一般的なボタン */
        QPushButton {
            background-color: transparent;
            border: 1px solid #3B82F6;
            border-radius: 10px;
            color: #3B82F6;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: rgba(59, 130, 246, 0.1);
        }
        QPushButton:pressed {
            background-color: rgba(59, 130, 246, 0.2);
        }
        QPushButton:disabled {
            border: 1px solid #4A505F;
            color: #4A505F;
        }
        /* 音声ファイルを開くボタン・保存ボタンを少し大きめに */
        QPushButton#primaryButton {
            border: 2px solid #00D2FF;
            color: #00D2FF;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 14px;
        }
        QPushButton#primaryButton:hover {
            background-color: rgba(0, 210, 255, 0.1);
        }
        /* スライダー */
        QSlider::groove:horizontal {
            border: none;
            height: 6px;
            background: #2A2C35;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3B82F6, stop:1 #00D2FF);
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #D9E2EC;
            border: none;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #FFFFFF;
            box-shadow: 0 0 5px #00D2FF;
        }
        /* トグルボタン風 (Mute) */
        QPushButton#toggleButton {
            background-color: #3B82F6;
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 4px;
        }
        QPushButton#toggleButton:checked {
            background-color: #333845;
            color: #90A0B0;
        }
        /* スピンボックス（数値入力） */
        QSpinBox {
            background-color: #2A2C35;
            border: 1px solid #3B82F6;
            border-radius: 6px;
            color: #FFFFFF;
            padding: 2px;
        }
        QSpinBox::up-button, QSpinBox::down-button {
            width: 14px;
            background: transparent;
        }
        QSpinBox:disabled {
            border: 1px solid #4A505F;
            color: #4A505F;
        }
        /* チェックボックス（AIモデルなど） */
        QCheckBox {
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 9px;
            border: 2px solid #3B82F6;
        }
        QCheckBox::indicator:unchecked {
            background-color: #12141A;
        }
        QCheckBox::indicator:checked {
            background-color: #00D2FF;
        }
        """
        self.setStyleSheet(style)
        
    def _white_icon(self, standard_icon, size=22):
        """システム標準アイコンをピクセルごと白色に変換して返す"""
        pixmap = self.style().standardIcon(standard_icon).pixmap(QSize(size, size))
        white = QPixmap(pixmap.size())
        white.fill(Qt.transparent)
        painter = QPainter(white)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(white.rect(), QColor(255, 255, 255))
        painter.end()
        return QIcon(white)

    def setup_timers(self):
        """タイマーをセットアップ"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_progress_bar)
        
    def init_ui(self):
        """UIを初期化"""
        self.setWindowTitle("StemCraft Modern Pro Player")
        self.setGeometry(100, 100, 800, 750)
        self.setMaximumHeight(950)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # レイアウト（余白・間隔を調整）
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 12, 12, 12)
        
        # ファイルが開かれているかの表示
        file_info_layout = QHBoxLayout()
        self.file_label = QLabel("ファイル: 未読込")
        self.file_label.setFont(QFont("Arial", 9))
        file_info_layout.addWidget(self.file_label)
        main_layout.addLayout(file_info_layout)
        
        # ファイル開くボタン
        open_file_layout = QHBoxLayout()
        self.open_btn = QPushButton("音声ファイルを開く")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.open_btn.clicked.connect(self.open_file)
        open_file_layout.addWidget(self.open_btn)
        main_layout.addLayout(open_file_layout)
        
        # 再生時間表示
        time_layout = QHBoxLayout()
        self.time_label = QLabel("0:00")
        self.duration_label = QLabel("0:00")
        time_layout.addWidget(self.time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.duration_label)
        main_layout.addLayout(time_layout)
        
        # プログレスバー（シークバー）
        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1000)
        self.progress_bar.sliderMoved.connect(self.on_seek)
        self.progress_bar.sliderPressed.connect(self.on_slider_pressed)
        self.progress_bar.sliderReleased.connect(self.on_slider_released)
        main_layout.addWidget(self.progress_bar)
        
        # 再生コントロール
        playback_layout = QHBoxLayout()
        
        # 左側に余白を入れてボタンを中央寄りにする
        playback_layout.addStretch()
        
        # ボタンエリア（システムの標準メディアアイコンを白色化して使用）
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        _icon_size = QSize(22, 22)
        _play_pause_style = """
            QPushButton {
                border: 2px solid #00D2FF;
                border-radius: 10px;
                padding: 5px 16px;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 210, 255, 0.15);
                border-color: #33DFFF;
            }
            QPushButton:disabled {
                border: 2px solid #3A3F4B;
            }
        """
        _stop_style = """
            QPushButton {
                border: 2px solid #00D2FF;
                border-radius: 10px;
                padding: 5px 16px;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 0.2);
                border-color: #EF4444;
            }
            QPushButton:disabled {
                border: 2px solid #3A3F4B;
            }
        """
        
        self.play_btn = QPushButton()
        self.play_btn.setObjectName("playButton")
        self.play_btn.setFixedHeight(36)
        self.play_btn.setMinimumWidth(60)
        self.play_btn.setIcon(self._white_icon(self.style().SP_MediaPlay))
        self.play_btn.setIconSize(_icon_size)
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.setStyleSheet(_play_pause_style)
        buttons_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton()
        self.pause_btn.setObjectName("pauseButton")
        self.pause_btn.setFixedHeight(36)
        self.pause_btn.setMinimumWidth(60)
        self.pause_btn.setIcon(self._white_icon(self.style().SP_MediaPause))
        self.pause_btn.setIconSize(_icon_size)
        self.pause_btn.clicked.connect(self.pause_audio)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.setStyleSheet(_play_pause_style)
        buttons_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton()
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setMinimumWidth(60)
        self.stop_btn.setIcon(self._white_icon(self.style().SP_MediaStop))
        self.stop_btn.setIconSize(_icon_size)
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet(_stop_style)
        buttons_layout.addWidget(self.stop_btn)
        
        playback_layout.addLayout(buttons_layout)
        
        # ボタンと音量の間の余白
        playback_layout.addStretch()
        
        # 音量スライダーエリア (右寄せになる)
        volume_layout = QHBoxLayout()
        volume_label = QLabel("🔊")
        volume_label.setFont(QFont("Arial", 12))
        volume_layout.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(120)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.volume_slider)
        
        playback_layout.addLayout(volume_layout)
        
        main_layout.addLayout(playback_layout)
        
        # 音源分離（ステム）コントロールエリア
        self.stems_widget = QWidget()
        self.stems_widget.setObjectName("stems_widget")
        self.stems_layout = QVBoxLayout()
        self.stems_layout.setSpacing(4)
        self.stems_layout.setContentsMargins(8, 8, 8, 8)
        self.stems_widget.setLayout(self.stems_layout)
        self.stems_widget.setVisible(False)
        main_layout.addWidget(self.stems_widget)
        
        # AI除去チェックボックス（2つのモード）
        ai_group = QGroupBox("AI分離モデル選択")
        ai_group.setFont(QFont("Arial", 10, QFont.Bold))
        ai_layout = QVBoxLayout()
        ai_layout.setSpacing(6)
        ai_layout.setContentsMargins(12, 24, 12, 10)
        
        self.ai_4stem_check = QCheckBox("4パート分離 (標準)")
        self.ai_4stem_check.setFont(QFont("Arial", 9))
        self.ai_4stem_check.setEnabled(self.ai_available)
        self.ai_4stem_check.clicked.connect(self.on_4stem_clicked)
        
        self.ai_6stem_check = QCheckBox("6パート分離 (Piano/Guitar)")
        self.ai_6stem_check.setFont(QFont("Arial", 9))
        self.ai_6stem_check.setEnabled(self.ai_available)
        self.ai_6stem_check.clicked.connect(self.on_6stem_clicked)
        
        # AI初期化状態表示ラベル
        self.ai_status_label = QLabel("（初回は初期化中...）")
        self.ai_status_label.setFont(QFont("Arial", 8))
        self.ai_status_label.setStyleSheet("color: #90A0B0; font-weight: normal;")
        
        row1 = QHBoxLayout()
        row1.addWidget(self.ai_4stem_check)
        row1.addStretch()
        row1.addWidget(self.ai_status_label)
        
        row2 = QHBoxLayout()
        row2.addWidget(self.ai_6stem_check)
        row2.addStretch()
        
        ai_layout.addLayout(row1)
        ai_layout.addLayout(row2)
        ai_group.setLayout(ai_layout)
        main_layout.addWidget(ai_group)
        
        # AI初期化プログレスバー
        self.ai_init_progress = QProgressBar()
        self.ai_init_progress.setMaximum(0)  # 不確定モード
        self.ai_init_progress.setMinimum(0)
        self.ai_init_progress.setStyleSheet(
            "QProgressBar { height: 10px; border-radius: 5px; background-color: #2A2C35; color: transparent; }"
            "QProgressBar::chunk { background-color: #4CAF50; border-radius: 5px; }"
        )
        self.ai_init_progress.setVisible(self.ai_initializing)
        main_layout.addWidget(self.ai_init_progress)
        
        # 処理進捗表示
        self.progress_status = QLabel("")
        self.progress_status.setFont(QFont("Arial", 8))
        main_layout.addWidget(self.progress_status)
        
        # 音源分離用プログレスバー
        self.ai_separation_progress = QProgressBar()
        self.ai_separation_progress.setRange(0, 100)
        self.ai_separation_progress.setValue(0)
        self.ai_separation_progress.setStyleSheet(
            "QProgressBar { height: 10px; border-radius: 5px; background-color: #2A2C35; color: transparent; }"
            "QProgressBar::chunk { background-color: #00D2FF; border-radius: 5px; }"
        )
        self.ai_separation_progress.setVisible(False)
        main_layout.addWidget(self.ai_separation_progress)

        # ─── テンポ・ピッチセクション ────────────────────────────────────────
        self.pitch_tempo_group = QGroupBox("テンポ・ピッチ調整")
        self.pitch_tempo_group.setFont(QFont("Arial", 10, QFont.Bold))
        self.pitch_tempo_group.setEnabled(False)
        pt_layout = QVBoxLayout()
        pt_layout.setSpacing(6)
        pt_layout.setContentsMargins(12, 24, 12, 10)

        # テンポ行
        tempo_row = QHBoxLayout()
        tempo_row.addWidget(QLabel("BPM:"))
        self.tempo_detected_label = QLabel("—")
        self.tempo_detected_label.setFixedWidth(45)
        self.tempo_detected_label.setAlignment(Qt.AlignCenter)
        self.tempo_detected_label.setStyleSheet("color: #00D2FF; font-weight: bold; font-size: 13px;")
        tempo_row.addWidget(self.tempo_detected_label)
        tempo_row.addWidget(QLabel("→"))
        self.tempo_spinbox = QSpinBox()
        self.tempo_spinbox.setRange(20, 300)
        self.tempo_spinbox.setValue(120)
        self.tempo_spinbox.setFixedWidth(60)
        tempo_row.addWidget(self.tempo_spinbox)
        self.apply_tempo_btn = QPushButton("BPM適用")
        self.apply_tempo_btn.clicked.connect(self.apply_tempo)
        tempo_row.addWidget(self.apply_tempo_btn)
        tempo_row.addStretch()
        pt_layout.addLayout(tempo_row)

        # ピッチ行
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("ピッチ:"))
        self.pitch_current_label = QLabel("0")
        self.pitch_current_label.setFixedWidth(45)
        self.pitch_current_label.setAlignment(Qt.AlignCenter)
        self.pitch_current_label.setStyleSheet("color: #00D2FF; font-weight: bold; font-size: 13px;")
        pitch_row.addWidget(self.pitch_current_label)
        pitch_row.addWidget(QLabel("→"))
        self.pitch_spinbox = QSpinBox()
        self.pitch_spinbox.setRange(-12, 12)
        self.pitch_spinbox.setValue(0)
        self.pitch_spinbox.setFixedWidth(60)
        self.pitch_spinbox.valueChanged.connect(self.on_pitch_value_changed)
        pitch_row.addWidget(self.pitch_spinbox)
        self.apply_pitch_btn = QPushButton("ピッチ適用")
        self.apply_pitch_btn.clicked.connect(self.apply_pitch)
        pitch_row.addWidget(self.apply_pitch_btn)
        pitch_row.addStretch()
        pt_layout.addLayout(pitch_row)
        
        # キー行
        key_row = QHBoxLayout()
        key_row.addStretch()
        key_row.addWidget(QLabel("キー:"))
        self.key_detected_label = QLabel("—")
        self.key_detected_label.setFixedWidth(100)
        key_row.addWidget(self.key_detected_label)
        key_row.addWidget(QLabel("→"))
        self.key_transposed_label = QLabel("—")
        self.key_transposed_label.setFixedWidth(100)
        key_row.addWidget(self.key_transposed_label)
        pt_layout.addLayout(key_row)

        self.pitch_tempo_group.setLayout(pt_layout)
        main_layout.addWidget(self.pitch_tempo_group)

        # ピッチ・テンポ変換プログレスバー
        self.pt_conversion_progress = QProgressBar()
        self.pt_conversion_progress.setRange(0, 100)
        self.pt_conversion_progress.setValue(0)
        self.pt_conversion_progress.setStyleSheet(
            "QProgressBar { height: 10px; border-radius: 5px; background-color: #2A2C35; color: transparent; }"
            "QProgressBar::chunk { background-color: #3B82F6; border-radius: 5px; }"
        )
        self.pt_conversion_progress.setVisible(False)
        main_layout.addWidget(self.pt_conversion_progress)
        # ─────────────────────────────────────────────────────────────────────

        # 保存ボタン / 設定ボタン
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("音声保存")
        self.save_btn.setObjectName("primaryButton")
        self.save_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        self.settings_btn = QPushButton("⚙ 設定")
        self.settings_btn.setFont(QFont("Arial", 10))
        self.settings_btn.clicked.connect(self.open_settings)
        save_layout.addWidget(self.settings_btn)
        main_layout.addLayout(save_layout)
        
        central_widget.setLayout(main_layout)
    
    def open_file(self):
        """音声ファイルを開く"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "音声ファイルを選択",
            "",
            "音声ファイル (*.mp3 *.wav *.flac *.ogg *.m4a *.mp4);;全てのファイル (*.*)"
        )
        
        if file_path:
            success, err_msg = self.audio_processor.load_audio(file_path)
            if success:
                self.current_file = file_path
                self.file_label.setText(f"ファイル: {Path(file_path).name}")
                
                # 再生中なら先に停止（set_audio 前に停止しないとデータ競合の恐れ）
                if self.audio_player.is_playing:
                    self.stop_audio()
                
                # プレイヤーに音声をセット
                audio_data = self.audio_processor.get_audio_data()
                self.audio_player.set_audio(audio_data, self.audio_processor.sr_loaded)
                
                # シークバー設定
                duration = self.audio_processor.get_duration()
                self.duration_label.setText(self.format_time(duration))
                
                # 前の分離状態をリセット
                self.stems = {}
                self._original_stems = {}
                self._converted_audio = None
                self.stems_widget.setVisible(False)
                self.is_vocal_removed = False
                self.use_ai_removal = False
                self.ai_4stem_check.setChecked(False)
                self.ai_6stem_check.setChecked(False)
                
                # UIの状態を更新
                self.play_btn.setEnabled(True)
                self.pause_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

                # ピッチ・テンポウィジェットをリセットして自動検出を開始
                self._reset_pitch_tempo_ui()
                self.start_auto_detect()
            else:
                self.file_label.setText("ファイル: 読込失敗")
                QMessageBox.critical(
                    self,
                    "読み込みエラー",
                    err_msg or "ファイルの読み込みに失敗しました。"
                )
    
    def play_audio(self):
        """音声を再生"""
        if self.audio_player.is_paused:
            # 一時停止からの再開であればセットし直さずそのまま play()
            self.audio_player.play()
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.update_timer.start(100)
            return

        if self.audio_processor.get_audio_data() is not None:
            if self.stems:
                # ステム再生モードでは、プレイヤー再設定後にUI上の音量/MUTE状態を再適用する
                self.audio_player.set_multi_track_audio(self.stems, self.audio_processor.sr_loaded)
                self.apply_stem_mix_settings()
            else:
                audio_data = self._converted_audio if self._converted_audio is not None else self.audio_processor.get_audio_data()
                self.audio_player.set_audio(
                    audio_data,
                    self.audio_processor.sr_loaded
                )
            
            self.audio_player.play()
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.update_timer.start(100)
    
    def pause_audio(self):
        """音声を一時停止"""
        self.audio_player.pause()
        self.update_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
    
    def stop_audio(self):
        """音声を停止"""
        self.audio_player.stop()
        self.update_timer.stop()
        self.progress_bar.setValue(0)
        self.time_label.setText("0:00")
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
    
    def on_4stem_clicked(self):
        """4パート分離クリック時の排他制御"""
        if self.ai_4stem_check.isChecked():
            self.ai_6stem_check.setChecked(False)
            self.start_separation('htdemucs')
        else:
            self.stop_separation()

    def on_6stem_clicked(self):
        """6パート分離クリック時の排他制御"""
        if self.ai_6stem_check.isChecked():
            self.ai_4stem_check.setChecked(False)
            self.start_separation('htdemucs_6s')
        else:
            self.stop_separation()

    def start_separation(self, model_name):
        """音源分離を開始"""
        if not self.ai_available:
            self.ai_4stem_check.setChecked(False)
            self.ai_6stem_check.setChecked(False)
            self.progress_status.setText("❌ Demucsが利用不可です")
            return
        
        # 実行中はUIスレッドをブロックしない
        if self.vocal_removal_worker and self.vocal_removal_worker.isRunning():
            self.progress_status.setText("⏳ 既に音源分離を実行中です。完了までお待ちください")
            return
        
        if not self.current_file:
            self.ai_4stem_check.setChecked(False)
            self.ai_6stem_check.setChecked(False)
            self.progress_status.setText("❌ 先にファイルを開いてください")
            return
        
        mode_text = "4パート" if model_name == 'htdemucs' else "6パート"
        self.progress_status.setText(f"🤖 AI解析中 ({mode_text})... 初回はモデルロードに時間がかかります")
        
        self.use_ai_removal = True
        
        self.ai_separation_progress.setValue(0)
        self.ai_separation_progress.setVisible(True)
        
        # スレッドでAI除去処理を実行
        if self.vocal_removal_worker is not None:
            self.vocal_removal_worker.deleteLater()

        self.vocal_removal_worker = VocalRemovalWorker(self.current_file, model_name)
        self.ai_4stem_check.setEnabled(False)
        self.ai_6stem_check.setEnabled(False)
        self.vocal_removal_worker.finished.connect(self.on_vocal_removal_finished)
        self.vocal_removal_worker.error.connect(self.on_vocal_removal_error)
        self.vocal_removal_worker.progress.connect(self.on_vocal_removal_progress)
        self.vocal_removal_worker.progress_percent.connect(self.ai_separation_progress.setValue)
        self.vocal_removal_worker.start()

    def stop_separation(self, clear_status=True):
        """分離を解除して元に戻す"""
        if self.vocal_removal_worker and self.vocal_removal_worker.isRunning():
            self.progress_status.setText("⏳ 分離処理中は解除できません。完了後に解除してください")
            if self.vocal_removal_worker.model_name == 'htdemucs':
                self.ai_4stem_check.setChecked(True)
                self.ai_6stem_check.setChecked(False)
            else:
                self.ai_4stem_check.setChecked(False)
                self.ai_6stem_check.setChecked(True)
            return

        self.is_vocal_removed = False
        self.stems = {}  # ステムをリセット
        self._original_stems = {}
        self._converted_audio = None
        self.stems_widget.setVisible(False)
        self.ai_separation_progress.setVisible(False)
        
        if clear_status:
            self.progress_status.setText("")
        self.use_ai_removal = False
        
        # 再生中の場合を停止して元に戻す
        if self.audio_player.is_playing:
            self.stop_audio()
        
        # プレイヤーを元の状態に戻す
        if self.audio_processor.get_audio_data() is not None:
            self.audio_player.set_audio(
                self.audio_processor.get_audio_data(),
                self.audio_processor.sr_loaded
            )
        
        self.ai_4stem_check.setEnabled(self.ai_available)
        self.ai_6stem_check.setEnabled(self.ai_available)

    def on_vocal_removal_finished(self):
        """AI分離完了時の処理"""
        self.ai_separation_progress.setVisible(False)
        
        # ワーカーが意図しないうちに停止した場合は何もしない
        if not self.use_ai_removal:
            return

        if self.vocal_removal_worker is None or self.vocal_removal_worker.result is None:
            return

        self.stems = self.vocal_removal_worker.result
        self._original_stems = dict(self.stems)  # 元ステムを保存
        self.ai_4stem_check.setEnabled(self.ai_available)
        self.ai_6stem_check.setEnabled(self.ai_available)
        self.progress_status.setText("✓ 音源分離完了")

        # 分離により音声が初期化されるため、ピッチ・テンポUIをリセットして再検出
        self._reset_pitch_tempo_ui()
        self._converted_audio = None
        self.start_auto_detect()
        
        # 再生中の場合、一旦停止
        was_playing = self.audio_player.is_playing
        if was_playing:
            self.stop_audio()
        
        # UI構築
        self.setup_stem_controls(self.stems)
        self.stems_widget.setVisible(True)
        
        # プレイヤーにセット
        self.audio_player.set_multi_track_audio(self.stems, self.audio_processor.sr_loaded)
        self.apply_stem_mix_settings()
        
        self.is_vocal_removed = True
        
        if was_playing:
            self.play_audio()
    
    def on_vocal_removal_error(self, error_msg):
        """AI分離エラー時"""
        self.ai_separation_progress.setVisible(False)
        self.ai_4stem_check.setEnabled(self.ai_available)
        self.ai_6stem_check.setEnabled(self.ai_available)
        self.ai_4stem_check.setChecked(False)
        self.ai_6stem_check.setChecked(False)
        self.progress_status.setText(f"❌ {error_msg}")
        self.stop_separation(clear_status=False)

    def on_vocal_removal_progress(self, message):
        """AI分離進捗表示"""
        self.progress_status.setText(message)

    def apply_stem_mix_settings(self):
        """UI上のステム音量/MUTE状態をプレイヤーへ反映"""
        for name, slider in self.stem_sliders.items():
            volume = slider.value() / 100.0
            if name in self.stem_mute_buttons and self.stem_mute_buttons[name].isChecked():
                volume = 0.0
            self.audio_player.set_track_volume(name, volume)

    def setup_stem_controls(self, stems):
        """ステムごとの音量スライダーとミュートボタンを作成"""
        # 既存のスライダーを削除
        while self.stems_layout.count():
            item = self.stems_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.stem_sliders = {}
        self.stem_mute_buttons = {}
        
        # タイトル
        title = QLabel("パート別音量調整:")
        title.setFont(QFont("Arial", 9, QFont.Bold))
        self.stems_layout.addWidget(title)
        
        # スライダー作成
        # 順序を固定したい: Vocals, Drums, Bass, Piano, Guitar, Other
        preferred_order = ['vocals', 'drums', 'bass', 'piano', 'guitar', 'other']
        remaining = [k for k in stems.keys() if k not in preferred_order]
        order = preferred_order + remaining
        
        for name in order:
            if name not in stems:
                continue
                
            row_layout = QHBoxLayout()
            
            # ラベル (先頭文字大文字)
            label_text = name.capitalize()
            label = QLabel(f"{label_text}:")
            label.setFixedWidth(60)
            row_layout.addWidget(label)
            
            # ミュートボタン
            mute_btn = QPushButton("On")
            mute_btn.setObjectName("toggleButton")
            mute_btn.setCheckable(True)
            mute_btn.setFixedWidth(50)
            
            def make_toggled_handler(btn, stem_name):
                def handler(checked):
                    btn.setText("Mute" if checked else "On")
                    self.on_stem_mute_toggled(stem_name, checked)
                return handler
                
            mute_btn.toggled.connect(make_toggled_handler(mute_btn, name))
            row_layout.addWidget(mute_btn)
            self.stem_mute_buttons[name] = mute_btn
            
            # スライダー
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(100)  # Default 100%
            
            # スライダーのコールバック
            # lambdaで変数をキャプチャする場合、デフォルト引数を使うのが定石
            slider.valueChanged.connect(lambda val, n=name: self.on_stem_volume_changed(n, val))
            
            row_layout.addWidget(slider)
            
            # 数値表示ラベル
            val_label = QLabel("100%")
            val_label.setFixedWidth(40)
            slider.valueChanged.connect(lambda val, l=val_label: l.setText(f"{val}%"))
            
            row_layout.addWidget(val_label)
            
            # コンテナに追加
            container = QWidget()
            row_layout.setContentsMargins(0, 1, 0, 1)
            container.setLayout(row_layout)
            self.stems_layout.addWidget(container)
            
            self.stem_sliders[name] = slider

    def on_stem_mute_toggled(self, name, muted):
        """各ステムのミュート切替"""
        if muted:
            # ミュート: 音量を0にする（スライダーは維持）
            self.audio_player.set_track_volume(name, 0.0)
        else:
            # ミュート解除: 現在のスライダー値に戻す
            if name in self.stem_sliders:
                val = self.stem_sliders[name].value()
                self.audio_player.set_track_volume(name, val / 100.0)

    def on_stem_volume_changed(self, name, value):
        """各ステムの音量変更"""
        # ミュート中なら内部音量は更新しない
        if hasattr(self, 'stem_mute_buttons') and name in self.stem_mute_buttons:
            if self.stem_mute_buttons[name].isChecked():
                return

        # 0-100 -> 0.0-1.0
        vol = value / 100.0
        self.audio_player.set_track_volume(name, vol)

    def on_slider_pressed(self):
        """シークバー押下"""
        self.is_seeking = True
        self.update_timer.stop()
    
    def on_slider_released(self):
        """シークバー解放"""
        self.is_seeking = False
        if self.audio_player.is_playing:
            self.update_timer.start(100)
    
    def on_seek(self, position):
        """シーク処理"""
        total_samples = self.audio_player.get_total_samples()
        if total_samples > 0:
            # 位置を計算（0-1000 → サンプル番号）
            seek_position = int((position / 1000.0) * total_samples)
            
            # シーク実行
            self.audio_player.seek(seek_position)
            
            # 時間表示更新
            sample_rate = self.audio_player.sr
            if not sample_rate:
                return
            current_time = seek_position / sample_rate
            self.time_label.setText(self.format_time(current_time))
    
    def on_volume_changed(self, value):
        """音量変更 (0-100)"""
        self.audio_player.set_volume(value / 100.0)

    def update_progress_bar(self):
        """プログレスバー更新"""
        total_samples = self.audio_player.get_total_samples()
        if total_samples > 0 and not self.is_seeking:
            progress = int((self.audio_player.current_position / total_samples * 1000.0))
            self.progress_bar.blockSignals(True)
            self.progress_bar.setValue(progress)
            self.progress_bar.blockSignals(False)
            
            # 現在時刻を更新
            sample_rate = self.audio_player.sr
            if not sample_rate:
                return
            current_time = self.audio_player.current_position / sample_rate
            self.time_label.setText(self.format_time(current_time))

        # 自然終了時はタイマーとボタン状態を復帰
        if not self.audio_player.is_currently_playing() and self.update_timer.isActive():
            self.update_timer.stop()
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
    
    @staticmethod
    def format_time(seconds):
        """秒を MM:SS 形式に変換"""
        if seconds < 0:
            seconds = 0
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"
    
    def open_settings(self):
        """ffmpeg 設定ダイアログを開く"""
        current = self._config.get('ffmpeg_path', '')
        dlg = FfmpegSettingsDialog(current, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            new_path = dlg.get_path()
            self._config['ffmpeg_path'] = new_path
            _save_config(self._config)
            self.audio_processor._ffmpeg_path = new_path or None

    def save_audio(self):
        """現在のミックス設定で音声を保存"""
        has_stems = bool(self.stems)
        original_audio = self.audio_processor.get_audio_data()

        if not has_stems and original_audio is None:
            self.progress_status.setText("❌ 先に音源分離を実行してください")
            return
        
        # デフォルトファイル名を生成
        if self.current_file:
            original_name = Path(self.current_file).stem
            suffix = "_mix" if has_stems else "_original"
            default_filename = f"{original_name}{suffix}"
        else:
            default_filename = "output"
        
        filters = "WAV ファイル (*.wav);;MP3 ファイル (*.mp3);;FLAC ファイル (*.flac);;OGG ファイル (*.ogg)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "音声を保存",
            default_filename,
            filters
        )
        
        if file_path:
            # 拡張子がない場合の補完
            if not Path(file_path).suffix:
                if "MP3" in selected_filter:
                    file_path += ".mp3"
                elif "FLAC" in selected_filter:
                    file_path += ".flac"
                elif "OGG" in selected_filter:
                    file_path += ".ogg"
                else:
                    file_path += ".wav"

            self.progress_status.setText("保存中...")
            QApplication.processEvents() # UI更新

            if has_stems:
                # ユーザーが調整した音量でミックス
                tracks = self.stems
                first_track = next(iter(tracks.values()))
                mixed = np.zeros_like(first_track)

                for name, track in tracks.items():
                    vol = self.audio_player.track_volumes.get(name, 1.0)
                    if vol > 0.001:
                        mixed += track * vol

                # クリップ処理
                save_data = np.clip(mixed, -1.0, 1.0)
            else:
                save_data = self._converted_audio if self._converted_audio is not None else original_audio
            
            if self.audio_processor.save_audio(save_data, file_path):
                self.progress_status.setText(f"✓ 保存完了: {Path(file_path).name}")
            else:
                self.progress_status.setText("❌ 保存に失敗しました")
    
    def start_ai_initialization(self):
        """AIモデルの初期化を開始"""
        if self.advanced_remover is None:
            self.on_ai_initialization_error("Demucsが初期化されていません")
            return
        
        self.model_init_worker = ModelInitializationWorker(self.advanced_remover)
        self.model_init_worker.progress.connect(self.on_ai_init_progress)
        self.model_init_worker.finished.connect(self.on_ai_initialization_finished)
        self.model_init_worker.error.connect(self.on_ai_initialization_error)
        self.model_init_worker.start()
    
    def on_ai_init_progress(self, message):
        """AI初期化プログレス表示"""
        self.progress_status.setText(message)
    
    def on_ai_initialization_finished(self):
        """AI初期化完了"""
        self.ai_initializing = False
        self.ai_available = True
        self.ai_init_progress.setVisible(False)
        self.ai_4stem_check.setEnabled(True)
        self.ai_6stem_check.setEnabled(True)
        self.ai_status_label.setText("✓ Demucs準備完了")
        self.progress_status.setText("✓ Demucsモデルの初期化が完了しました")
    
    def on_ai_initialization_error(self, error_message):
        """AI初期化エラー"""
        self.ai_initializing = False
        self.ai_available = False
        self.ai_init_progress.setVisible(False)
        self.ai_4stem_check.setEnabled(False)
        self.ai_6stem_check.setEnabled(False)
        self.ai_status_label.setText(f"❌ {error_message}")
        self.progress_status.setText(f"❌ {error_message}")

    # ──────────────────────────────────────────────────────────────────────────
    # ピッチ・テンポ 自動検出
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_pitch_tempo_ui(self):
        """ピッチ・テンポ UI をリセットして無効化する"""
        self.pitch_tempo_group.setEnabled(False)
        self.tempo_detected_label.setText("—")
        self.tempo_spinbox.setValue(120)
        self.key_detected_label.setText("—")
        self.key_transposed_label.setText("—")
        self.pitch_spinbox.setValue(0)
        self._current_pitch_steps = 0

    def start_auto_detect(self):
        """BPM・キーの自動検出を開始する（バックグラウンド）"""
        if self.auto_detect_worker is not None and self.auto_detect_worker.isRunning():
            return

        self.progress_status.setText("🔍 BPM・キーを自動検出中...")

        if self.auto_detect_worker is not None:
            self.auto_detect_worker.deleteLater()

        self.auto_detect_worker = AutoDetectWorker(self.audio_processor)
        self.auto_detect_worker.finished.connect(self.on_auto_detect_finished)
        self.auto_detect_worker.error.connect(self.on_auto_detect_error)
        self.auto_detect_worker.start()

    def on_auto_detect_finished(self, bpm: int, key: str):
        """自動検出完了 → UI 更新"""
        self.tempo_detected_label.setText(f"{bpm}")
        self.tempo_spinbox.setValue(bpm)
        self.key_detected_label.setText(key)
        self.key_transposed_label.setText(key)
        self.pitch_tempo_group.setEnabled(True)
        self.progress_status.setText(f"✓ 検出完了: {bpm} BPM / キー {key}")

    def on_auto_detect_error(self, msg: str):
        """自動検出エラー"""
        self.progress_status.setText(f"⚠ 自動検出エラー: {msg}")
        # エラーでもウィジェット自体は有効化して手動入力できるようにする
        self.pitch_tempo_group.setEnabled(True)

    # ──────────────────────────────────────────────────────────────────────────
    # ピッチ QSpinBox リアルタイム更新
    # ──────────────────────────────────────────────────────────────────────────

    def on_pitch_value_changed(self, value: int):
        """ピッチ値変更時にキー表示をリアルタイム更新（変換処理は行わない）"""
        base_key = self.audio_processor.detected_key
        if base_key:
            transposed = self.audio_processor.transpose_key(base_key, value)
            self.key_transposed_label.setText(transposed)
        else:
            self.key_transposed_label.setText("—")

    # ──────────────────────────────────────────────────────────────────────────
    # ピッチ・テンポ適用
    # ──────────────────────────────────────────────────────────────────────────

    def apply_pitch(self):
        """ピッチ変換を開始（現在のBPM設定も同時に適用）"""
        if self.pitch_tempo_worker is not None and self.pitch_tempo_worker.isRunning():
            self.progress_status.setText("⏳ 変換処理中です。完了をお待ちください")
            return

        n_steps = self.pitch_spinbox.value()
        self._current_pitch_steps = n_steps
        rate = self._current_tempo_rate()
        self._start_pitch_tempo_worker(n_steps=n_steps, rate=rate)

    def apply_tempo(self):
        """テンポ変換を開始（現在のピッチ設定も同時に適用）"""
        if self.pitch_tempo_worker is not None and self.pitch_tempo_worker.isRunning():
            self.progress_status.setText("⏳ 変換処理中です。完了をお待ちください")
            return

        rate = self._current_tempo_rate()
        self._start_pitch_tempo_worker(n_steps=self._current_pitch_steps, rate=rate)

    def _current_tempo_rate(self) -> float:
        """現在のテンポスピンボックスから変換レートを算出する"""
        original_bpm = self.audio_processor.detected_bpm
        if not original_bpm or original_bpm <= 0:
            return 1.0
        target_bpm = self.tempo_spinbox.value()
        return target_bpm / original_bpm

    def _start_pitch_tempo_worker(self, n_steps: int = 0, rate: float = 1.0):
        """PitchTempoWorker を起動する共通処理"""
        if self.audio_processor.get_audio_data() is None:
            self.progress_status.setText("❌ 先にファイルを開いてください")
            return

        if self.pitch_tempo_worker is not None:
            self.pitch_tempo_worker.deleteLater()

        self.pitch_tempo_group.setEnabled(False)
        stems = dict(self._original_stems) if self._original_stems else None
        self.pitch_tempo_worker = PitchTempoWorker(
            self.audio_processor, stems=stems, n_steps=n_steps, rate=rate
        )
        self.pitch_tempo_worker.finished.connect(self.on_pitch_tempo_finished)
        self.pitch_tempo_worker.error.connect(self.on_pitch_tempo_error)
        self.pitch_tempo_worker.progress.connect(lambda msg: self.progress_status.setText(msg))
        self.pitch_tempo_worker.progress_percent.connect(self.pt_conversion_progress.setValue)
        self.pt_conversion_progress.setRange(0, 100)
        self.pt_conversion_progress.setValue(0)
        self.pt_conversion_progress.setVisible(True)
        self.pitch_tempo_worker.start()

    def on_pitch_tempo_finished(self, result):
        """変換完了 → プレイヤーに反映"""
        self.pt_conversion_progress.setRange(0, 100)
        self.pt_conversion_progress.setVisible(False)
        self.pitch_tempo_group.setEnabled(True)
        sr = self.audio_processor.sr_loaded

        was_playing = self.audio_player.is_playing
        if was_playing:
            self.stop_audio()

        if isinstance(result, dict):
            # ステムごとの変換結果
            self.stems = result
            self._converted_audio = None
            self.audio_player.set_multi_track_audio(self.stems, sr)
            self.apply_stem_mix_settings()
        else:
            # 単一トラックの変換結果
            self._converted_audio = result
            self.audio_player.set_audio(result, sr)

        # 現在値ラベルを更新
        self.pitch_current_label.setText(str(self._current_pitch_steps))
        self.progress_status.setText("✓ 変換完了")

        if was_playing:
            self.play_audio()

    def on_pitch_tempo_error(self, msg: str):
        """変換エラー"""
        self.pt_conversion_progress.setRange(0, 100)
        self.pt_conversion_progress.setVisible(False)
        self.pitch_tempo_group.setEnabled(True)
        self.progress_status.setText(f"❌ {msg}")

    def closeEvent(self, event):
        """終了時に再生とワーカースレッドを停止"""
        self.update_timer.stop()
        self.audio_player.stop()

        if self.vocal_removal_worker is not None and self.vocal_removal_worker.isRunning():
            self.vocal_removal_worker.requestInterruption()
            self.vocal_removal_worker.quit()
            self.vocal_removal_worker.wait(5000)

        if self.model_init_worker is not None and self.model_init_worker.isRunning():
            self.model_init_worker.requestInterruption()
            self.model_init_worker.quit()
            self.model_init_worker.wait(5000)

        if self.auto_detect_worker is not None and self.auto_detect_worker.isRunning():
            self.auto_detect_worker.requestInterruption()
            self.auto_detect_worker.quit()
            self.auto_detect_worker.wait(5000)

        if self.pitch_tempo_worker is not None and self.pitch_tempo_worker.isRunning():
            self.pitch_tempo_worker.requestInterruption()
            self.pitch_tempo_worker.quit()
            self.pitch_tempo_worker.wait(5000)

        super().closeEvent(event)


def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    
    # アプリケーションスタイル設定
    app.setStyle('Fusion')
    
    window = StemCraftApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
