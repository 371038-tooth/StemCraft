"""
メインGUIアプリケーション
ボーカル除去機能付きミュージックプレイヤー
"""

import sys
from pathlib import Path
import numpy as np

# Torch/Demucsを先に読み込んでDLL依存関係を確定させ、後続のQtロードによるDLL競合を避ける
from src.advanced_vocal_remover import get_advanced_remover

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QFileDialog, QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from src.audio_processor import AudioProcessor
from src.audio_player import AudioPlayer


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
                # result is now a dict of stems
                self.result = advanced_remover.separate_audio(self.audio_file)
            else:
                raise Exception("Demucsモデルが利用不可です。")

            self.finished.emit()
        except Exception as e:
            self.error.emit(f"音源分離エラー: {str(e)}")


class StemCraftApp(QMainWindow):
    """メインアプリケーションウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.audio_processor = AudioProcessor()
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
        
        # AI除去の初期化状態
        self.advanced_remover = get_advanced_remover()
        self.ai_available = False
        self.ai_initializing = True  # 初期化中フラグ
        
        self.init_ui()
        self.setup_timers()
        self.start_ai_initialization()  # バックグラウンドでAI初期化を開始
        
    def setup_timers(self):
        """タイマーをセットアップ"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_progress_bar)
        
    def init_ui(self):
        """UIを初期化"""
        self.setWindowTitle("StemCraft - ボーカル分離ミュージックプレイヤー")
        self.setGeometry(100, 100, 600, 500)
        
        # メインウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # レイアウト
        main_layout = QVBoxLayout()
        
        # ファイルが開かれているかの表示
        file_info_layout = QHBoxLayout()
        self.file_label = QLabel("ファイル: 未読込")
        self.file_label.setFont(QFont("Arial", 10))
        file_info_layout.addWidget(self.file_label)
        main_layout.addLayout(file_info_layout)
        
        # ファイル開くボタン
        open_file_layout = QHBoxLayout()
        self.open_btn = QPushButton("🎵 音声ファイルを開く")
        self.open_btn.setFont(QFont("Arial", 11))
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
        
        self.play_btn = QPushButton("▶ 再生")
        self.play_btn.setFont(QFont("Arial", 10))
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ 一時停止")
        self.pause_btn.setFont(QFont("Arial", 10))
        self.pause_btn.clicked.connect(self.pause_audio)
        self.pause_btn.setEnabled(False)
        playback_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setFont(QFont("Arial", 10))
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        playback_layout.addWidget(self.stop_btn)
        
        # 音量スライダー
        playback_layout.addSpacing(20)
        volume_label = QLabel("🔊 音量:")
        playback_layout.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        playback_layout.addWidget(self.volume_slider)
        
        main_layout.addLayout(playback_layout)
        
        # 音源分離（ステム）コントロールエリア
        self.stems_widget = QWidget()
        self.stems_layout = QVBoxLayout()
        self.stems_widget.setLayout(self.stems_layout)
        self.stems_widget.setVisible(False)
        main_layout.addWidget(self.stems_widget)
        
        # AI除去チェックボックス（2つのモード）
        ai_removal_layout = QHBoxLayout()
        
        self.ai_4stem_check = QCheckBox("🤖 4パート分離 (標準)")
        self.ai_4stem_check.setFont(QFont("Arial", 10))
        self.ai_4stem_check.setEnabled(self.ai_available)
        self.ai_4stem_check.clicked.connect(self.on_4stem_clicked)
        ai_removal_layout.addWidget(self.ai_4stem_check)
        
        self.ai_6stem_check = QCheckBox("🤖 6パート分離 (Piano/Guitar)")
        self.ai_6stem_check.setFont(QFont("Arial", 10))
        self.ai_6stem_check.setEnabled(self.ai_available)
        self.ai_6stem_check.clicked.connect(self.on_6stem_clicked)
        ai_removal_layout.addWidget(self.ai_6stem_check)
        
        ai_removal_layout.addStretch()
        
        # AI初期化状態表示ラベル
        self.ai_status_label = QLabel("（初回は初期化中...）")
        self.ai_status_label.setFont(QFont("Arial", 8))
        ai_removal_layout.addWidget(self.ai_status_label)
        main_layout.addLayout(ai_removal_layout)
        
        # AI初期化プログレスバー
        self.ai_init_progress = QProgressBar()
        self.ai_init_progress.setMaximum(0)  # 不確定モード
        self.ai_init_progress.setMinimum(0)
        self.ai_init_progress.setStyleSheet(
            "QProgressBar { height: 10px; border-radius: 5px; background-color: #f0f0f0; }"
            "QProgressBar::chunk { background-color: #4CAF50; }"
        )
        self.ai_init_progress.setVisible(self.ai_initializing)
        main_layout.addWidget(self.ai_init_progress)
        
        # 処理進捗表示
        self.progress_status = QLabel("")
        self.progress_status.setFont(QFont("Arial", 9))
        main_layout.addWidget(self.progress_status)
        
        # 保存ボタン
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 オフボーカル保存")
        self.save_btn.setFont(QFont("Arial", 11))
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
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
            if self.audio_processor.load_audio(file_path):
                self.current_file = file_path
                self.file_label.setText(f"ファイル: {Path(file_path).name}")
                
                # プレイヤーに音声をセット
                audio_data = self.audio_processor.get_audio_data()
                self.audio_player.set_audio(audio_data, self.audio_processor.sr_loaded)
                
                # シークバー設定
                duration = self.audio_processor.get_duration()
                self.duration_label.setText(self.format_time(duration))
                
                # 前の分離状態をリセット
                if self.audio_player.is_playing:
                    self.stop_audio()
                self.stems = {}
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
            else:
                self.file_label.setText("ファイル: 読込失敗")
    
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
                self.audio_player.set_audio(
                    self.audio_processor.get_audio_data(),
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
        
        # スレッドでAI除去処理を実行
        if self.vocal_removal_worker is not None:
            self.vocal_removal_worker.deleteLater()

        self.vocal_removal_worker = VocalRemovalWorker(self.current_file, model_name)
        self.ai_4stem_check.setEnabled(False)
        self.ai_6stem_check.setEnabled(False)
        self.vocal_removal_worker.finished.connect(self.on_vocal_removal_finished)
        self.vocal_removal_worker.error.connect(self.on_vocal_removal_error)
        self.vocal_removal_worker.progress.connect(self.on_vocal_removal_progress)
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
        self.stems_widget.setVisible(False)
        
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
        # ワーカーが意図しないうちに停止した場合は何もしない
        if not self.use_ai_removal:
            return

        if self.vocal_removal_worker is None or self.vocal_removal_worker.result is None:
            return

        self.stems = self.vocal_removal_worker.result
        self.ai_4stem_check.setEnabled(self.ai_available)
        self.ai_6stem_check.setEnabled(self.ai_available)
        self.progress_status.setText("✓ 音源分離完了")
        
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
        title.setFont(QFont("Arial", 10, QFont.Bold))
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
            mute_btn = QPushButton("Mute")
            mute_btn.setCheckable(True)
            mute_btn.setFixedWidth(50)
            # Checked状態（ミュートON）を赤くする
            mute_btn.setStyleSheet("""
                QPushButton { background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; }
                QPushButton:checked { background-color: #ff6b6b; color: white; border: 1px solid #d63031; }
            """)
            mute_btn.toggled.connect(lambda checked, n=name: self.on_stem_mute_toggled(n, checked))
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
                save_data = original_audio
            
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
