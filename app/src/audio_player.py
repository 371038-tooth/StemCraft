"""
オーディオプレイヤーモジュール
音声の再生を担当
"""

import numpy as np
import threading
import sounddevice as sd
import time


class AudioPlayer:
    """オーディオプレイヤークラス"""
    
    def __init__(self):
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
        self.audio_data = None
        self.tracks = {}  # {name: numpy_array}
        self.track_volumes = {}  # {name: float}
        self.sr = None
        self.stream = None
        self.play_thread = None
        self._stop_flag = False
        self.start_time = 0
        self.pause_time = 0
        self.volume = 1.0
        self._state_lock = threading.Lock()

    def set_volume(self, volume):
        """全体のマスター音量を設定 (0.0 - 1.0)"""
        with self._state_lock:
            self.volume = max(0.0, min(1.0, volume))
        
    def set_track_volume(self, track_name, volume):
        """各トラックの音量を個別設定"""
        with self._state_lock:
            if track_name in self.track_volumes:
                self.track_volumes[track_name] = max(0.0, min(1.0, volume))

    def set_audio(self, audio_data, sample_rate):
        """
        再生する音声データをセット（単一トラック）
        """
        with self._state_lock:
            self.audio_data = np.asarray(audio_data, dtype=np.float32)
            self.tracks = {}  # マルチトラック情報をクリア
            self.track_volumes = {}
            self.sr = sample_rate
            self.current_position = 0
            self._stop_flag = False
        
    def set_multi_track_audio(self, tracks, sample_rate):
        """
        再生する音声データをセット（マルチトラック）
        
        Args:
            tracks: { 'vocals': data, 'drums': data ... }
            sample_rate: サンプリングレート
        """
        if not tracks:
            raise ValueError("tracks is empty")

        with self._state_lock:
            self.tracks = tracks
            self.track_volumes = {name: 1.0 for name in tracks}  # 初期音量はすべて100%
            self.sr = sample_rate
        
            # duration計算や互換性のために audio_data に最初のトラックの長さを参照できるデータを入れる
            # ※ 実際に playback で使うわけではないが、外部から len(player.audio_data) アクセスがあるため
            first_key = next(iter(tracks.keys()))
            self.audio_data = tracks[first_key]  # 参照用
        
            self.current_position = 0
            self._stop_flag = False
        
    def _ensure_stopped(self):
        """既存の再生スレッドを確実に停止して待機"""
        with self._state_lock:
            self._stop_flag = True
            self.is_playing = False
            thread = self.play_thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                # 停止完了を確認できない場合は再開を拒否して二重再生を防ぐ
                return False

        with self._state_lock:
            if self.play_thread is thread:
                self.play_thread = None
        return True

    def play(self):
        """音声を再生開始"""
        with self._state_lock:
            has_audio = self.audio_data is not None
            has_tracks = bool(self.tracks)
            paused = self.is_paused
            pause_time = self.pause_time
            start_time = self.start_time

        if not has_audio and not has_tracks:
            return

        if paused:
            # 一時停止から再開（前のスレッドの終了を確認）
            if not self._ensure_stopped():
                return
            with self._state_lock:
                self.is_paused = False
                self.start_time = time.time() - (pause_time - start_time)
        else:
            if not self._ensure_stopped():
                return
            with self._state_lock:
                self.current_position = 0
                self.start_time = time.time()

        with self._state_lock:
            self.is_playing = True
            self._stop_flag = False
            self.play_thread = threading.Thread(target=self._play_audio, daemon=True)
            thread = self.play_thread
        thread.start()
    
    def _play_audio(self):
        """内部再生処理"""
        try:
            # マルチトラックモード判定
            with self._state_lock:
                use_multitrack = bool(self.tracks)
            
            # 参照用長さを取得
            if use_multitrack:
                with self._state_lock:
                    tracks_snapshot = dict(self.tracks)
                first_track = next(iter(tracks_snapshot.values()))
                ref_len = min(len(track) for track in tracks_snapshot.values())
                # チャンネル数
                base_channels = first_track.shape[1] if len(first_track.shape) > 1 else 1
            else:
                with self._state_lock:
                    single_audio = self.audio_data.copy()
                ref_len = len(single_audio)
                base_channels = single_audio.shape[1] if len(single_audio.shape) > 1 else 1
            
            channels = base_channels
            
            # ストリーム再生
            def audio_callback(outdata, frames, time_info, status):
                if status:
                    # print(f"Audio callback status: {status}")
                    pass
                
                # Check control flag
                with self._state_lock:
                    playing = self.is_playing
                    current_position = self.current_position

                if not playing:
                    raise sd.CallbackStop
                
                # Calculate chunk size
                remaining = ref_len - current_position
                
                if remaining <= 0:
                    outdata[:] = 0
                    raise sd.CallbackStop # Signal completion

                chunk_size = min(frames, remaining)
                # end_pos = self.current_position + chunk_size
                # chunk = audio[self.current_position:end_pos]
                
                # データ生成
                if use_multitrack:
                    # ミックスダウン
                    start = current_position
                    end = start + chunk_size
                    
                    mixed = np.zeros((chunk_size, channels), dtype=np.float32)
                    
                    with self._state_lock:
                        track_volumes_snapshot = dict(self.track_volumes)

                    for name, track in tracks_snapshot.items():
                        vol = track_volumes_snapshot.get(name, 1.0)
                        if vol > 0.001:
                            part = track[start:end]
                            # チャンネル合わせ
                            if len(part.shape) == 1:
                                part = part[:, np.newaxis]
                            part_channels = part.shape[1]
                            
                            # 単純加算 (チャンネル数が合う前提、またはbroadcast)
                            # プレイヤーの channels と track の channels が違う場合は簡易処理
                            if part_channels == channels:
                                mixed += part * vol
                            elif part_channels == 1 and channels > 1:
                                mixed += part * vol # broadcast
                            # else: 複雑なチャンネル変換は省略
                    
                    chunk = mixed
                else:
                    chunk = single_audio[current_position : current_position + chunk_size]

                # Reshape/Broadcasting handle
                input_channels = 1
                if len(chunk.shape) > 1:
                    input_channels = chunk.shape[1]
                else:
                    chunk = chunk.reshape(-1, 1) # Treat as (N, 1)
                    input_channels = 1

                output_channels = outdata.shape[1]
                with self._state_lock:
                    master_volume = self.volume
                
                # Copy data safely with Volume control
                valid_channels = min(input_channels, output_channels)
                outdata[:chunk_size, :valid_channels] = chunk[:, :valid_channels] * master_volume
                
                # Zero fill rest if chunk is smaller than frames
                if chunk_size < frames:
                    outdata[chunk_size:, :] = 0
                
                # Zero fill extra channels if output has more than input
                if output_channels > input_channels:
                     outdata[:chunk_size, input_channels:] = 0

                with self._state_lock:
                    self.current_position += chunk_size
            
            with sd.OutputStream(channels=channels, samplerate=self.sr, 
                                callback=audio_callback, blocksize=4096):
                
                # Check status periodically
                while True:
                    with self._state_lock:
                        current_position = self.current_position
                        should_stop = self._stop_flag
                        playing = self.is_playing

                    if not playing:
                        break
                    if current_position >= ref_len:
                        break
                    if should_stop:
                        break
                    time.sleep(0.1)

        except Exception as e:
            print(f"再生エラー: {e}")
        finally:
            with self._state_lock:
                self.is_playing = False
                self.play_thread = None
    
    def pause(self):
        """音声を一時停止"""
        with self._state_lock:
            if self.is_playing:
                self.is_paused = True
                self.pause_time = time.time()
                self.is_playing = False
    
    def stop(self):
        """音声を停止"""
        self._ensure_stopped()
        with self._state_lock:
            self.is_paused = False
            self.current_position = 0
    
    def get_total_samples(self):
        """再生可能な総サンプル数を取得"""
        with self._state_lock:
            if self.tracks:
                return min(len(track) for track in self.tracks.values())
            if self.audio_data is not None:
                return len(self.audio_data)
        return 0

    def seek(self, position):
        """
        再生位置をシーク
        
        Args:
            position: シーク先のサンプル番号
        """
        total = self.get_total_samples()
        if total > 0:
            with self._state_lock:
                self.current_position = max(0, min(position, total - 1))
                if self.sr:
                    self.start_time = time.time() - (self.current_position / self.sr)
    
    def is_currently_playing(self):
        """現在再生中かどうか"""
        with self._state_lock:
            return self.is_playing or self.is_paused
