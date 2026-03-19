"""
オーディオ処理モジュール
ボーカル除去と音声ファイル処理を担当
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
from pydub import AudioSegment


class AudioProcessor:
    """オーディオ処理クラス"""
    
    def __init__(self):
        self.sr = 44100  # サンプリングレート
        self.y = None  # 音声データ
        self.sr_loaded = None
        
    def load_audio(self, file_path):
        """
        音声ファイルを読み込む（m4a、mp3 など多形式対応）
        
        Args:
            file_path: 音声ファイルのパス
            
        Returns:
            bool: 成功した場合True
        """
        try:
            file_path = Path(file_path)
            file_ext = file_path.suffix.lower()
            
            # librosa がネイティブにサポート: wav, mp3, flac など
            # m4a, mp4, aac は pydub で変換が必要
            unsupported_formats = ['.m4a', '.mp4', '.aac', '.alac']
            
            if file_ext in unsupported_formats:
                # pydub を使用して wav に変換
                self.y, self.sr_loaded = self._load_with_pydub(str(file_path))
            else:
                # librosa で直接読み込み
                self.y, self.sr_loaded = librosa.load(str(file_path), sr=self.sr, mono=False)

                # librosa mono=False は (channels, samples) を返すため (samples, channels) に揃える
                if isinstance(self.y, np.ndarray) and self.y.ndim == 2:
                    self.y = self.y.T

            # モノラル(1D)の場合は (samples, 1) に整形して後続処理と統一
            if isinstance(self.y, np.ndarray) and self.y.ndim == 1:
                self.y = self.y.reshape(-1, 1)
            
            return True
        except Exception as e:
            print(f"エラー: {e}")
            return False
    
    def _load_with_pydub(self, file_path):
        """
        pydub を使用して音声ファイルを読み込む
        
        Args:
            file_path: 音声ファイルのパス
            
        Returns:
            tuple: (y, sr) 音声データとサンプリングレート
        """
        tmp_path = None
        try:
            # pydub で音声ファイルを読み込む
            audio = AudioSegment.from_file(file_path)
            
            # 一時的な wav ファイルに変換
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            audio.export(tmp_path, format="wav")
            
            # librosa で読み込む
            y, sr = librosa.load(tmp_path, sr=self.sr, mono=False)

            # (channels, samples) -> (samples, channels)
            if isinstance(y, np.ndarray) and y.ndim == 2:
                y = y.T
            elif isinstance(y, np.ndarray) and y.ndim == 1:
                y = y.reshape(-1, 1)
            
            return y, sr
        except Exception as e:
            print(f"pydub を使用した読み込みエラー: {e}")
            raise
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()
    
    def save_audio(self, audio_data, output_path):
        """
        音声データをファイルに保存
        
        Args:
            audio_data: 音声データ
            output_path: 出力ファイルのパス
            
        Returns:
            bool: 成功した場合True
        """
        try:
            if self.sr_loaded is None:
                print("保存エラー: 音声が読み込まれていません")
                return False

            ext = Path(output_path).suffix.lower()
            
            # soundfileが標準でサポートする形式のうち、クラッシュしない安全なもの
            # OGGはlibsndfileのVorbisエンコーダによるシークレットクラッシュの恐れがあるため外す
            if ext in ['.wav', '.flac']:
                sf.write(output_path, audio_data, self.sr_loaded)
            else:
                # MP3/OGG等のためpydubを使ってエクスポート
                # 一旦16bit PCM WAVとして一時保存 (pydub互換性のため)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    sf.write(tmp_path, audio_data, self.sr_loaded, subtype='PCM_16')
                    audio = AudioSegment.from_wav(tmp_path)
                    
                    export_format = ext[1:] # .mp3 -> mp3
                    if export_format == 'm4a':
                        export_format = 'ipod'
                        
                    # MP3の場合はビットレートを指定
                    kwargs = {}
                    if export_format == 'mp3':
                        kwargs['bitrate'] = '192k'
                        
                    audio.export(output_path, format=export_format, **kwargs)
                finally:
                    # 一時ファイルを削除
                    if Path(tmp_path).exists():
                        Path(tmp_path).unlink()
                        
            return True
        except Exception as e:
            print(f"保存エラー: {e}")
            return False
    
    def get_duration(self):
        """
        現在読み込んでいる音声の長さを取得
        
        Returns:
            float: 長さ（秒）
        """
        if self.y is None:
            return 0
        return len(self.y) / self.sr_loaded
    
    def get_audio_data(self):
        """現在の音声データを取得"""
        return self.y
    
    def clear(self):
        """音声データをクリア"""
        self.y = None
        self.sr_loaded = None
