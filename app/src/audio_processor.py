"""
オーディオ処理モジュール
ボーカル除去と音声ファイル処理を担当
"""

import numpy as np
import librosa
import librosa.feature.rhythm
import soundfile as sf
from pathlib import Path
import tempfile
from pydub import AudioSegment

try:
    import pedalboard as _pedalboard
    _PEDALBOARD_AVAILABLE = True
except ImportError:
    _PEDALBOARD_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Krumhansl-Schmuckler 調性プロファイル（モジュール定数）
# ──────────────────────────────────────────────────────────────────────────────
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# 半音インデックス → 正準音名（♯/♭混在の慣例的表記）
_NOTE_NAMES = ["C", "C#", "D", "E♭", "E", "F", "F#", "G", "A♭", "A", "B♭", "B"]

# 音名文字列 → 半音インデックス（異名同音を含む）
_NOTE_ALIASES: dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "D♭": 1,
    "D": 2,
    "D#": 3, "E♭": 3,
    "E": 4, "F♭": 4,
    "F": 5, "E#": 5,
    "F#": 6, "G♭": 6,
    "G": 7,
    "G#": 8, "A♭": 8,
    "A": 9,
    "A#": 10, "B♭": 10,
    "B": 11, "C♭": 11,
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度を計算する（ゼロ除算安全）"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


class AudioProcessor:
    """オーディオ処理クラス"""
    
    def __init__(self):
        self.sr = 44100  # サンプリングレート
        self.y = None  # 音声データ
        self.sr_loaded = None
        self.detected_bpm: int | None = None   # 検出BPM
        self.detected_key: str | None = None   # 検出キー（例："A♭ メジャー"）
        
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
        self.detected_bpm = None
        self.detected_key = None

    # ──────────────────────────────────────────────────────────────────────────
    # テンポ・キー自動検出
    # ──────────────────────────────────────────────────────────────────────────

    def detect_tempo(self, y: np.ndarray, sr: int) -> int:
        """
        BPM を検出して int に丸めて返す。

        Args:
            y: モノラル 1D float32 配列
            sr: サンプリングレート

        Returns:
            int: 検出 BPM
        """
        tempo = librosa.feature.rhythm.tempo(y=y.astype(np.float32), sr=sr)
        # librosa >= 0.10 は numpy array を返す場合があるため flatten して取り出す
        bpm = float(np.atleast_1d(tempo)[0])
        return int(round(bpm))

    def detect_key(self, y: np.ndarray, sr: int) -> str:
        """
        Krumhansl-Schmuckler アルゴリズムでキーを推定し文字列で返す。

        Args:
            y: モノラル 1D float32 配列
            sr: サンプリングレート

        Returns:
            str: 例 "A♭ メジャー" / "E マイナー"
        """
        chroma = librosa.feature.chroma_cqt(y=y.astype(np.float32), sr=sr)
        chroma_mean = chroma.mean(axis=1)  # shape (12,)

        best_score = -np.inf
        best_note = "C"
        best_mode = "メジャー"

        for shift in range(12):
            major_profile = np.roll(_KS_MAJOR, shift)
            minor_profile = np.roll(_KS_MINOR, shift)

            score_major = _cosine_similarity(chroma_mean, major_profile)
            score_minor = _cosine_similarity(chroma_mean, minor_profile)

            if score_major > best_score:
                best_score = score_major
                best_note = _NOTE_NAMES[shift]
                best_mode = "メジャー"

            if score_minor > best_score:
                best_score = score_minor
                best_note = _NOTE_NAMES[shift]
                best_mode = "マイナー"

        return f"{best_note} {best_mode}"

    # ──────────────────────────────────────────────────────────────────────────
    # ピッチ・テンポ変換
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_ch_first(audio: np.ndarray) -> np.ndarray:
        """内部ユーティリティ: (samples, channels) -> (channels, samples)"""
        audio = np.asarray(audio, dtype=np.float32)
        return audio.T if audio.ndim == 2 else audio[np.newaxis, :]

    @staticmethod
    def _from_ch_first(result: np.ndarray, original_ndim: int) -> np.ndarray:
        """内部ユーティリティ: (channels, samples) -> (samples, channels)"""
        return result.T if original_ndim == 2 else result[0].reshape(-1, 1)

    def apply_pitch_and_tempo(self, audio: np.ndarray, n_steps: int, rate: float, sr: int) -> np.ndarray:
        """ピッチとテンポを一括適用（pedalboard を優先使用）

        Args:
            audio: (samples, channels) 形式の ndarray
            n_steps: 半音単位のピッチシフト量（0 で変換なし）
            rate: タイムストレッチ倍率（1.0 で変換なし、>1 で高速）
            sr: サンプリングレート

        Returns:
            ndarray: (samples_out, channels) 形式の変換済み音声
        """
        if n_steps == 0 and abs(rate - 1.0) <= 0.001:
            return np.asarray(audio, dtype=np.float32)

        if _PEDALBOARD_AVAILABLE:
            ch_first = self._to_ch_first(audio)
            result = _pedalboard.time_stretch(
                ch_first,
                float(sr),
                stretch_factor=float(rate),
                pitch_shift_in_semitones=float(n_steps),
            )
            return self._from_ch_first(result, audio.ndim)

        # pedalboard なしのフォールバック: librosa で順次適用
        result = np.asarray(audio, dtype=np.float32)
        if n_steps != 0:
            result = self.apply_pitch_shift(result, n_steps, sr)
        if abs(rate - 1.0) > 0.001:
            result = self.apply_time_stretch(result, rate, sr)
        return result
    def apply_pitch_shift(self, audio: np.ndarray, n_steps: int, sr: int) -> np.ndarray:
        """
        ピッチシフトを適用する。ステレオ入力にも対応。

        Args:
            audio: (samples, channels) 形式の ndarray
            n_steps: 半音単位のシフト量（整数）
            sr: サンプリングレート

        Returns:
            ndarray: (samples, channels) 形式のピッチシフト済み音声
        """
        audio = np.asarray(audio, dtype=np.float32)
        n_channels = audio.shape[1] if audio.ndim == 2 else 1

        if n_channels == 1:
            shifted = librosa.effects.pitch_shift(
                audio[:, 0] if audio.ndim == 2 else audio,
                sr=sr,
                n_steps=float(n_steps),
            )
            return shifted.reshape(-1, 1)

        channels_out = []
        for ch in range(n_channels):
            shifted_ch = librosa.effects.pitch_shift(
                audio[:, ch],
                sr=sr,
                n_steps=float(n_steps),
            )
            channels_out.append(shifted_ch)

        return np.stack(channels_out, axis=1).astype(np.float32)

    def apply_time_stretch(self, audio: np.ndarray, rate: float, sr: int) -> np.ndarray:
        """
        タイムストレッチを適用する。ステレオ入力にも対応。

        Args:
            audio: (samples, channels) 形式の ndarray
            rate: 再生速度倍率（>1 で速く、<1 で遅くなる）
            sr: サンプリングレート（現在は未使用だが将来の拡張用に保持）

        Returns:
            ndarray: (samples_out, channels) 形式のストレッチ済み音声
        """
        audio = np.asarray(audio, dtype=np.float32)
        n_channels = audio.shape[1] if audio.ndim == 2 else 1

        if n_channels == 1:
            stretched = librosa.effects.time_stretch(
                audio[:, 0] if audio.ndim == 2 else audio,
                rate=float(rate),
            )
            return stretched.reshape(-1, 1)

        channels_out = []
        for ch in range(n_channels):
            stretched_ch = librosa.effects.time_stretch(
                audio[:, ch],
                rate=float(rate),
            )
            channels_out.append(stretched_ch)

        # チャンネルごとに長さが僅かにずれる場合があるため最小長に揃える
        min_len = min(c.shape[0] for c in channels_out)
        trimmed = [c[:min_len] for c in channels_out]
        return np.stack(trimmed, axis=1).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # キー変換ユーティリティ
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def transpose_key(key_str: str, semitones: int) -> str:
        """
        キー文字列を指定した半音数だけ移調して返す。

        Args:
            key_str: 例 "A♭ メジャー"
            semitones: 半音数（負も可）

        Returns:
            str: 移調後のキー文字列
        """
        parts = key_str.rsplit(" ", 1)
        if len(parts) != 2:
            return key_str
        note_str, mode = parts
        note_str = note_str.strip()
        mode = mode.strip()

        if note_str not in _NOTE_ALIASES:
            return key_str

        idx = (_NOTE_ALIASES[note_str] + semitones) % 12
        return f"{_NOTE_NAMES[idx]} {mode}"
