"""
AudioProcessor の ffmpeg 検出・設定メソッドおよび load_audio 戻り値のテスト
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# app/src を import パスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio_processor import AudioProcessor


# ──────────────────────────────────────────────────────────────────────────────
# _find_ffmpeg
# ──────────────────────────────────────────────────────────────────────────────

class TestFindFfmpeg:
    def test_returns_path_when_in_path(self):
        """PATH に ffmpeg がある場合はそのパスを返す"""
        with patch('shutil.which', return_value='/usr/bin/ffmpeg'):
            result = AudioProcessor._find_ffmpeg()
        assert result == '/usr/bin/ffmpeg'

    def test_returns_local_when_not_in_path(self, tmp_path):
        """PATH にないがローカルに存在する場合はローカルパスを返す"""
        # ローカル ffmpeg.exe を模擬
        local_ffmpeg = tmp_path / 'ffmpeg' / 'bin' / 'ffmpeg.exe'
        local_ffmpeg.parent.mkdir(parents=True)
        local_ffmpeg.touch()

        with patch('shutil.which', return_value=None):
            # __file__ の親の親を tmp_path に向ける
            with patch.object(Path, 'exists', return_value=True):
                # 実際の Local パス解決は内部で行われるため、
                # exists() が True を返せば文字列パスが返ることを確認
                result = AudioProcessor._find_ffmpeg()
        # None でなければローカルパスが解決されている
        assert result is not None

    def test_returns_none_when_not_found(self):
        """ffmpeg がどこにもない場合は None を返す"""
        with patch('shutil.which', return_value=None):
            with patch.object(Path, 'exists', return_value=False):
                result = AudioProcessor._find_ffmpeg()
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# _configure_pydub_ffmpeg
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigurePydubFfmpeg:
    @pytest.fixture(autouse=True)
    def restore_pydub_state(self):
        """AudioSegment のクラス変数をテスト後に元に戻す"""
        from pydub import AudioSegment
        original_converter = AudioSegment.converter
        ffprobe_existed = hasattr(AudioSegment, "ffprobe")
        original_ffprobe = getattr(AudioSegment, "ffprobe", None)
        yield
        AudioSegment.converter = original_converter
        if ffprobe_existed:
            AudioSegment.ffprobe = original_ffprobe
        elif hasattr(AudioSegment, "ffprobe"):
            del AudioSegment.ffprobe

    def test_sets_converter(self, tmp_path):
        """converter が指定したパスに設定される"""
        from pydub import AudioSegment
        ffmpeg_path = str(tmp_path / 'ffmpeg.exe')
        # ffprobe は存在しないケースでも converter だけ設定される
        AudioProcessor._configure_pydub_ffmpeg(ffmpeg_path)
        assert AudioSegment.converter == ffmpeg_path

    def test_sets_ffprobe_when_exists(self, tmp_path):
        """ffprobe.exe が同ディレクトリに存在する場合は ffprobe も設定される"""
        from pydub import AudioSegment
        ffmpeg_path = tmp_path / 'ffmpeg.exe'
        ffprobe_path = tmp_path / 'ffprobe.exe'
        ffprobe_path.touch()

        AudioProcessor._configure_pydub_ffmpeg(str(ffmpeg_path))
        assert AudioSegment.ffprobe == str(ffprobe_path)


# ──────────────────────────────────────────────────────────────────────────────
# __init__ ffmpeg_path パラメータ
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioProcessorInit:
    def test_default_ffmpeg_path_is_none(self):
        processor = AudioProcessor()
        assert processor._ffmpeg_path is None

    def test_ffmpeg_path_is_stored(self):
        processor = AudioProcessor(ffmpeg_path='/custom/ffmpeg')
        assert processor._ffmpeg_path == '/custom/ffmpeg'


# ──────────────────────────────────────────────────────────────────────────────
# load_audio 戻り値タプル
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadAudioReturnValue:
    def test_success_returns_true_empty_string(self, tmp_path):
        """WAV ファイルの読み込み成功時に (True, '') を返す"""
        import numpy as np
        import soundfile as sf

        wav_path = tmp_path / 'test.wav'
        sr = 22050
        data = np.zeros(sr, dtype=np.float32)
        sf.write(str(wav_path), data, sr)

        processor = AudioProcessor()
        result = processor.load_audio(str(wav_path))
        assert result == (True, '')

    def test_failure_returns_false_with_message(self, tmp_path):
        """存在しないファイルの読み込み失敗時に (False, <メッセージ>) を返す"""
        processor = AudioProcessor()
        result = processor.load_audio(str(tmp_path / 'nonexistent.wav'))
        success, msg = result
        assert success is False
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_m4a_without_ffmpeg_returns_false_with_japanese_message(self, tmp_path):
        """ffmpeg がない環境で M4A を読もうとすると日本語エラーメッセージが返る"""
        # 空の .m4a ファイルを作成（中身は問わない）
        m4a_path = tmp_path / 'test.m4a'
        m4a_path.write_bytes(b'\x00' * 16)

        processor = AudioProcessor(ffmpeg_path=None)
        with patch.object(AudioProcessor, '_find_ffmpeg', return_value=None):
            success, msg = processor.load_audio(str(m4a_path))

        assert success is False
        assert 'ffmpeg' in msg
