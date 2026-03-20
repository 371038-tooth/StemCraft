"""
AudioProcessor のピッチ・テンポ検出/変換メソッドのテスト
"""

import sys
import os
import pytest
import numpy as np

# app/src を import パスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio_processor import AudioProcessor
import src.audio_processor as audio_processor_module


# ──────────────────────────────────────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    return AudioProcessor()


def _make_sine(freq=440.0, sr=22050, duration=2.0, channels=2):
    """テスト用サイン波を生成して (samples, channels) 形式で返す"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 1:
        return mono.reshape(-1, 1), sr
    else:
        stereo = np.stack([mono, mono], axis=1)  # (samples, 2)
        return stereo, sr


# ──────────────────────────────────────────────────────────────────────────────
# detect_tempo
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectTempo:
    def test_returns_int(self, processor):
        audio, sr = _make_sine(duration=5.0)
        mono = audio[:, 0]
        result = processor.detect_tempo(mono, sr)
        assert isinstance(result, int)

    def test_positive_bpm(self, processor):
        audio, sr = _make_sine(duration=5.0)
        mono = audio[:, 0]
        result = processor.detect_tempo(mono, sr)
        assert result > 0

    def test_reasonable_range(self, processor):
        """librosa は通常 30-300 BPM の範囲内を返す"""
        audio, sr = _make_sine(duration=5.0)
        mono = audio[:, 0]
        result = processor.detect_tempo(mono, sr)
        assert 30 <= result <= 300


# ──────────────────────────────────────────────────────────────────────────────
# detect_key
# ──────────────────────────────────────────────────────────────────────────────

VALID_KEY_NAMES = [
    "C メジャー", "C# メジャー", "D♭ メジャー", "D メジャー", "D# メジャー", "E♭ メジャー",
    "E メジャー", "F メジャー", "F# メジャー", "G♭ メジャー", "G メジャー", "G# メジャー",
    "A♭ メジャー", "A メジャー", "A# メジャー", "B♭ メジャー", "B メジャー",
    "C マイナー", "C# マイナー", "D♭ マイナー", "D マイナー", "D# マイナー", "E♭ マイナー",
    "E マイナー", "F マイナー", "F# マイナー", "G♭ マイナー", "G マイナー", "G# マイナー",
    "A♭ マイナー", "A マイナー", "A# マイナー", "B♭ マイナー", "B マイナー",
]


class TestDetectKey:
    def test_returns_string(self, processor):
        audio, sr = _make_sine()
        mono = audio[:, 0]
        result = processor.detect_key(mono, sr)
        assert isinstance(result, str)

    def test_returns_valid_key(self, processor):
        """返す文字列が既知のキー名形式に合致すること"""
        audio, sr = _make_sine(duration=4.0)
        mono = audio[:, 0]
        result = processor.detect_key(mono, sr)
        # "X メジャー" または "X マイナー" の形式であること
        assert "メジャー" in result or "マイナー" in result

    def test_contains_note_name(self, processor):
        audio, sr = _make_sine(duration=4.0)
        mono = audio[:, 0]
        result = processor.detect_key(mono, sr)
        # 音名部分が含まれる
        note_names = ["C", "D", "E", "F", "G", "A", "B"]
        assert any(n in result for n in note_names)


# ──────────────────────────────────────────────────────────────────────────────
# apply_pitch_shift
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyPitchShift:
    def test_output_shape_mono(self, processor):
        audio, sr = _make_sine(channels=1)
        out = processor.apply_pitch_shift(audio, n_steps=2, sr=sr)
        assert out.shape == audio.shape

    def test_output_shape_stereo(self, processor):
        audio, sr = _make_sine(channels=2)
        out = processor.apply_pitch_shift(audio, n_steps=2, sr=sr)
        assert out.shape == audio.shape

    def test_zero_shift_preserves_length(self, processor):
        audio, sr = _make_sine(channels=2)
        out = processor.apply_pitch_shift(audio, n_steps=0, sr=sr)
        assert out.shape == audio.shape

    def test_output_dtype_float32(self, processor):
        audio, sr = _make_sine(channels=2)
        out = processor.apply_pitch_shift(audio, n_steps=1, sr=sr)
        assert out.dtype == np.float32

    def test_negative_shift(self, processor):
        audio, sr = _make_sine(channels=1)
        out = processor.apply_pitch_shift(audio, n_steps=-2, sr=sr)
        assert out.shape == audio.shape


# ──────────────────────────────────────────────────────────────────────────────
# apply_time_stretch
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyTimeStretch:
    def test_output_shape_mono(self, processor):
        audio, sr = _make_sine(channels=1, duration=2.0)
        rate = 1.5  # 1.5倍速
        out = processor.apply_time_stretch(audio, rate=rate, sr=sr)
        # サンプル数が approx 1/rate 倍になる
        expected_len = int(len(audio) / rate)
        assert abs(len(out) - expected_len) <= 512  # STFTのホップサイズ誤差を許容

    def test_output_shape_stereo(self, processor):
        audio, sr = _make_sine(channels=2, duration=2.0)
        rate = 0.8
        out = processor.apply_time_stretch(audio, rate=rate, sr=sr)
        # チャンネル数は変わらない
        assert out.shape[1] == 2

    def test_output_dtype_float32(self, processor):
        audio, sr = _make_sine(channels=2)
        out = processor.apply_time_stretch(audio, rate=1.0, sr=sr)
        assert out.dtype == np.float32

    def test_rate_1_preserves_length(self, processor):
        audio, sr = _make_sine(channels=2, duration=2.0)
        out = processor.apply_time_stretch(audio, rate=1.0, sr=sr)
        assert abs(len(out) - len(audio)) <= 512


# ──────────────────────────────────────────────────────────────────────────────
# apply_pitch_and_tempo
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyPitchAndTempo:
    def test_noop_keeps_shape_and_dtype(self, processor):
        audio, sr = _make_sine(channels=2, duration=2.0)
        out = processor.apply_pitch_and_tempo(audio, n_steps=0, rate=1.0, sr=sr)
        assert out.shape == audio.shape
        assert out.dtype == np.float32

    def test_pitch_only_keeps_shape(self, processor):
        audio, sr = _make_sine(channels=2, duration=2.0)
        out = processor.apply_pitch_and_tempo(audio, n_steps=2, rate=1.0, sr=sr)
        assert out.shape == audio.shape

    def test_tempo_only_changes_length(self, processor):
        audio, sr = _make_sine(channels=2, duration=2.0)
        rate = 1.25
        out = processor.apply_pitch_and_tempo(audio, n_steps=0, rate=rate, sr=sr)
        expected_len = int(len(audio) / rate)
        assert abs(len(out) - expected_len) <= 1024

    def test_progress_callback_called(self, processor):
        audio, sr = _make_sine(channels=2, duration=1.0)
        progress_values = []

        out = processor.apply_pitch_and_tempo(
            audio,
            n_steps=1,
            rate=1.0,
            sr=sr,
            progress_callback=lambda p: progress_values.append(int(p)),
        )

        assert out.shape[1] == audio.shape[1]
        assert len(progress_values) >= 1
        assert progress_values[-1] == 100

    def test_fallback_to_librosa_when_pedalboard_fails(self, processor, monkeypatch):
        audio, sr = _make_sine(channels=2, duration=2.0)

        monkeypatch.setattr(audio_processor_module, "_PEDALBOARD_AVAILABLE", True)
        monkeypatch.setattr(audio_processor_module, "_PEDALBOARD_HAS_TIMESTRETCH", True)

        def broken_time_stretch(*args, **kwargs):
            raise RuntimeError("simulated pedalboard failure")

        class _DummyPedalboardModule:
            time_stretch = staticmethod(broken_time_stretch)

        monkeypatch.setattr(audio_processor_module, "_pedalboard", _DummyPedalboardModule, raising=False)

        out = processor.apply_pitch_and_tempo(audio, n_steps=0, rate=1.2, sr=sr)
        assert out.ndim == 2
        assert out.shape[1] == audio.shape[1]


# ──────────────────────────────────────────────────────────────────────────────
# transpose_key
# ──────────────────────────────────────────────────────────────────────────────

class TestTransposeKey:
    def test_zero_semitones_unchanged(self):
        result = AudioProcessor.transpose_key("C メジャー", 0)
        assert result == "C メジャー"

    def test_plus_one_semitone(self):
        result = AudioProcessor.transpose_key("C メジャー", 1)
        # C + 1 = C# または D♭
        assert result in ("C# メジャー", "D♭ メジャー")

    def test_minus_one_semitone(self):
        result = AudioProcessor.transpose_key("C メジャー", -1)
        # C - 1 = B
        assert result in ("B メジャー",)

    def test_plus_twelve_same_note(self):
        result = AudioProcessor.transpose_key("C メジャー", 12)
        assert result == "C メジャー"

    def test_minus_twelve_same_note(self):
        result = AudioProcessor.transpose_key("A マイナー", -12)
        assert result == "A マイナー"

    def test_ab_major_plus_two(self):
        """検出キーが A♭ メジャーのとき +2 で B♭ メジャーになること"""
        result = AudioProcessor.transpose_key("A♭ メジャー", 2)
        assert result in ("B♭ メジャー", "A# メジャー")

    def test_ab_major_minus_one(self):
        """A♭ - 1 = G"""
        result = AudioProcessor.transpose_key("A♭ メジャー", -1)
        assert result == "G メジャー"

    def test_preserves_scale_type_major(self):
        result = AudioProcessor.transpose_key("D メジャー", 3)
        assert "メジャー" in result

    def test_preserves_scale_type_minor(self):
        result = AudioProcessor.transpose_key("E マイナー", 5)
        assert "マイナー" in result

    def test_wrap_around_from_b(self):
        """B + 1 = C (ラップアラウンド)"""
        result = AudioProcessor.transpose_key("B メジャー", 1)
        assert result == "C メジャー"

    def test_fsharp_major_plus_one(self):
        """F# + 1 = G"""
        result = AudioProcessor.transpose_key("F# メジャー", 1)
        assert result == "G メジャー"


# ──────────────────────────────────────────────────────────────────────────────
# detected_bpm / detected_key インスタンス変数
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectedAttributes:
    def test_initial_values_none(self, processor):
        assert processor.detected_bpm is None
        assert processor.detected_key is None
