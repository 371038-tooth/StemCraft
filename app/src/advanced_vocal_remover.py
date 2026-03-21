"""
高度なボーカル除去モジュール（AI解析）
Demucsを使用したAI解析ボーカル除去
"""

import threading

# Torch/Demucsのインポート（他ライブラリより先に読み込むことでDLL競合を回避）
# 特にNumPyより先にTorchをインポートすることが重要 (OpenMP DLL競合対策)
TORCH_AVAILABLE = False
AI_REMOVAL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Torch import failed. Basic vocal removal only. Details: {type(e).__name__}: {e}")
    TORCH_AVAILABLE = False
except OSError as e:
    print(f"Warning: Torch initialization error (DLL load failure). Details: {type(e).__name__}: {e}")
    TORCH_AVAILABLE = False

from pathlib import Path
import numpy as np
import soundfile as sf
import tempfile
from pydub import AudioSegment
from .audio_processor import AudioProcessor

# Demucsのインポート（Torch利用可時のみ）
if TORCH_AVAILABLE:
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        AI_REMOVAL_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Failed to import Demucs: {e}")
        AI_REMOVAL_AVAILABLE = False


# soundfile が対応しないフォーマット（pydub 経由で変換が必要）
_SOUNDFILE_UNSUPPORTED = {'.m4a', '.mp4', '.aac', '.alac'}


class AdvancedVocalRemover:
    """AI解析ボーカル除去クラス（Demucs使用）"""
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.sr = 44100
        self.device = None
        self.initialized = False
        self.initialization_error = None
        self._lock = threading.Lock()

    def _select_device(self):
        """利用可能なデバイスを選択（CUDAが使えない場合はCPUへフォールバック）"""
        if not TORCH_AVAILABLE:
            return None

        if torch.cuda.is_available():
            try:
                # GPU世代が PyTorch ビルドに含まれない場合があるためチェック
                capability = torch.cuda.get_device_capability(0)
                arch = f"sm_{capability[0]}{capability[1]}"
                arch_list_fn = getattr(torch.cuda, "get_arch_list", None)
                if callable(arch_list_fn):
                    arch_list = arch_list_fn() or []
                    if arch_list and arch not in arch_list:
                        name = torch.cuda.get_device_name(0)
                        print("[Warning] GPU detected but unsupported by this PyTorch build")
                        print(f"  GPU: {name} ({arch})")
                        print(f"  Supported architectures: {' '.join(arch_list)}")
                        print("  -> Running AI removal on CPU (GPU PyTorch update may help)")
                        return "cpu"

                # 実際にCUDAテンソルを確保できるかテスト
                torch.zeros(1, device="cuda")
                return "cuda"
            except Exception as e:
                print("[Warning] CUDA available but not usable")
                print(f"  Details: {type(e).__name__}: {e}")
                print("  -> Running AI removal on CPU")
                return "cpu"

        return "cpu"
    
    def initialize_model(self, model_name='htdemucs'):
        """
        Demucsモデルを初期化
        Args:
            model_name: 使用するモデル名 (htdemucs=4stem, htdemucs_6s=6stem)
        """
        with self._lock:
            if not TORCH_AVAILABLE:
                self.initialization_error = "Torch/Demucsが利用不可です"
                return False
                
            # 既に同じモデルがロードされている場合はスキップ
            if self.initialized and self.model is not None and self.current_model_name == model_name:
                return True
            
            try:
                # 初回のみデバイス選択
                if self.device is None:
                    self.device = self._select_device() or "cpu"
                
                print(f"[AI Init] Demucs model initialization... (Model: {model_name}, Device: {self.device})")
                
                # メモリ解放
                if self.model is not None:
                    del self.model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # モデルを取得
                self.model = get_model(model_name)
                self.current_model_name = model_name
                
                if self.device:
                    try:
                        self.model.to(self.device)
                    except Exception:
                        # 予期せぬ互換性問題がある場合はCPUにフォールバック
                        print("[Warning] Failed to transfer model to GPU. Using CPU.")
                        self.device = "cpu"
                        self.model.to(self.device)
                self.model.eval()  # 評価モード
                self.initialized = True
                print("[OK] Demucs model initialization complete")
                return True
            except Exception as e:
                self.initialization_error = str(e)
                self.model = None
                self.current_model_name = None
                self.initialized = False
                print(f"[Error] Demucs model initialization failed: {e}")
                return False
    
    def separate_audio(self, audio_path, output_dir=None, progress_callback=None):
        """
        AI解析で音源を分離（ボーカル、ドラム、ベース、その他）
        
        Args:
            audio_path: 入力音声ファイルパス
            output_dir: 出力ディレクトリ（オプション）
            progress_callback: 進捗を返すコールバック関数 func(current, total)
        
        Returns:
            dict: { 'vocals': np.array, 'drums': np.array, 'bass': np.array, 'other': np.array }
            各配列は (samples, channels) の形状
        """
        with self._lock:
            if not self.initialized or self.model is None:
                raise Exception("Demucsモデルが初期化されていません。")
            
            if not TORCH_AVAILABLE:
                raise Exception("Torch/Demucsが利用不可です")
            
            try:
                # 音声ファイルを読み込み
                print(f"[Audio] Loading audio file: {Path(audio_path).name}")
                # librosaなどで読み込むと (channels, samples) だったりするが、sf.readは (samples, channels)
                suffix = Path(audio_path).suffix.lower()
                if suffix in _SOUNDFILE_UNSUPPORTED:
                    ffmpeg_path = AudioProcessor._find_ffmpeg()
                    if ffmpeg_path is None:
                        raise RuntimeError(
                            f"{suffix} ファイルの変換に ffmpeg が必要です。\n\n"
                            "以下のいずれかを実行してください：\n"
                            "  1. run.bat を再実行して ffmpeg を自動セットアップする\n"
                            "  2. https://ffmpeg.org からダウンロードし PATH に追加する\n"
                            "  3. 設定画面で ffmpeg の実行ファイルのパスを指定する"
                        )
                    AudioProcessor._configure_pydub_ffmpeg(ffmpeg_path)
                    seg = AudioSegment.from_file(audio_path)
                    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    try:
                        seg.export(tmp_path, format='wav')
                        audio, sr = sf.read(tmp_path)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                else:
                    audio, sr = sf.read(audio_path)
                
                # Demucs向けにステレオ化し、(channels, samples) へ変換
                if len(audio.shape) == 1:
                    # モノラル -> ステレオ
                    audio = np.stack([audio, audio], axis=1)
                elif len(audio.shape) == 2:
                    channels = audio.shape[1]
                    if channels == 1:
                        # 1ch -> 2ch
                        audio = np.repeat(audio, 2, axis=1)
                    elif channels > 2:
                        # 3ch以上は平均でダウンミックスして2ch化
                        print(f"[Warning] Multi-channel input detected ({channels}ch). Downmixing to stereo.")
                        mono = np.mean(audio, axis=1, keepdims=True)
                        audio = np.repeat(mono, 2, axis=1)
                else:
                    raise Exception(f"Unsupported audio shape: {audio.shape}")

                # (samples, channels) -> (channels, samples)
                audio = audio.T
                
                # Tensorに変換 (1, channels, samples)
                original_length = audio.shape[1]  # 元の音声長を保持
                wav = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
                
                print(f"[Separation] Source separation in progress... Input shape: {wav.shape}")
                
                # tqdmのモンキーパッチを用いて進捗を取得する
                original_tqdm = None
                if progress_callback:
                    import demucs.apply
                    original_tqdm = getattr(demucs.apply.tqdm, 'tqdm', None)
                    
                    class PatchedTqdm:
                        def __init__(self_tqdm, iterable, **kwargs):
                            try:
                                self_tqdm.iterable = list(iterable)
                                self_tqdm.total = len(self_tqdm.iterable)
                            except TypeError:
                                self_tqdm.iterable = iterable
                                self_tqdm.total = getattr(iterable, '__len__', lambda: 0)()
                            self_tqdm.current = 0
                            
                            # 初期状態を送信
                            if self_tqdm.total > 0:
                                progress_callback(0, self_tqdm.total)
                                
                        def __iter__(self_tqdm):
                            for item in self_tqdm.iterable:
                                yield item
                                self_tqdm.current += 1
                                if self_tqdm.total > 0:
                                    progress_callback(self_tqdm.current, self_tqdm.total)
                                    
                    demucs.apply.tqdm.tqdm = PatchedTqdm

                # Demucsで分離
                try:
                    with torch.no_grad():
                        sources = apply_model(self.model, wav, shifts=1, split=True, overlap=0.25, progress=bool(progress_callback))
                finally:
                    # パッチを戻す
                    if progress_callback and original_tqdm is not None:
                        import demucs.apply
                        demucs.apply.tqdm.tqdm = original_tqdm
                
                # sources: (batch, sources_count, channels, time)
                # model.sources: ['drums', 'bass', 'other', 'vocals'] (example)
                
                stems = {}
                source_names = self.model.sources
                
                # 各ステムを抽出して辞書に格納
                for i, name in enumerate(source_names):
                    source_wav = sources[0, i]  # (channels, time)
                    source_np = source_wav.cpu().numpy()
                    # 元の音声長にトリミング（Demucsのパディング分を除去）
                    source_np = source_np[:, :original_length]
                    # (channels, samples) -> (samples, channels)
                    stems[name] = source_np.T
                
                # GPUメモリを解放
                del sources, wav
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                print("[OK] Source separation complete")
                return stems
            
            except Exception as e:
                raise Exception(f"AI解析エラー: {str(e)}")

    def get_status(self):
        """状態を取得"""
        if not TORCH_AVAILABLE:
            return "✗ Torch/Demucsの環境設定が必要（基本除去は利用可能）"
        elif self.initialized:
            return f"[OK] Demucs AI removal enabled (Device: {self.device})"
        elif self.initialization_error:
            return f"✗ 初期化エラー: {self.initialization_error}"
        else:
            return "⏳ 初期化中..."


# グローバルインスタンス
_advanced_remover = None
_advanced_remover_init_failed = False


def get_advanced_remover():
    """AdvancedVocalRemoverのシングルトンを取得"""
    global _advanced_remover, _advanced_remover_init_failed
    if _advanced_remover is None and not _advanced_remover_init_failed:
        try:
            _advanced_remover = AdvancedVocalRemover()
        except Exception as e:
            print(f"AdvancedVocalRemover初期化エラー: {e}")
            _advanced_remover_init_failed = True
    return _advanced_remover


if __name__ == "__main__":
    remover = get_advanced_remover()
    if remover:
        print(remover.get_status())
    else:
        print("✗ AI解析ボーカル除去は利用不可")
