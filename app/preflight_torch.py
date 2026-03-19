"""Preflight checks for Torch/Demucs environment.

Exit codes:
  0: OK
  1: torch missing
  2: CUDA expected (NVIDIA present) but torch.cuda not available
  3: CUDA tensor allocation failed (likely incompatible build/driver)
"""

from __future__ import annotations

import importlib.util
import os
import sys


def main() -> int:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return 1

    try:
        import torch  # noqa: F401
    except OSError:
        # DLL初期化失敗など。環境再セットアップを促すため非0を返す
        return 10
    except Exception:
        return 11

    has_nvidia = os.environ.get("HAS_NVIDIA") == "1"
    if not has_nvidia:
        return 0

    if not torch.cuda.is_available():
        return 2

    try:
        torch.zeros(1, device="cuda")
    except Exception:
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
