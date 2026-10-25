"""
check_gpu.py — Phase 0 environment check.

Prints Python / torch / CUDA status, ffmpeg availability, and the resolved
model + threshold configuration, then gives a plain-English verdict on whether
you'll run at GPU speed or in the slower CPU fallback.

    python check_gpu.py
"""

from __future__ import annotations

import shutil
import sys


def main() -> int:
    print("=" * 64)
    print(" Interview Recording Audit System - environment check")
    print("=" * 64)

    print(f"Python : {sys.version.split()[0]}")

    # torch / CUDA --------------------------------------------------------- #
    try:
        import torch

        print(f"torch  : {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"CUDA available : {cuda}")
        if cuda:
            print(f"  GPU          : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA runtime : {torch.version.cuda}")
        else:
            print("  No NVIDIA CUDA GPU detected - falling back to CPU.")
    except ImportError:
        print("torch  : NOT INSTALLED  (pip install -r requirements.txt)")

    # ffmpeg --------------------------------------------------------------- #
    ff = shutil.which("ffmpeg")
    print(f"ffmpeg : {ff or 'NOT FOUND on PATH'}")

    # Resolved config ------------------------------------------------------ #
    try:
        import config

        print("-" * 64)
        print("Resolved configuration:")
        for k, v in config.summary().items():
            print(f"  {k:26} = {v}")
        device = config.DEVICE
    except Exception as exc:  # pragma: no cover - defensive
        print(f"\nCould not import config: {exc}")
        return 1

    # Verdict -------------------------------------------------------------- #
    print("-" * 64)
    if device == "cuda":
        print("READY: GPU detected. Full-speed audits enabled.")
    else:
        print("READY (CPU mode): everything runs, but a 60-min video will take")
        print("much longer than the 5-15 min GPU target. Use short clips for")
        print("testing, or run on an NVIDIA machine for full-length audits.")
        print("(Set AUDIT_DEVICE / WHISPER_MODEL in .env to override defaults.)")
    if not ff:
        print("\nWARNING: ffmpeg not found on PATH — ingest/audio extraction will fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
