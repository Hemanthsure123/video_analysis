"""
audio_chunks.py — Gatekeeper test fixture (isolated; not used by the pipeline).

Split a recording into fixed-length windows so a long interview can be
transcribed chunk by chunk (and a failed chunk retried) instead of in one pass.
"""

from __future__ import annotations


def chunk_windows(duration_sec: float, chunk_sec: float = 60.0) -> list[tuple[float, float]]:
    """Return consecutive (start, end) windows that cover the whole recording.

    Every window is chunk_sec long except possibly the last one, which is
    clipped to end at duration_sec. An empty recording yields no windows.
    """
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    if duration_sec <= 0:
        return []

    n_chunks = int(duration_sec // chunk_sec)
    return [
        (i * chunk_sec, min((i + 1) * chunk_sec, duration_sec))
        for i in range(n_chunks)
    ]
