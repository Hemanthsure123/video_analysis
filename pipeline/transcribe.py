"""
transcribe.py — Phase 2 (delegates to the configured transcription backend).

Input : audio_path.
Steps : faster-whisper (large-v3 on GPU / smaller on CPU) -> segments with
        word-level timestamps + full text; aggregate avg_logprob / no_speech_prob
        into an overall confidence + a human-readable low_confidence note.
Output: {"full_text": str,
         "segments": [{"start", "end", "text", "words": [...], ...}],
         "confidence": {"score": float, "low_confidence": bool, "note": str},
         "language": str, "language_probability": float, "duration": float}

The backend is chosen by config.TRANSCRIPTION_BACKEND so Sarvam's API can be
swapped in later without touching the rest of the pipeline.

CLI test:  python -m pipeline.transcribe "<audio.wav>"
"""

from __future__ import annotations

import config


def transcribe(audio_path: str) -> dict:
    backend = config.TRANSCRIPTION_BACKEND
    if backend == "whisper_local":
        from transcription.whisper_local import transcribe as _impl
    elif backend == "sarvam":
        from transcription.sarvam_client import transcribe as _impl  # type: ignore
    else:
        raise ValueError(f"Unknown TRANSCRIPTION_BACKEND: {backend!r}")
    return _impl(audio_path)


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print('Usage: python -m pipeline.transcribe "<audio.wav>"')
        raise SystemExit(2)

    t0 = time.time()
    result = transcribe(sys.argv[1])
    elapsed = time.time() - t0

    conf = result["confidence"]
    print("\n" + "=" * 64)
    print(f"Transcribed in {elapsed:.1f}s  | audio {result['duration']:.1f}s "
          f"| {len(result['segments'])} segments | {len(result['full_text'])} chars")
    print(f"Language: {result['language']} (p={result['language_probability']:.2f})")
    print(f"Confidence: score={conf['score']:.2f}  low_confidence={conf['low_confidence']}")
    print(f"Note: {conf['note']}")
    print("-" * 64)
    print("First ~600 chars of transcript:")
    print(result["full_text"][:600])
    print("-" * 64)
    print("First 3 segments (with first words):")
    for s in result["segments"][:3]:
        w = s["words"][:6]
        print(f"  [{s['start']:.2f}-{s['end']:.2f}] {s['text'][:80]}")
        if w:
            print("      words:", ", ".join(f"{x['word']!r}@{x['start']:.2f}" for x in w))
