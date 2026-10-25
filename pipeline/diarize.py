"""
diarize.py — Phase 4.

Input : audio_path, expected_speaker_count.
Steps : pyannote -> speaker turns; identify PRIMARY speaker (most talk time);
        count distinct speakers. If distinct > expected, the speakers beyond the
        expected count (the least-talking ones) are treated as "extra" and their
        speech becomes extra_voice_windows — the suspicious signal.

        IMPORTANT: this detects that an extra voice EXISTS; it CANNOT identify
        who it is. A legitimate interviewer within the expected count is never
        flagged.

Output: {"turns": [{"start", "end", "speaker_label"}],
         "primary_speaker": str,
         "distinct_speaker_count": int,
         "expected_speaker_count": int,
         "extra_voice_windows": [{"start", "end"}],
         "speaker_talk_time": {label: seconds},
         "note": str}

Requires HF_TOKEN in .env and one-time acceptance of the pyannote model terms
(see README). CLI test:  python -m pipeline.diarize "<audio.wav>" [expected]
"""

from __future__ import annotations

import config

_PIPELINE = None


class DiarizeError(Exception):
    """User-facing diarization failure (missing token, terms not accepted, etc.)."""


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not config.HF_TOKEN:
            raise DiarizeError(
                "HF_TOKEN is not set. Add it to .env and accept the pyannote model "
                "terms (see README) to enable diarization."
            )
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizeError(
                "pyannote.audio is not installed (pip install pyannote.audio)."
            ) from exc

        print(f"[diarize] loading {config.PYANNOTE_PIPELINE} ...")
        pipe = None
        last_exc = None
        # The auth-token kwarg name changed across pyannote/huggingface versions.
        for kw in ("token", "use_auth_token"):
            try:
                pipe = Pipeline.from_pretrained(config.PYANNOTE_PIPELINE, **{kw: config.HF_TOKEN})
                break
            except TypeError as exc:
                last_exc = exc  # wrong kwarg name; try the other
                continue
            except Exception as exc:
                last_exc = exc
                break
        if pipe is None:
            raise DiarizeError(
                f"Could not load the pyannote pipeline: {last_exc}\n"
                f"Check that HF_TOKEN is valid (Read access to gated repos) and that you "
                f"accepted the terms for '{config.PYANNOTE_PIPELINE}' "
                "AND pyannote/segmentation-3.0 (pyannote 4.x also gates "
                "pyannote/speaker-diarization-community-1)."
            )

        if pipe is None:
            raise DiarizeError(
                "pyannote returned no pipeline — usually means the model terms were "
                "not accepted for your HF account."
            )

        if config.ON_GPU:
            try:
                import torch
                pipe.to(torch.device("cuda"))
            except Exception as exc:  # pragma: no cover
                print(f"[diarize] could not move pipeline to CUDA ({exc}); using CPU.")
        _PIPELINE = pipe
    return _PIPELINE


def _load_waveform(audio_path: str):
    """Load a PCM WAV into an in-memory (waveform, sample_rate) dict for pyannote.

    Avoids pyannote 4.x's torchcodec file decoding, which needs FFmpeg shared DLLs
    that aren't present on this Windows setup. Our audio is always 16 kHz mono PCM.
    """
    import wave

    import numpy as np
    import torch

    with wave.open(audio_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        # Not 16-bit PCM — let pyannote try the path directly as a fallback.
        return audio_path

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).T  # (channel, time)
    else:
        data = data[None, :]
    return {"waveform": torch.from_numpy(data), "sample_rate": sr}


def _merge_windows(windows: list, gap: float) -> list:
    """Merge [{start,end}] windows that touch or sit within `gap` seconds."""
    out: list = []
    for w in sorted(windows, key=lambda x: x["start"]):
        if out and w["start"] - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], w["end"])
        else:
            out.append({"start": w["start"], "end": w["end"]})
    return out


def diarize(audio_path: str, expected_speaker_count: int | None = None) -> dict:
    expected = int(expected_speaker_count or config.EXPECTED_SPEAKERS_DEFAULT)
    pipeline = _get_pipeline()

    result = pipeline(_load_waveform(audio_path))
    # pyannote 4.x returns DiarizeOutput(.speaker_diarization); 3.x returns an Annotation.
    annotation = getattr(result, "speaker_diarization", result)

    turns: list = []
    talk_time: dict = {}
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker_label": speaker,
        })
        talk_time[speaker] = talk_time.get(speaker, 0.0) + (segment.end - segment.start)

    talk_time = {k: round(v, 2) for k, v in talk_time.items()}
    distinct = len(talk_time)

    # Rank speakers by talk time; the top `expected` are allowed, the rest extra.
    ranked = sorted(talk_time, key=lambda k: talk_time[k], reverse=True)
    primary = ranked[0] if ranked else None
    extra_speakers = set(ranked[expected:]) if distinct > expected else set()

    extra_windows = _merge_windows(
        [{"start": t["start"], "end": t["end"]} for t in turns
         if t["speaker_label"] in extra_speakers],
        gap=config.EVENT_MERGE_GAP_SEC,
    )
    # Drop sub-threshold windows (diarization over-segmentation artifacts).
    extra_windows = [
        w for w in extra_windows
        if (w["end"] - w["start"]) >= config.DIARIZE_MIN_EXTRA_VOICE_SEC
    ]

    if distinct <= expected:
        note = (
            f"Detected {distinct} distinct voice(s), within the expected {expected}. "
            "No extra-voice signal."
        )
    else:
        note = (
            f"Detected {distinct} distinct voices, MORE than the expected {expected}. "
            f"Speech from the {len(extra_speakers)} extra voice(s) is flagged for human "
            "review. This indicates an extra voice EXISTS but cannot identify who it is."
        )

    return {
        "turns": turns,
        "primary_speaker": primary,
        "distinct_speaker_count": distinct,
        "expected_speaker_count": expected,
        "extra_voice_windows": extra_windows,
        "speaker_talk_time": talk_time,
        "note": note,
    }


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print('Usage: python -m pipeline.diarize "<audio.wav>" [expected_speakers]')
        raise SystemExit(2)

    expected = int(sys.argv[2]) if len(sys.argv) > 2 else config.EXPECTED_SPEAKERS_DEFAULT
    try:
        t0 = time.time()
        out = diarize(sys.argv[1], expected)
        elapsed = time.time() - t0
    except DiarizeError as e:
        print(f"DIARIZE ERROR: {e}")
        raise SystemExit(1)

    print("\n" + "=" * 64)
    print(f"Diarized in {elapsed:.1f}s")
    print(f"distinct speakers: {out['distinct_speaker_count']} (expected {out['expected_speaker_count']})")
    print(f"primary speaker  : {out['primary_speaker']}")
    print(f"talk time        : {out['speaker_talk_time']}")
    print(f"extra-voice windows: {len(out['extra_voice_windows'])}")
    for w in out["extra_voice_windows"][:8]:
        print(f"    {w}")
    print(f"note: {out['note']}")
    print(f"total turns: {len(out['turns'])} (first 5)")
    for t in out["turns"][:5]:
        print(f"    {t}")
