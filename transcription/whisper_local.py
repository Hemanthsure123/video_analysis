"""
whisper_local.py — default transcription backend (faster-whisper). Phase 2.

Thin module so transcription/sarvam_client.py can be swapped in later if accent
accuracy is poor. Loads the model once per (model, device, compute_type) on first
call, on the configured device.

Returns the contract documented in pipeline/transcribe.py:
    {"full_text", "segments": [{start, end, text, avg_logprob, no_speech_prob,
      words: [{start, end, word, probability}]}],
     "confidence": {"score", "low_confidence", "note"},
     "language", "language_probability", "duration"}
"""

from __future__ import annotations

import math

import config

# Cache loaded models so repeated calls (e.g. from the Gradio UI) don't reload.
_MODEL_CACHE: dict = {}


def _get_model():
    key = (config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    model = _MODEL_CACHE.get(key)
    if model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is not installed (pip install faster-whisper)."
            ) from exc
        print(
            f"[transcribe] loading Whisper '{config.WHISPER_MODEL}' on "
            f"{config.WHISPER_DEVICE} ({config.WHISPER_COMPUTE_TYPE})..."
        )
        model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            cpu_threads=config.WHISPER_CPU_THREADS,
        )
        _MODEL_CACHE[key] = model
    return model


def _aggregate_confidence(segments: list, language: str, language_probability: float) -> dict:
    """Roll Whisper's per-segment avg_logprob / no_speech_prob into one signal.

    avg_logprob is a per-token log-probability (~[-1, 0]); exp() of its
    duration-weighted mean gives an intuitive 0..1 "mean token confidence".
    """
    if not segments:
        return {
            "score": 0.0,
            "low_confidence": True,
            "note": "No speech detected in the audio; transcription is empty.",
        }

    total_dur = 0.0
    w_logprob = 0.0
    w_nospeech = 0.0
    for s in segments:
        dur = max(s["end"] - s["start"], 1e-3)
        total_dur += dur
        w_logprob += s["avg_logprob"] * dur
        w_nospeech += s["no_speech_prob"] * dur

    mean_logprob = w_logprob / total_dur
    mean_nospeech = w_nospeech / total_dur
    score = max(0.0, min(1.0, math.exp(mean_logprob)))

    reasons = []
    if mean_logprob < config.TRANSCRIPTION_LOW_CONF_AVG_LOGPROB:
        reasons.append(
            f"avg log-prob {mean_logprob:.2f} is below {config.TRANSCRIPTION_LOW_CONF_AVG_LOGPROB:.2f}"
        )
    if mean_nospeech > config.TRANSCRIPTION_HIGH_NO_SPEECH_PROB:
        reasons.append(
            f"high no-speech probability ({mean_nospeech:.2f})"
        )
    low = bool(reasons)

    note = (
        f"Detected language '{language}' (p={language_probability:.2f}). "
        f"Mean token confidence ~{score:.0%}."
    )
    if low:
        note += (
            " LOW CONFIDENCE: " + "; ".join(reasons) + ". This usually means a noisy "
            "recording, a weak mic, heavy accent, or crosstalk — treat the transcript "
            "and any scoring derived from it with caution."
        )
    return {
        "score": round(score, 4),
        "low_confidence": low,
        "note": note,
        "mean_avg_logprob": round(mean_logprob, 4),
        "mean_no_speech_prob": round(mean_nospeech, 4),
    }


def transcribe(audio_path: str) -> dict:
    model = _get_model()

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=config.WHISPER_BEAM_SIZE,
        language=config.WHISPER_LANGUAGE,
        word_timestamps=True,
        vad_filter=config.WHISPER_VAD_FILTER,
    )

    segments: list = []
    for seg in segments_gen:  # generator: transcription happens as we iterate
        words = []
        for w in seg.words or []:
            words.append(
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word,
                    "probability": round(w.probability, 4),
                }
            )
        segments.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "words": words,
            }
        )

    full_text = " ".join(s["text"] for s in segments).strip()
    confidence = _aggregate_confidence(
        segments, info.language, info.language_probability
    )

    return {
        "full_text": full_text,
        "segments": segments,
        "confidence": confidence,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
    }
