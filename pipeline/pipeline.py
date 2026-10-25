"""
pipeline.py — orchestrates the full audit, running the three independent heavy
stages (transcribe / vision / diarize) CONCURRENTLY when config.PARALLEL_STAGES is
on. They share no state and their native inference releases the GIL, so wall-clock
approaches max(stage) instead of sum(stage) — the main performance lever for a
full-length recording.

run_pipeline() returns the raw stage outputs + merged signals + LLM report so the
UI (app.py) only has to render them.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from pipeline.ingest import ingest
from pipeline.report import CHECK_TO_TYPES, build_report
from pipeline.signals import build_signals
from pipeline.transcribe import transcribe
from pipeline.vision import analyze_video

VISION_CHECKS = {"phone", "extra_person", "gaze", "external_devices", "possible_llm"}
_EMPTY_VISION = {"events": {}, "metrics": {}, "per_frame": [], "sample_interval_sec": config.SAMPLE_INTERVAL_SEC}


def _safe_diarize(audio_path: str, expected: int):
    """Diarize, returning None (not raising) if HF access / pyannote is unavailable."""
    try:
        from pipeline.diarize import DiarizeError, diarize
        return diarize(audio_path, expected)
    except Exception as exc:  # DiarizeError or any load failure
        print(f"[pipeline] diarization skipped: {exc}")
        return None


def run_pipeline(source: str, is_url: bool, expected_speakers: int,
                 enabled_checks: dict | None = None, custom_prompt: str = "",
                 progress=None) -> dict:
    enabled_checks = enabled_checks or {k: True for k in CHECK_TO_TYPES}

    def emit(frac, desc):
        if progress is not None:
            progress(frac, desc=desc)

    want_vision = any(enabled_checks.get(k) for k in VISION_CHECKS) or not any(enabled_checks.values())
    want_diar = enabled_checks.get("extra_voices", False) or not any(enabled_checks.values())

    emit(0.05, "Ingesting (download + audio extraction)...")
    ing = ingest(source, is_url=is_url)

    results: dict = {"transcribe": None, "vision": _EMPTY_VISION, "diarize": None}

    # Build the work set.
    jobs = {"transcribe": lambda: transcribe(ing["audio_path"])}
    if want_vision:
        jobs["vision"] = lambda: analyze_video(ing["video_path"], fps=ing["fps"])
    if want_diar:
        jobs["diarize"] = lambda: _safe_diarize(ing["audio_path"], expected_speakers)

    stage_label = {
        "transcribe": f"Transcribing (Whisper '{config.WHISPER_MODEL}' on {config.DEVICE})",
        "vision": "Analysing video (YOLO + MediaPipe)",
        "diarize": "Diarizing (pyannote)",
    }

    if config.PARALLEL_STAGES and len(jobs) > 1:
        emit(0.15, "Analysing in parallel: " + ", ".join(jobs) + " ...")
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = {ex.submit(fn): name for name, fn in jobs.items()}
            done = 0
            for fut in as_completed(futures):
                name = futures[fut]
                results[name] = fut.result()
                done += 1
                emit(0.15 + 0.6 * done / len(jobs), f"{stage_label[name]} done ({done}/{len(jobs)})")
    else:
        frac = 0.15
        for name, fn in jobs.items():
            emit(frac, stage_label[name] + " ...")
            results[name] = fn()
            frac += 0.6 / max(1, len(jobs))

    emit(0.85, "Merging evidence...")
    sig = build_signals(results["transcribe"], results["diarize"], results["vision"])

    emit(0.92, f"Writing report ({config.LLM_BACKEND})...")
    report = build_report(
        results["transcribe"]["full_text"], sig["evidence_log"],
        sig["summary_metrics"], custom_prompt or "", enabled_checks,
    )
    emit(1.0, "Done")

    return {
        "ingest": ing,
        "transcription": results["transcribe"],
        "vision": results["vision"],
        "diarization": results["diarize"],
        "signals": sig,
        "report": report,
    }
