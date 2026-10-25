"""
signals.py — Phase 5.

Merge transcription + diarization + vision into ONE chronological evidence log
plus summary metrics for the report.

- Extra-voice signal comes from diarization.extra_voice_windows (calibrated
  against the expected speaker count), NOT the raw speaker count.
- Expected-vs-detected speaker counts and transcription confidence are carried
  into summary_metrics so the report can frame things correctly.
- A SOFT, low-confidence "possible_external_assistance" signal is derived ONLY by
  correlating sustained off-center gaze with steady, reading-style speaking
  cadence. It is explicitly low-confidence and for human review — never proof.

Every cheating-type entry is a flag for a human reviewer with a confidence LEVEL
and a timestamp — never a definitive accusation.

Output: {"evidence_log": [{timestamp, timestamp_str, end, type, confidence,
                           detail}],
         "summary_metrics": {...}}

CLI test:  python -m pipeline.signals   (runs on cached/real clip outputs)
"""

from __future__ import annotations

import statistics

import config


def fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _entry(start: float, end: float, etype: str, confidence: str, detail: str) -> dict:
    return {
        "timestamp": round(start, 2),
        "timestamp_str": fmt_ts(start),
        "end": round(end, 2),
        "type": etype,
        "confidence": confidence,  # "low" | "medium" | "high"
        "detail": detail,
    }


def _derive_assistance(vision: dict, transcription: dict) -> list:
    """SOFT low-confidence signal: sustained off-center gaze + steady cadence.

    Conservative on purpose. Requires word timestamps and enough words in the
    window. Empty when there isn't enough gaze/word data (which is common and
    should NOT be read as 'all clear').
    """
    out: list = []
    gaze_windows = [
        e for e in vision.get("events", {}).get("gaze_off_center", [])
        if (e["end"] - e["start"]) >= config.ASSIST_MIN_OFFCENTER_SEC
    ]
    words = [w for s in transcription.get("segments", []) for w in s.get("words", [])]
    if not gaze_windows or not words:
        return out

    for gw in gaze_windows:
        ws = [w for w in words if w["start"] >= gw["start"] and w["end"] <= gw["end"]]
        if len(ws) < 6:
            continue
        gaps = [ws[i + 1]["start"] - ws[i]["end"] for i in range(len(ws) - 1)]
        gaps = [g for g in gaps if g >= 0]
        if len(gaps) < 4:
            continue
        mean_gap = statistics.mean(gaps)
        std_gap = statistics.pstdev(gaps)
        # Steady, reading-like: small and very even inter-word gaps.
        if mean_gap < 0.6 and std_gap < 0.25:
            out.append({
                "start": gw["start"], "end": gw["end"], "direction": gw["direction"],
            })
    return out


def build_signals(transcription: dict, diarization: dict | None, vision: dict) -> dict:
    evidence: list = []
    ev = vision.get("events", {})

    # --- Vision: phone (physical only; on-screen phones already filtered) ---
    for e in ev.get("phone_visible", []):
        lvl = "high" if e["confidence"] >= 0.6 else "medium"
        evidence.append(_entry(
            e["start"], e["end"], "phone_visible", lvl,
            f"A phone appears in frame {fmt_ts(e['start'])}–{fmt_ts(e['end'])} "
            f"(detector confidence {e['confidence']:.2f}).",
        ))

    # --- Vision: external devices (often the screen surface in screen-shares) ---
    for e in ev.get("external_device", []):
        note = " (likely the recorded screen itself in a screen-share)" if e["type"] == "monitor" else ""
        evidence.append(_entry(
            e["start"], e["end"], "external_device", "low",
            f"A {e['type']} is visible {fmt_ts(e['start'])}–{fmt_ts(e['end'])}"
            f"{note} (conf {e['confidence']:.2f}).",
        ))

    # --- Vision: extra physical person ---
    for e in ev.get("extra_person", []):
        evidence.append(_entry(
            e["start"], e["end"], "extra_person", "medium",
            f"More than one physical person detected (up to {e['max_faces']}) "
            f"{fmt_ts(e['start'])}–{fmt_ts(e['end'])}.",
        ))

    # --- Vision: gaze off-center (inherently low confidence) ---
    for e in ev.get("gaze_off_center", []):
        evidence.append(_entry(
            e["start"], e["end"], "gaze_off_center", "low",
            f"Head turned {e['direction']} {fmt_ts(e['start'])}–{fmt_ts(e['end'])} "
            "(gaze is unreliable from a single webcam; for human review).",
        ))

    # --- Diarization: extra voices beyond the expected count ---
    if diarization:
        exp = diarization.get("expected_speaker_count")
        for w in diarization.get("extra_voice_windows", []):
            evidence.append(_entry(
                w["start"], w["end"], "extra_voice", "medium",
                f"An extra voice beyond the expected {exp} speaker(s) is heard "
                f"{fmt_ts(w['start'])}–{fmt_ts(w['end'])}. Detects that an extra voice "
                "EXISTS; cannot identify who it is.",
            ))

    # --- Soft assistance signal ---
    assist = _derive_assistance(vision, transcription)
    for a in assist:
        evidence.append(_entry(
            a["start"], a["end"], "possible_external_assistance", "low",
            f"LOW CONFIDENCE / human review only: sustained off-center gaze "
            f"({a['direction']}) with steady, reading-like speech "
            f"{fmt_ts(a['start'])}–{fmt_ts(a['end'])}. NOT proof of anything.",
        ))

    evidence.sort(key=lambda x: x["timestamp"])

    summary_metrics = _build_summary(transcription, diarization, vision, evidence, assist)
    return {"evidence_log": evidence, "summary_metrics": summary_metrics}


def _build_summary(transcription, diarization, vision, evidence, assist) -> dict:
    vm = vision.get("metrics", {})
    conf = transcription.get("confidence", {})

    counts: dict = {}
    for e in evidence:
        counts[e["type"]] = counts.get(e["type"], 0) + 1

    if diarization:
        speakers = {
            "expected": diarization.get("expected_speaker_count"),
            "detected": diarization.get("distinct_speaker_count"),
            "primary_speaker": diarization.get("primary_speaker"),
            "extra_voice_windows": len(diarization.get("extra_voice_windows", [])),
            "note": diarization.get("note"),
        }
    else:
        speakers = {"note": "Diarization not run (no HF token / disabled)."}

    return {
        "duration_sec": transcription.get("duration"),
        "transcription": {
            "score": conf.get("score"),
            "low_confidence": conf.get("low_confidence"),
            "language": transcription.get("language"),
            "language_probability": transcription.get("language_probability"),
            "note": conf.get("note"),
        },
        "speakers": speakers,
        "vision": {
            "frames_sampled": vm.get("frames_sampled"),
            "frames_with_face": vm.get("frames_with_face"),
            "gaze_off_center_pct": vm.get("gaze_off_center_pct"),
            "gaze_breakdown": vm.get("gaze_breakdown"),
            "total_phone_seconds": vm.get("total_phone_seconds"),
            "external_device_seconds": vm.get("external_device_seconds"),
            "extra_person_windows": vm.get("extra_person_windows"),
            "max_persons": vm.get("max_persons"),
            "max_faces": vm.get("max_faces"),
            "screen_present_pct": vm.get("screen_present_pct"),
            "max_persons_on_screen": vm.get("max_persons_on_screen"),
            "on_screen_person_seconds": vm.get("on_screen_person_seconds"),
        },
        "evidence_counts": counts,
        "possible_external_assistance": {
            "windows": len(assist),
            "note": (
                "Soft, low-confidence signal for human review only. Absence here does "
                "NOT mean 'all clear' — it often just means insufficient gaze/word data."
            ),
        },
    }


if __name__ == "__main__":
    # Quick self-test against the cached real clip outputs (see scripts/run_clip.py),
    # falling back to a tiny synthetic example if no cache is present.
    import json
    from pathlib import Path

    cache = Path(config.WORK_DIR) / "clip_outputs.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        tr, di, vi = data["transcription"], data.get("diarization"), data["vision"]
    else:
        print("No cached clip outputs found; using a tiny synthetic example.")
        tr = {"duration": 120, "language": "en", "language_probability": 0.6,
              "segments": [], "confidence": {"score": 0.7, "low_confidence": False,
                                             "note": "synthetic"}}
        di = None
        vi = {"events": {"phone_visible": [{"start": 10, "end": 14, "confidence": 0.7}],
                         "external_device": [], "extra_person": [], "gaze_off_center": []},
              "metrics": {"frames_sampled": 60}}

    out = build_signals(tr, di, vi)
    print(f"\nEvidence log: {len(out['evidence_log'])} entries")
    for e in out["evidence_log"][:20]:
        print(f"  [{e['timestamp_str']}] {e['type']:28} ({e['confidence']:6}) {e['detail'][:80]}")
    print("\nSummary metrics:")
    print(json.dumps(out["summary_metrics"], indent=2, ensure_ascii=False)[:1600])
