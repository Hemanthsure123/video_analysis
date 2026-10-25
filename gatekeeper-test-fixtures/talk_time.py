"""
talk_time.py — Gatekeeper test fixture (isolated; not used by the pipeline).

Summarise how long each speaker talked from diarization turns shaped like
pipeline.diarize output: [{"start": 1.2, "end": 4.8, "speaker_label": "SPEAKER_00"}, ...]
Adds share-of-speech percentages, turn counts and the dominant speaker on top
of the raw per-label seconds that diarize already reports.
"""

from __future__ import annotations


def talk_time_summary(turns: list[dict], min_turn_sec: float = 0.3) -> dict:
    """Return per-speaker talk time, share of speech and turn counts.

    Turns shorter than min_turn_sec (diarization blips) and turns with
    end <= start are ignored. Back-to-back turns by the same speaker count as
    one turn. Shares are percentages of total speech, rounded to one decimal.
    The dominant speaker is whoever talked longest; ties go to the label that
    sorts first.

    Output: {"total_sec": float,
             "speakers": {label: {"seconds", "share_pct", "turns"}},
             "dominant_speaker": str | None}
    """
    seconds: dict[str, float] = {}
    turn_counts: dict[str, int] = {}
    prev_label = None
    for t in sorted(turns, key=lambda x: x["start"]):
        dur = t["end"] - t["start"]
        if dur <= 0 or dur < min_turn_sec:
            continue
        label = t["speaker_label"]
        seconds[label] = seconds.get(label, 0.0) + dur
        if label != prev_label:
            turn_counts[label] = turn_counts.get(label, 0) + 1
            prev_label = label

    total = sum(seconds.values(), 0.0)
    speakers = {
        label: {
            "seconds": round(sec, 2),
            "share_pct": round(100.0 * sec / total, 1) if total else 0.0,
            "turns": turn_counts[label],
        }
        for label, sec in sorted(seconds.items())
    }
    dominant = min(seconds, key=lambda s: (-seconds[s], s)) if seconds else None
    return {"total_sec": round(total, 2), "speakers": speakers, "dominant_speaker": dominant}
