"""
report.py — Phase 6.

Calls the configured LLM to turn the transcript + evidence log + summary metrics
into three sections:
  a) GENUINENESS ASSESSMENT — narrative + structured flags (timestamp / detected /
     confidence), framed for a HUMAN reviewer. Extra-voice flags are phrased
     relative to the expected speaker count and never claim who it was.
  b) Q&A EVALUATION — inferred questions, per-answer summary + ratings
     (correctness / depth / clarity) + a supporting quote; overall rating.
     Caveated up front when transcription confidence is low.
  c) CUSTOM PROMPT RESPONSE — answers the user's prompt from transcript + evidence.

Rules enforced in the prompt: use only facts present in the transcript / evidence
log; never invent quotes; if a question wasn't really answered, say so; keep the
human-review framing for all genuineness flags.

Output: a structured dict the UI renders into sections + clickable timestamps.

CLI test:  python -m pipeline.report   (uses cached clip outputs + a sample prompt)
"""

from __future__ import annotations

import json

import config
from llm.base import LLMError, get_client

# UI checkbox key -> evidence types it controls.
CHECK_TO_TYPES = {
    "phone": ["phone_visible"],
    "extra_person": ["extra_person"],
    "gaze": ["gaze_off_center"],
    "extra_voices": ["extra_voice"],
    "external_devices": ["external_device"],
    "possible_llm": ["possible_external_assistance"],
}

# Leave headroom for the system prompt, evidence, and the model's JSON output.
_MAX_TRANSCRIPT_CHARS = max(20000, (config.LLM_CONTEXT_TOKENS - 8000) * 4)

_SYSTEM = """\
You assist a HUMAN REVIEWER auditing an interview recording. You are an ASSISTANT, \
NOT a judge. You never make definitive accusations of cheating.

Hard rules:
- Use ONLY facts present in the provided transcript and evidence log. Do not invent \
events, people, or quotes. Every supporting_quote MUST be copied verbatim from the \
transcript.
- Every genuineness item is a "flag for human review" with a confidence level \
(low/medium/high) and a timestamp taken from the evidence log. Frame them as things \
for the reviewer to verify, not conclusions.
- Gaze and "possible external assistance" are inherently unreliable from a single \
webcam; always label them low confidence and explicitly for human review.
- For extra voices, phrase relative to the EXPECTED speaker count (e.g. "an extra \
voice beyond the expected N was detected at ..."). NEVER claim who it was; an extra \
voice detector cannot identify identity.
- If transcription confidence is low, say so up front in the Q&A section and note \
that scoring reliability is reduced.
- If a question was not actually answered, say so plainly instead of inventing a score.
- Only include genuineness flags for the check categories the user enabled.
- Reflect input quality honestly; do not imply more certainty than the data supports.

Respond with a SINGLE JSON object, no markdown, matching exactly:
{
  "genuineness_assessment": {
    "narrative": "2-5 sentence overview framed for a human reviewer",
    "input_quality_caveat": "1-2 sentences on how recording quality bounds reliability",
    "flags": [
      {"timestamp": "MM:SS", "seconds": <number>, "type": "<evidence type>",
       "detected": "what was detected", "confidence": "low|medium|high",
       "for_reviewer": "what the human should verify"}
    ]
  },
  "qa_evaluation": {
    "transcription_confidence_caveat": "<string or null>",
    "questions": [
      {"question": "inferred question", "answered": <true|false>,
       "answer_summary": "summary grounded in transcript",
       "ratings": {"correctness": <1-5>, "depth": <1-5>, "clarity": <1-5>},
       "supporting_quote": "verbatim transcript quote"}
    ],
    "overall": {"rating": "<1-5>", "justification": "brief, grounded justification"}
  },
  "custom_prompt_response": "direct answer to the user's custom prompt, or a note if empty"
}
"""


def _cap_transcript(text: str) -> tuple[str, bool]:
    if len(text) <= _MAX_TRANSCRIPT_CHARS:
        return text, False
    head = _MAX_TRANSCRIPT_CHARS // 2
    tail = _MAX_TRANSCRIPT_CHARS - head
    return text[:head] + "\n...[transcript truncated for length]...\n" + text[-tail:], True


def _user_payload(transcript, evidence, summary, custom_prompt, enabled_checks, truncated):
    enabled = [k for k, v in enabled_checks.items() if v] or ["(all)"]
    parts = [
        "## CONTEXT (summary metrics)",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "",
        f"## ENABLED GENUINENESS CHECKS: {', '.join(enabled)}",
        "",
        "## EVIDENCE LOG (the only detections you may cite for genuineness flags)",
        json.dumps(evidence, ensure_ascii=False, indent=2) if evidence
        else "(no detections for the enabled checks)",
        "",
        f"## USER'S CUSTOM PROMPT\n{custom_prompt.strip() or '(none provided)'}",
        "",
        "## TRANSCRIPT" + (" (truncated)" if truncated else ""),
        transcript or "(empty transcript)",
    ]
    return "\n".join(parts)


def build_report(full_text: str, evidence_log: list, summary_metrics: dict,
                 custom_prompt: str = "", enabled_checks: dict | None = None) -> dict:
    enabled_checks = enabled_checks or {k: True for k in CHECK_TO_TYPES}

    # Filter evidence to the enabled check categories (all if none selected).
    selected_types = set()
    for key, on in enabled_checks.items():
        if on:
            selected_types.update(CHECK_TO_TYPES.get(key, []))
    if selected_types:
        evidence = [e for e in evidence_log if e["type"] in selected_types]
    else:
        evidence = list(evidence_log)

    transcript, truncated = _cap_transcript(full_text or "")
    user = _user_payload(transcript, evidence, summary_metrics, custom_prompt or "",
                         enabled_checks, truncated)

    client = get_client()
    try:
        raw = client.complete(_SYSTEM, user, json_mode=True, temperature=0.2)
    except LLMError as e:
        return {"error": str(e)}

    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        report = {"error": "Model did not return valid JSON.", "raw": raw}

    low_conf = bool(summary_metrics.get("transcription", {}).get("low_confidence"))
    model_by_backend = {
        "openai": config.OPENAI_MODEL,
        "gemini": config.GEMINI_MODEL,
        "groq": config.GROQ_MODEL,
    }
    report["_meta"] = {
        "model": model_by_backend.get(config.LLM_BACKEND, config.OPENAI_MODEL),
        "backend": config.LLM_BACKEND,
        "transcription_low_confidence": low_conf,
        "evidence_used": len(evidence),
        "transcript_truncated": truncated,
        "enabled_checks": enabled_checks,
    }
    return report


if __name__ == "__main__":
    from pathlib import Path

    from pipeline.signals import build_signals

    cache = Path(config.WORK_DIR) / "clip_outputs.json"
    if not cache.exists():
        print("No cached outputs. Run: python scripts/run_clip.py first.")
        raise SystemExit(1)

    data = json.loads(cache.read_text(encoding="utf-8"))
    sig = build_signals(data["transcription"], data.get("diarization"), data["vision"])

    report = build_report(
        data["transcription"]["full_text"],
        sig["evidence_log"],
        sig["summary_metrics"],
        custom_prompt="Summarize the candidate's communication style in 2 sentences.",
        enabled_checks={k: True for k in CHECK_TO_TYPES},
    )

    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
