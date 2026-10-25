"""
app.py — Gradio UI for the Interview Recording Audit System (Phase 7).

Inputs: Drive link / file upload, custom prompt, expected-speakers control, and
genuineness checkboxes. Runs the full pipeline with a live progress indicator and
renders Genuineness / Q&A / Custom-prompt / Raw-evidence sections. Every timestamp
is a clickable link that seeks the embedded video player.

    python app.py            # http://127.0.0.1:7860
"""

from __future__ import annotations

import html
import traceback

import gradio as gr

import config
from pipeline.ingest import IngestError
from pipeline.pipeline import run_pipeline
from pipeline.signals import fmt_ts

# UI checkbox label -> internal check key.
CHECK_LABELS = {
    "Phone use": "phone",
    "Extra person": "extra_person",
    "Gaze direction": "gaze",
    "Extra voices": "extra_voices",
    "External devices": "external_devices",
    "Possible LLM/tool usage": "possible_llm",
}

# Injected into <head>: a global seek() + delegated click handler so timestamp
# links (added dynamically with the report) seek the embedded <video>.
HEAD_JS = """
<script>
function seekAudit(t){
  const v = document.querySelector('#audit_video video');
  if(v){ v.currentTime = t; v.play().catch(()=>{}); v.scrollIntoView({behavior:'smooth', block:'center'}); }
}
document.addEventListener('click', function(e){
  const el = e.target.closest('.ts-link');
  if(el){ e.preventDefault(); seekAudit(parseFloat(el.dataset.t)); }
});
</script>
<style>
.ts-link{color:#2563eb;cursor:pointer;font-weight:600;text-decoration:none;font-variant-numeric:tabular-nums;}
.ts-link:hover{text-decoration:underline;}
.conf-low{color:#6b7280;} .conf-medium{color:#d97706;} .conf-high{color:#dc2626;font-weight:700;}
.flag-row{padding:6px 0;border-bottom:1px solid #eee;}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.78em;border:1px solid currentColor;}
.qa-card{border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;margin:8px 0;background:#fafafa;}
.rating{display:inline-block;margin-right:12px;font-size:.9em;}
.muted{color:#6b7280;font-size:.9em;}
</style>
"""

INTRO = "# Interview Recording Audit\nAn assistant for a **human reviewer** — every flag has a confidence level and a clickable timestamp, never a verdict."

QUALITY_NOTE = (
    "Reliability depends on recording quality: clear mic, quiet room, good front "
    "lighting, webcam near eye level, candidate fully in frame. Off-camera devices/"
    "people cannot be detected; gaze and 'possible tool use' are low-confidence."
)


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _ts_link(seconds, label=None) -> str:
    label = label or fmt_ts(seconds)
    return f'<a href="#" class="ts-link" data-t="{float(seconds):.2f}">{_esc(label)}</a>'


# --------------------------------------------------------------------------- #
# HTML renderers
# --------------------------------------------------------------------------- #
def _confidence_html(summary: dict) -> str:
    t = summary.get("transcription", {})
    score = t.get("score")
    low = t.get("low_confidence")
    pct = f"{score:.0%}" if isinstance(score, (int, float)) else "n/a"
    badge = "🟠 LOW" if low else "🟢 OK"
    sp = summary.get("speakers", {})
    spk = (f"detected {sp.get('detected')} / expected {sp.get('expected')}"
           if sp.get("detected") is not None else _esc(sp.get("note", "")))
    return (
        f"<h3>Transcription confidence: {pct} ({badge})</h3>"
        f"<div class='muted'>{_esc(t.get('note',''))}</div>"
        f"<div class='muted'>Language: {_esc(t.get('language'))} "
        f"(p={_esc(t.get('language_probability'))}) &nbsp;|&nbsp; Speakers: {spk} "
        f"&nbsp;|&nbsp; Duration: {_esc(summary.get('duration_sec'))}s</div>"
    )


def _genuineness_html(report: dict) -> str:
    g = report.get("genuineness_assessment", {})
    out = ["<h3>Genuineness assessment</h3>"]
    if g.get("narrative"):
        out.append(f"<p>{_esc(g['narrative'])}</p>")
    if g.get("input_quality_caveat"):
        out.append(f"<p class='muted'><em>{_esc(g['input_quality_caveat'])}</em></p>")
    flags = g.get("flags", []) or []
    if not flags:
        out.append("<p class='muted'>No flags raised for the enabled checks. "
                   "(Absence is not proof of genuineness — see input-quality caveat.)</p>")
    else:
        out.append(f"<p><strong>{len(flags)} flag(s) for human review:</strong></p>")
        for f in flags:
            conf = str(f.get("confidence", "low")).lower()
            secs = f.get("seconds", 0) or 0
            ts = _ts_link(secs, f.get("timestamp"))
            out.append(
                f"<div class='flag-row'>{ts} "
                f"<span class='badge conf-{_esc(conf)}'>{_esc(conf)}</span> "
                f"<strong>{_esc(f.get('type'))}</strong> — {_esc(f.get('detected'))}"
                f"<br><span class='muted'>For reviewer: {_esc(f.get('for_reviewer'))}</span></div>"
            )
    return "".join(out)


def _qa_html(report: dict) -> str:
    qa = report.get("qa_evaluation", {})
    out = ["<h3>Q&amp;A evaluation</h3>"]
    if qa.get("transcription_confidence_caveat"):
        out.append(f"<p class='muted'><strong>Caveat:</strong> "
                   f"{_esc(qa['transcription_confidence_caveat'])}</p>")
    questions = qa.get("questions", []) or []
    if not questions:
        out.append("<p class='muted'>No distinct questions could be inferred from the transcript.</p>")
    for i, q in enumerate(questions, 1):
        r = q.get("ratings", {}) or {}
        answered = q.get("answered")
        ans_badge = "" if answered is None else (
            "<span class='badge conf-low'>not answered</span>" if not answered else "")
        ratings = " ".join(
            f"<span class='rating'>{k}: <strong>{_esc(r.get(k) if r.get(k) is not None else '–')}</strong>/5</span>"
            for k in ("correctness", "depth", "clarity")
        )
        quote = q.get("supporting_quote")
        out.append(
            f"<div class='qa-card'><div><strong>Q{i}.</strong> {_esc(q.get('question'))} {ans_badge}</div>"
            f"<div>{_esc(q.get('answer_summary'))}</div>"
            f"<div>{ratings}</div>"
            + (f"<div class='muted'>“{_esc(quote)}”</div>" if quote else "")
            + "</div>"
        )
    overall = qa.get("overall", {})
    if overall:
        out.append(f"<p><strong>Overall rating: {_esc(overall.get('rating'))}/5</strong> — "
                   f"{_esc(overall.get('justification'))}</p>")
    return "".join(out)


def _custom_html(report: dict, custom_prompt: str) -> str:
    resp = report.get("custom_prompt_response")
    if not (custom_prompt or "").strip():
        return "<h3>Custom prompt</h3><p class='muted'>No custom prompt provided.</p>"
    return (f"<h3>Custom prompt answer</h3><p class='muted'>Prompt: {_esc(custom_prompt)}</p>"
            f"<p>{_esc(resp)}</p>")


def _evidence_html(evidence: list) -> str:
    if not evidence:
        return "<p class='muted'>No evidence entries.</p>"
    rows = ["<table style='width:100%;border-collapse:collapse;'>"
            "<tr><th style='text-align:left'>Time</th><th style='text-align:left'>Type</th>"
            "<th style='text-align:left'>Conf</th><th style='text-align:left'>Detail</th></tr>"]
    for e in evidence:
        conf = str(e.get("confidence", "")).lower()
        rows.append(
            f"<tr><td>{_ts_link(e['timestamp'], e['timestamp_str'])}</td>"
            f"<td>{_esc(e['type'])}</td>"
            f"<td><span class='conf-{_esc(conf)}'>{_esc(conf)}</span></td>"
            f"<td>{_esc(e['detail'])}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


# --------------------------------------------------------------------------- #
# Pipeline handler
# --------------------------------------------------------------------------- #
def run_audit(drive_url, upload_path, custom_prompt, speaker_preset, custom_speakers,
              selected_checks, progress=gr.Progress()):
    blank = (None, "", "", "", "", "")
    drive_url = (drive_url or "").strip()
    if not drive_url and not upload_path:
        return (*blank, "⚠️ Provide a Drive link or upload a file.")

    # Resolve expected speakers + enabled checks.
    if speaker_preset and speaker_preset.startswith("Solo"):
        expected = 1
    elif speaker_preset and speaker_preset.startswith("Live"):
        expected = 2
    else:
        expected = int(custom_speakers or config.EXPECTED_SPEAKERS_DEFAULT)
    checks = {CHECK_LABELS[l]: True for l in (selected_checks or [])}

    source, is_url = (drive_url, True) if drive_url else (upload_path, False)

    try:
        out = run_pipeline(source, is_url, expected, checks, custom_prompt or "",
                           progress=lambda f, desc="": progress(f, desc=desc))
        sig, report, ing = out["signals"], out["report"], out["ingest"]

        if report.get("error"):
            err = f"⚠️ Report LLM error: {report['error']}"
            return (ing["video_path"], _confidence_html(sig["summary_metrics"]),
                    f"<p>{_esc(err)}</p>", "", "", _evidence_html(sig["evidence_log"]), err)

        return (
            ing["video_path"],
            _confidence_html(sig["summary_metrics"]),
            _genuineness_html(report),
            _qa_html(report),
            _custom_html(report, custom_prompt or ""),
            _evidence_html(sig["evidence_log"]),
            f"✅ Done. Diarization: {'yes' if out['diarization'] else 'skipped'} | "
            f"Vision: {'yes' if out['vision'].get('per_frame') else 'skipped'} | "
            f"Report model: {report.get('_meta', {}).get('model')}",
        )
    except IngestError as e:
        return (*blank, f"⚠️ Ingest error: {e}")
    except Exception as e:
        traceback.print_exc()
        return (*blank, f"⚠️ Error: {e}")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Interview Recording Audit") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                drive_url = gr.Textbox(label="Google Drive link (anyone-with-link)",
                                       placeholder="https://drive.google.com/file/d/.../view")
                upload = gr.File(label="...or upload a video file", type="filepath")
                speaker_preset = gr.Radio(
                    ["Solo screen recording (1)", "Live interview with interviewer (2)",
                     "Custom (use number)"],
                    value="Live interview with interviewer (2)",
                    label="Expected speakers",
                )
                custom_speakers = gr.Number(value=2, precision=0, label="Custom expected count")
                checks = gr.CheckboxGroup(
                    list(CHECK_LABELS.keys()), value=list(CHECK_LABELS.keys()),
                    label="Genuineness checks to run",
                )
                custom_prompt = gr.Textbox(label="Custom prompt (optional)", lines=2,
                                           placeholder="e.g. Did the candidate explain recursion correctly?")
                run_btn = gr.Button("Run Audit", variant="primary")
                gr.Markdown(f"<sub>{QUALITY_NOTE}</sub>")
            with gr.Column(scale=1):
                video = gr.Video(label="Recording (timestamps below seek here)",
                                 elem_id="audit_video")
                status = gr.Markdown("")
                confidence = gr.HTML()

        genuineness = gr.HTML()
        qa = gr.HTML()
        custom_answer = gr.HTML()
        with gr.Accordion("Raw evidence log", open=False):
            evidence = gr.HTML()

        run_btn.click(
            run_audit,
            inputs=[drive_url, upload, custom_prompt, speaker_preset, custom_speakers, checks],
            outputs=[video, confidence, genuineness, qa, custom_answer, evidence, status],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860,
                        allowed_paths=[str(config.WORK_DIR)], head=HEAD_JS)
