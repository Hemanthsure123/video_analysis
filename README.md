# Interview Recording Audit System

A **local, GPU-accelerated** desktop tool that audits a single interview recording and
produces, on screen:

1. A **Genuineness** report — timestamped, confidence-labelled *flags for human review*
   (phone use, extra people, off-camera gaze, extra voices, external devices, possible
   external-assistance moments).
2. A **Q&A** report — inferred questions with per-answer ratings and an overall rating.
3. A direct answer to a **custom prompt** you type.
4. An expandable **raw evidence log**.

Every timestamp in the results is clickable and seeks the embedded video player to that
moment.

> **This is an assistant for a human reviewer, not an automatic judge.** Every
> cheating-type signal is surfaced as a *flag with a confidence level and a timestamp* —
> never a definitive accusation. Gaze and "possible LLM/tool usage" are inherently
> unreliable from a single webcam feed and produce false positives. The tool surfaces
> moments; **you** decide.

---

## How it works (pipeline)

```
ingest -> transcribe -> diarize -> vision -> signals -> report -> Gradio UI
```

| Stage          | Library                              | Notes                                            |
| -------------- | ------------------------------------ | ------------------------------------------------ |
| Download/audio | gdown + ffmpeg                       | Public Drive link or local upload -> 16 kHz WAV  |
| Transcription  | faster-whisper (large-v3)            | Word-level timestamps + confidence signal        |
| Diarization    | pyannote.audio                       | Extra voices flagged vs an **expected** count    |
| Vision         | Ultralytics YOLO11 + MediaPipe       | Phone/device/person detection + head-pose gaze   |
| Report         | OpenAI gpt-4o (swappable)            | Genuineness + Q&A + custom-prompt answer         |

All model and library choices live in [config.py](config.py) so they are swappable.

---

## Requirements

- **Python 3.10+**
- **ffmpeg** on your PATH (`ffmpeg -version` should work)
- **GPU is optional but strongly recommended.** The tool **auto-detects** an NVIDIA
  CUDA GPU and uses larger, more accurate models on it; otherwise it falls back to CPU
  with smaller, faster models so everything still runs (just much slower).

### GPU / CUDA notes

| Environment            | What happens                                                            |
| ---------------------- | ----------------------------------------------------------------------- |
| NVIDIA GPU + CUDA      | `DEVICE=cuda`, Whisper `large-v3` (float16), `yolo11m`. Full speed.      |
| CPU only (e.g. AMD)    | `DEVICE=cpu`, Whisper `small` (int8), `yolo11n`. Works, but slow.        |

A 60-minute video targets **5–15 min** on a GPU. On CPU it can take **30–90+ min**, so
test with **short clips** locally and run full-length audits on an NVIDIA machine.
Force a choice with `AUDIT_DEVICE=cuda|cpu` in `.env`.

---

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1            # PowerShell  (use .venv\Scripts\activate.bat for cmd)

# 2a. (NVIDIA only) install the CUDA build of torch FIRST
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2b. install everything else (CPU torch is pulled in automatically if you skipped 2a)
pip install -r requirements.txt

# 3. configure secrets
copy .env.example .env                # then edit .env

# 4. verify the environment
python check_gpu.py
```

### Hugging Face token + pyannote terms (one-time, required for diarization)

`pyannote.audio` (4.x) gates its models behind a license you must accept once:

1. Create a token at <https://hf.co/settings/tokens>. Use a **classic "Read"** token, or
   a fine-grained token **with "read access to public gated repos"** — otherwise you'll
   get a `403` even after accepting terms. Put it in `.env` as `HF_TOKEN`.
2. While logged in, click **"Agree and access repository"** on ALL of:
   - <https://hf.co/pyannote/speaker-diarization-community-1>  (the 4.x pipeline)
   - <https://hf.co/pyannote/segmentation-3.0>
   - <https://hf.co/pyannote/speaker-diarization-3.1>  (legacy, optional)

Without this, the diarization stage fails to download the pipeline (it degrades
gracefully — the rest of the audit still runs, just without the extra-voice signal).

> Note: on this Windows setup, pyannote's torchcodec audio decoder can't load (it needs
> FFmpeg "full-shared" DLLs). We bypass it by decoding the 16 kHz mono WAV in memory, so
> no extra setup is needed.

### Report LLM key

Only the **report** stage calls an LLM. Pick one backend via `LLM_BACKEND` in `.env`:

| Backend | `LLM_BACKEND` | Key | Notes |
| ------- | ------------- | --- | ----- |
| Gemini  | `gemini`      | `GEMINI_API_KEY` | **Default.** Free tier; `gemini-2.5-flash`. |
| Groq    | `groq`        | `GROQ_API_KEY`   | Free tier; Llama 3.3 70B. |
| OpenAI  | `openai`      | `OPENAI_API_KEY` | `gpt-4o`; needs billing/credits. |

Everything else is local and free. (Gemini free tier rejects some models with
`limit: 0` per project/region — `gemini-2.5-flash` works; override with `GEMINI_MODEL`.)

---

## Performance

A 60-min video targets **5–15 min on a GPU**. The pipeline auto-tunes by device:

| Knob (`config.py` / `.env`) | GPU default | CPU default | Effect |
| --------------------------- | ----------- | ----------- | ------ |
| `PARALLEL_STAGES`           | on          | off (auto)  | Runs transcribe ∥ vision ∥ diarize. Big win on GPU; **slower on CPU** (core oversubscription — measured), so auto-off. |
| `WHISPER_BEAM_SIZE`         | 5           | 1           | Greedy is ~2.6× faster than beam=5 on CPU for a small accuracy cost (matters mainly for noisy/accented audio). |
| `WHISPER_MODEL`             | `large-v3`  | `small`     | Accuracy vs speed. |
| `YOLO_BATCH`                | 16          | 16          | Batched detection — speeds up GPU; neutral on CPU. |
| `VISION_IMGSZ`              | 640         | 640         | Lower (e.g. 480) = faster YOLO, less accurate. |
| `SAMPLE_INTERVAL_SEC`       | 2.0         | 2.0         | Fewer frames = faster vision (raise to 3–4 for long videos). |

On CPU, a full 65-min audit is realistically ~25–40 min (transcription + vision +
diarization are inherently heavy). For the 5–15 min target, run on an NVIDIA GPU.

---

## Recommended recording conditions

Accuracy is **bounded by recording quality**. The tool states its own confidence and
lowers it when input quality is poor. For best results the recording should have:

- **A clear external or headset mic** — the single biggest factor for transcription
  accuracy.
- **A quiet room** with minimal background noise or other people.
- **Good, even front-facing lighting** on the candidate's face.
- **The webcam at or near eye level** — a low laptop camera makes everyone look downward
  and inflates "gaze off-center" flags.
- **The candidate fully and consistently in frame.**
- **Local recording where possible** (fewer compression / connection artifacts).

> Gaze, extra-device, and extra-voice detection **degrade** when these conditions aren't
> met, and **off-camera devices or people simply cannot be detected.** Diarization can
> tell that an extra voice *exists* but **cannot identify who** it is.

---

## Project layout

```
video_analysis/
  app.py                 # Gradio UI (Phase 7)
  config.py              # model names, sampling, thresholds, paths — single source of truth
  check_gpu.py           # environment / CUDA check
  pipeline/
    ingest.py            # Phase 1  download + audio extraction
    transcribe.py        # Phase 2  faster-whisper
    diarize.py           # Phase 4  pyannote
    vision.py            # Phase 3  YOLO + MediaPipe
    signals.py           # Phase 5  merged evidence log
    report.py            # Phase 6  LLM report
  llm/
    base.py              # LLM interface
    openai_client.py     # default (gemini/groq optional)
  transcription/
    whisper_local.py     # default (sarvam optional)
  requirements.txt
  .env.example
```

## Status

All stages built and tested end-to-end on real data: ingest → transcribe → diarize →
vision → signals → Gemini report → Gradio UI with clickable timestamps. Run the
environment check with `python check_gpu.py`, then launch the app with `python app.py`.

On this CPU dev machine a full 65-min audit is ~1 hour (diarization dominates); use a
short clip to try it, or an NVIDIA GPU to hit the 5–15 min target.
