"""
config.py — single source of truth for the Interview Recording Audit System.

Everything swappable lives here: model names, the compute device, sampling
intervals, detection thresholds, the expected-speaker default, file paths, and
which LLM / transcription backend to use. Nothing else in the codebase should
hardcode a model name or a magic threshold — import it from here.

Device strategy: AUTO-DETECT. If an NVIDIA CUDA GPU is present we use it with
larger, more accurate models; otherwise we fall back to CPU with smaller, faster
models so the pipeline still runs (just slower). Override anything via .env.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = Path(os.getenv("AUDIT_WORK_DIR") or (PROJECT_ROOT / "work"))
DOWNLOAD_DIR = WORK_DIR / "downloads"
AUDIO_DIR = WORK_DIR / "audio"
FRAMES_DIR = WORK_DIR / "frames"
MODELS_DIR = Path(os.getenv("AUDIT_MODELS_DIR") or (PROJECT_ROOT / "models"))

for _d in (WORK_DIR, DOWNLOAD_DIR, AUDIO_DIR, FRAMES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# External binaries (ffmpeg / ffprobe)
# --------------------------------------------------------------------------- #
def _resolve_ffprobe(ffmpeg_path: str) -> str:
    """Find ffprobe; fall back to the directory ffmpeg lives in."""
    found = shutil.which("ffprobe")
    if found:
        return found
    if ffmpeg_path and ffmpeg_path != "ffmpeg":
        name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        cand = Path(ffmpeg_path).with_name(name)
        if cand.exists():
            return str(cand)
    return "ffprobe"


FFMPEG_BIN = os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = os.getenv("FFPROBE_BIN") or _resolve_ffprobe(FFMPEG_BIN)


# --------------------------------------------------------------------------- #
# Device detection
# --------------------------------------------------------------------------- #
def _detect_device() -> str:
    """Return 'cuda' if a usable NVIDIA GPU is present, else 'cpu'.

    torch is imported lazily so this file can be imported before torch is
    installed (early scaffolding) without crashing. Set AUDIT_DEVICE in .env to
    force a choice.
    """
    override = os.getenv("AUDIT_DEVICE", "").strip().lower()
    if override in {"cuda", "cpu"}:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = _detect_device()
ON_GPU = DEVICE == "cuda"


# --------------------------------------------------------------------------- #
# Transcription (faster-whisper, local) — see transcription/whisper_local.py
# --------------------------------------------------------------------------- #
TRANSCRIPTION_BACKEND = os.getenv("TRANSCRIPTION_BACKEND", "whisper_local")  # | "sarvam"

# Bigger / more accurate on GPU; smaller / faster on CPU so smoke tests stay quick.
WHISPER_MODEL = os.getenv("WHISPER_MODEL") or ("large-v3" if ON_GPU else "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE") or ("float16" if ON_GPU else "int8")
WHISPER_DEVICE = DEVICE
# Beam search is ~2.6x slower than greedy on CPU for a small accuracy gain that
# mostly matters for noisy/accented audio. Default: greedy on CPU (speed), full
# beam on GPU (accuracy is cheap there). Override with WHISPER_BEAM_SIZE.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE") or ("5" if ON_GPU else "1"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE") or None  # None = autodetect
# VAD filtering trims long silences (common in interviews) and reduces
# hallucinated text during quiet stretches.
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "1").lower() not in ("0", "false", "no")

# Transcription-confidence heuristics (aggregates of Whisper per-segment scores).
# avg_logprob ~ [-1, 0]; closer to 0 = more confident. no_speech_prob ~ [0, 1].
TRANSCRIPTION_LOW_CONF_AVG_LOGPROB = float(os.getenv("TRANSCRIPTION_LOW_CONF_AVG_LOGPROB", "-0.7"))
TRANSCRIPTION_HIGH_NO_SPEECH_PROB = float(os.getenv("TRANSCRIPTION_HIGH_NO_SPEECH_PROB", "0.6"))


# --------------------------------------------------------------------------- #
# Diarization (pyannote.audio) — see pipeline/diarize.py
# --------------------------------------------------------------------------- #
HF_TOKEN = os.getenv("HF_TOKEN")  # required for pyannote; accept model terms once on HF
# pyannote.audio 4.x's native pipeline. (The legacy "pyannote/speaker-diarization-3.1"
# also works but pulls community-1's embedding assets, so we target it directly.)
PYANNOTE_PIPELINE = os.getenv("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1")
EXPECTED_SPEAKERS_DEFAULT = int(os.getenv("EXPECTED_SPEAKERS_DEFAULT", "2"))
# Drop extra-voice windows shorter than this (sec) — sub-second hits are usually
# diarization over-segmentation artifacts, not a real extra speaker.
DIARIZE_MIN_EXTRA_VOICE_SEC = float(os.getenv("DIARIZE_MIN_EXTRA_VOICE_SEC", "0.5"))


# --------------------------------------------------------------------------- #
# Vision (Ultralytics YOLO + MediaPipe Face Landmarker) — see pipeline/vision.py
# --------------------------------------------------------------------------- #
YOLO_MODEL = os.getenv("YOLO_MODEL") or ("yolo11m.pt" if ON_GPU else "yolo11n.pt")
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))

# Screen-share handling: a person/phone whose box is >= this fraction inside a
# detected monitor/TV box is treated as "on-screen content" (e.g. a video-call
# tile), NOT a physical person/phone in the room. Reduces false genuineness flags.
ONSCREEN_CONTAINMENT = float(os.getenv("ONSCREEN_CONTAINMENT", "0.6"))

# COCO class names we care about. `person` is counted; the rest are "devices".
YOLO_PERSON_CLASS = "person"
YOLO_DEVICE_CLASSES = {
    "cell phone": "phone",
    "laptop": "laptop",
    "tv": "monitor",
    "book": "book",
}

# Frame sampling: analyse one frame every N seconds of video.
SAMPLE_INTERVAL_SEC = float(os.getenv("SAMPLE_INTERVAL_SEC", "2.0"))

# Max faces MediaPipe will look for per frame (supports the "extra person" signal).
VISION_MAX_FACES = int(os.getenv("VISION_MAX_FACES", "5"))

# MediaPipe Face Landmarker model (auto-downloaded to MODELS_DIR on first use).
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Gaze / head-pose thresholds, in degrees from center. Beyond these = off-center.
GAZE_YAW_THRESHOLD_DEG = float(os.getenv("GAZE_YAW_THRESHOLD_DEG", "20.0"))  # left / right
GAZE_PITCH_UP_THRESHOLD_DEG = float(os.getenv("GAZE_PITCH_UP_THRESHOLD_DEG", "15.0"))
GAZE_PITCH_DOWN_THRESHOLD_DEG = float(os.getenv("GAZE_PITCH_DOWN_THRESHOLD_DEG", "20.0"))

# Collapse consecutive "on" frames into one event range when gaps are <= this (sec).
EVENT_MERGE_GAP_SEC = float(os.getenv("EVENT_MERGE_GAP_SEC", "3.0"))


# --------------------------------------------------------------------------- #
# "Possible external assistance" soft signal — LOW CONFIDENCE, human-review only
# --------------------------------------------------------------------------- #
# Minimum sustained off-center gaze (sec) before we even consider the soft signal.
ASSIST_MIN_OFFCENTER_SEC = float(os.getenv("ASSIST_MIN_OFFCENTER_SEC", "8.0"))


# --------------------------------------------------------------------------- #
# Report LLM — see llm/
# --------------------------------------------------------------------------- #
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # | "gemini" | "groq"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Approx context budget (tokens) before we chunk the transcript. gpt-4o ~ 128K.
LLM_CONTEXT_TOKENS = int(os.getenv("LLM_CONTEXT_TOKENS", "120000"))


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
# Run transcription / vision / diarization concurrently (they are independent).
# On GPU the stages overlap and wall-clock approaches max(stage) instead of
# sum(stage). On CPU, however, they oversubscribe the cores and it runs SLOWER
# (measured), so "auto" enables it only on GPU. Override with PARALLEL_STAGES=1/0.
_ps = os.getenv("PARALLEL_STAGES", "auto").lower()
if _ps in ("1", "true", "yes"):
    PARALLEL_STAGES = True
elif _ps in ("0", "false", "no"):
    PARALLEL_STAGES = False
else:
    PARALLEL_STAGES = ON_GPU
# 0 = let ctranslate2 auto-pick (number of physical cores) — already optimal on CPU.
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))
# YOLO frames per batched inference call (big speedup vs one-by-one).
YOLO_BATCH = int(os.getenv("YOLO_BATCH", "16"))
# YOLO inference image size; lower = faster, slightly less accurate.
VISION_IMGSZ = int(os.getenv("VISION_IMGSZ", "640"))


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "90"))


def summary() -> dict:
    """Compact view of the resolved config — used by check_gpu.py and logging."""
    return {
        "device": DEVICE,
        "whisper_model": WHISPER_MODEL,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
        "yolo_model": YOLO_MODEL,
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
        "expected_speakers_default": EXPECTED_SPEAKERS_DEFAULT,
        "transcription_backend": TRANSCRIPTION_BACKEND,
        "llm_backend": LLM_BACKEND,
        "openai_model": OPENAI_MODEL,
        "work_dir": str(WORK_DIR),
        "hf_token_set": bool(HF_TOKEN),
        "openai_key_set": bool(OPENAI_API_KEY),
    }
