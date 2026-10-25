"""
ingest.py — Phase 1.

Input : a Google Drive URL (public / anyone-with-link) OR an uploaded file path.
Steps : download via gdown if a URL; validate it is a real video (ffprobe);
        use ffmpeg to extract 16 kHz mono WAV audio and read duration + fps.
Output: {"video_path": str, "audio_path": str, "duration_sec": float, "fps": float}

Fails gracefully with clear messages: bad/private link, not a video, no audio
track, ffmpeg missing, etc.

CLI test:  python -m pipeline.ingest "<drive-url-or-local-path>"
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import config


class IngestError(Exception):
    """Raised for any user-facing ingest failure (bad link, no audio, etc.)."""


# --------------------------------------------------------------------------- #
# URL handling
# --------------------------------------------------------------------------- #
_DRIVE_ID_PATTERNS = [
    r"/file/d/([A-Za-z0-9_-]{20,})",        # .../file/d/<id>/view
    r"[?&]id=([A-Za-z0-9_-]{20,})",         # ...open?id=<id>  / uc?id=<id>
    r"/d/([A-Za-z0-9_-]{20,})",             # short forms
]


def looks_like_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source.strip(), re.IGNORECASE))


_ILLEGAL_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(name: str, fallback_stem: str) -> str:
    """Make a Drive title safe as a Windows filename, preserving the extension.

    Drive titles routinely contain ':' '|' etc., which are illegal on Windows and
    crash gdown's temp-file creation, so we replace them and cap the length.
    """
    name = name.strip().rstrip(". ")
    stem, dot, ext = name.rpartition(".")
    if not dot:  # no extension present
        stem, ext = name, ""
    stem = _ILLEGAL_FILENAME.sub("_", stem).strip("_ ") or fallback_stem
    ext = _ILLEGAL_FILENAME.sub("", ext)
    stem = stem[:80].rstrip("_ ") or fallback_stem
    return f"{stem}.{ext}" if ext else stem


def extract_drive_file_id(url: str) -> str | None:
    """Best-effort extraction of a Google Drive file id from common URL shapes."""
    for pat in _DRIVE_ID_PATTERNS:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def download_from_drive(url: str, dest_dir: Path) -> Path:
    """Download a public Drive file with gdown. Returns the local path."""
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover
        raise IngestError("gdown is not installed (pip install gdown).") from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    before = set(dest_dir.iterdir())

    # gdown's API has shifted across versions (6.x dropped `fuzzy`); prefer the file-id
    # form we parse ourselves, falling back to the raw URL.
    file_id = extract_drive_file_id(url)

    # Peek at the remote filename so we can write to a Windows-safe explicit path.
    remote_name = None
    if file_id:
        try:
            meta = gdown.download(id=file_id, skip_download=True, quiet=True)
            remote_name = getattr(meta, "path", None) or getattr(meta, "local_path", None)
        except Exception:
            remote_name = None

    fallback_stem = file_id or "drive_download"
    if remote_name:
        out_path = dest_dir / _sanitize_filename(Path(remote_name).name, fallback_stem)
    else:
        out_path = dest_dir / f"{fallback_stem}.mp4"

    last_exc: Exception | None = None
    for kwargs in ([{"id": file_id}] if file_id else []) + [{"url": url}]:
        try:
            out = gdown.download(output=str(out_path), quiet=False, **kwargs)
        except Exception as exc:
            last_exc = exc
            continue
        if out:
            return Path(out)
        if out_path.exists():
            return out_path

    # Neither call returned a path: detect any file that newly appeared.
    new_files = sorted(set(dest_dir.iterdir()) - before, key=lambda p: p.stat().st_mtime)
    if new_files:
        return new_files[-1]

    if last_exc is not None:
        raise IngestError(f"Drive download failed: {last_exc}")

    raise IngestError(
        "Could not download the file. Make sure the link is shared as "
        "'Anyone with the link' and points to a single video file."
    )


# --------------------------------------------------------------------------- #
# ffprobe / ffmpeg
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise IngestError(
            f"Could not run '{cmd[0]}'. Is ffmpeg installed and on your PATH?"
        ) from exc


def _parse_fraction(value: str | None) -> float:
    """Parse ffprobe rates like '30000/1001' or '25/1' into a float."""
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_media(path: Path) -> dict:
    """Return {duration_sec, fps, has_video, has_audio} via ffprobe; validate it."""
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise IngestError(
            f"'{path.name}' could not be read as media. "
            f"ffprobe said: {proc.stderr.strip() or 'unknown error'}"
        )

    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(f"Could not parse ffprobe output for '{path.name}'.") from exc

    streams = meta.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise IngestError(f"'{path.name}' has no video stream — it is not a video file.")

    # Duration: prefer container duration, fall back to the video stream's.
    duration = 0.0
    fmt_dur = meta.get("format", {}).get("duration")
    if fmt_dur:
        duration = _parse_fraction(fmt_dur)
    if duration <= 0:
        duration = _parse_fraction(video_streams[0].get("duration"))

    # FPS: prefer average frame rate (handles VFR better), fall back to base rate.
    vs = video_streams[0]
    fps = _parse_fraction(vs.get("avg_frame_rate")) or _parse_fraction(vs.get("r_frame_rate"))

    return {
        "duration_sec": round(duration, 3),
        "fps": round(fps, 3),
        "has_video": True,
        "has_audio": bool(audio_streams),
    }


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract 16 kHz mono PCM WAV from the video with ffmpeg."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_BIN,
        "-y",
        "-i", str(video_path),
        "-vn",                  # drop video
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz
        "-c:a", "pcm_s16le",    # standard WAV
        str(audio_path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not audio_path.exists():
        raise IngestError(
            f"Audio extraction failed. ffmpeg said:\n{proc.stderr.strip()[-800:]}"
        )
    return audio_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def ingest(source: str, is_url: bool | None = None) -> dict:
    """Download (if needed), validate, and extract audio. See module docstring."""
    if not source or not str(source).strip():
        raise IngestError("No video source provided.")
    source = str(source).strip()

    if is_url is None:
        is_url = looks_like_url(source)

    if is_url:
        video_path = download_from_drive(source, config.DOWNLOAD_DIR)
    else:
        video_path = Path(source)
        if not video_path.exists():
            raise IngestError(f"File not found: {source}")

    info = probe_media(video_path)
    if not info["has_audio"]:
        raise IngestError(
            f"'{video_path.name}' has no audio track. Transcription and diarization "
            "need audio, so this recording cannot be audited."
        )

    duration_min = info["duration_sec"] / 60.0
    if duration_min > config.MAX_VIDEO_MINUTES:
        print(
            f"[ingest] WARNING: video is {duration_min:.0f} min, longer than the "
            f"{config.MAX_VIDEO_MINUTES}-min target. Processing may be slow."
        )

    audio_path = extract_audio(video_path, config.AUDIO_DIR / f"{video_path.stem}.wav")

    return {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "duration_sec": info["duration_sec"],
        "fps": info["fps"],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m pipeline.ingest "<drive-url-or-local-path>"')
        raise SystemExit(2)

    try:
        result = ingest(sys.argv[1])
    except IngestError as e:
        print(f"INGEST ERROR: {e}")
        raise SystemExit(1)

    print("Ingest OK:")
    for k, v in result.items():
        print(f"  {k:12} = {v}")
