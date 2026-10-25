"""
run_clip.py — dev harness.

Runs the heavy pipeline stages (transcribe + vision, optionally diarize) on a
single video once and caches the raw outputs to WORK_DIR/clip_outputs.json, so
Phase 5/6 (signals/report) can be iterated on without recomputing.

    python scripts/run_clip.py "<video>" [--diarize] [--expected N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow running as `python scripts/run_clip.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.ingest import ingest
from pipeline.transcribe import transcribe
from pipeline.vision import analyze_video

DEFAULT_CLIP = str(config.WORK_DIR / "testclips" / "interview_2min.mp4")


def main() -> int:
    args = sys.argv[1:]
    do_diar = "--diarize" in args
    expected = config.EXPECTED_SPEAKERS_DEFAULT
    if "--expected" in args:
        expected = int(args[args.index("--expected") + 1])
    positionals = [a for a in args if not a.startswith("--") and a.isascii() and not a.isdigit()]
    video = positionals[0] if positionals else DEFAULT_CLIP

    print(f"[run_clip] video    = {video}")
    print(f"[run_clip] diarize  = {do_diar} (expected {expected})")

    t0 = time.time()
    ing = ingest(video, is_url=False)
    print(f"[run_clip] ingest done ({time.time()-t0:.1f}s): {ing['duration_sec']}s @ {ing['fps']}fps")

    t1 = time.time()
    tr = transcribe(ing["audio_path"])
    print(f"[run_clip] transcribe done ({time.time()-t1:.1f}s): {len(tr['segments'])} segments")

    t2 = time.time()
    vi = analyze_video(ing["video_path"], fps=ing["fps"])
    print(f"[run_clip] vision done ({time.time()-t2:.1f}s): {vi['metrics']['frames_sampled']} frames")

    di = None
    if do_diar:
        from pipeline.diarize import DiarizeError, diarize
        t3 = time.time()
        try:
            di = diarize(ing["audio_path"], expected)
            print(f"[run_clip] diarize done ({time.time()-t3:.1f}s): "
                  f"{di['distinct_speaker_count']} speakers")
        except DiarizeError as e:
            print(f"[run_clip] diarize skipped: {e}")

    out_path = Path(config.WORK_DIR) / "clip_outputs.json"
    out_path.write_text(
        json.dumps({"ingest": ing, "transcription": tr, "diarization": di, "vision": vi},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[run_clip] cached -> {out_path}  (total {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
