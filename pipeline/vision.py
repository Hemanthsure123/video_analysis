"""
vision.py — Phase 3.

Input : video_path, fps, sampling interval (default 1 frame / 2 s; configurable).
Steps : sample frames (via ffmpeg); run YOLO (phone/laptop/tv/book/person) +
        MediaPipe Face Landmarker (head yaw/pitch -> gaze direction) on each;
        collapse consecutive frames into timestamped ranges.
Output: {"events": {"phone_visible":   [{start, end, confidence}],
                     "external_device": [{start, end, type, confidence}],
                     "extra_person":    [{start, end, max_faces}],
                     "gaze_off_center": [{start, end, direction}]},
         "metrics": {...aggregate stats...},
         "per_frame": [{t, persons, faces, phone, devices, yaw, pitch, gaze}],
         "sample_interval_sec": float}

CLI test:  python -m pipeline.vision "<video>" [interval_sec]
"""

from __future__ import annotations

import math
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

import config

# Lazily-initialised, cached heavy objects.
_YOLO = None
_FACE = None


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO

        print(f"[vision] loading YOLO '{config.YOLO_MODEL}' on {config.DEVICE}...")
        _YOLO = YOLO(config.YOLO_MODEL)
    return _YOLO


def _ensure_face_model() -> str:
    path = Path(config.FACE_LANDMARKER_MODEL)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("[vision] downloading MediaPipe face_landmarker model...")
        urllib.request.urlretrieve(config.FACE_LANDMARKER_URL, str(path))
    return str(path)


def _get_face():
    global _FACE
    if _FACE is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_face_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=config.VISION_MAX_FACES,
            output_facial_transformation_matrixes=True,
        )
        _FACE = mp_vision.FaceLandmarker.create_from_options(options)
        print("[vision] MediaPipe Face Landmarker ready.")
    return _FACE


# --------------------------------------------------------------------------- #
# Frame sampling (ffmpeg → JPEGs)
# --------------------------------------------------------------------------- #
def _extract_frames(video_path: str, interval: float) -> list:
    """Extract 1 frame every `interval` seconds. Returns [(index, t_sec, path)]."""
    stem = Path(video_path).stem
    out_dir = config.FRAMES_DIR / stem
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rate = 1.0 / interval
    pattern = str(out_dir / "f_%06d.jpg")
    cmd = [
        config.FFMPEG_BIN, "-y", "-i", str(video_path),
        "-vf", f"fps={rate}", "-q:v", "3", pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    frames = sorted(out_dir.glob("f_*.jpg"))
    if not frames:
        raise RuntimeError(
            f"Frame extraction produced no frames. ffmpeg said:\n{proc.stderr[-600:]}"
        )
    # Frame i (0-based) corresponds to input time ~ i * interval.
    return [(i, round(i * interval, 3), p) for i, p in enumerate(frames)]


# --------------------------------------------------------------------------- #
# Per-frame detection
# --------------------------------------------------------------------------- #
def _containment(inner: list, outer: list) -> float:
    """Fraction of the `inner` box area that lies within the `outer` box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _parse_yolo_result(r) -> dict:
    """Parse one Ultralytics Result, splitting detections into physical vs
    on-screen (inside a monitor/TV box). On-screen persons/phones are screen-share
    artifacts, not physical presences.

    Returns {persons_physical, persons_on_screen, phone_physical,
             phone_on_screen, devices: {type: conf}, screen_present}.
    """
    names = r.names
    boxes = []
    for box in r.boxes:
        boxes.append((names[int(box.cls)], float(box.conf), box.xyxy[0].tolist()))

    # Screen regions = detected monitor/TV boxes.
    screens = [xyxy for cls, _c, xyxy in boxes
               if config.YOLO_DEVICE_CLASSES.get(cls) == "monitor"]

    def on_screen(xyxy) -> bool:
        return any(_containment(xyxy, s) >= config.ONSCREEN_CONTAINMENT for s in screens)

    out = {
        "persons_physical": 0, "persons_on_screen": 0,
        "phone_physical": 0.0, "phone_on_screen": 0.0,
        "devices": {}, "screen_present": bool(screens),
    }
    for cls_name, conf, xyxy in boxes:
        if cls_name == config.YOLO_PERSON_CLASS:
            if on_screen(xyxy):
                out["persons_on_screen"] += 1
            else:
                out["persons_physical"] += 1
        elif cls_name in config.YOLO_DEVICE_CLASSES:
            mapped = config.YOLO_DEVICE_CLASSES[cls_name]
            if mapped == "phone":
                key = "phone_on_screen" if on_screen(xyxy) else "phone_physical"
                out[key] = max(out[key], conf)
            else:
                out["devices"][mapped] = max(out["devices"].get(mapped, 0.0), conf)
    return out


def _rotation_to_euler(R: np.ndarray) -> tuple:
    """Decompose a 3x3 rotation matrix into (pitch, yaw, roll) degrees (ZYX)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def _run_face(frame_rgb) -> tuple:
    """Return (face_count, yaw_deg, pitch_deg) for the largest face (or None pose)."""
    import mediapipe as mp

    landmarker = _get_face()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    faces = result.face_landmarks
    n = len(faces)
    if n == 0 or not result.facial_transformation_matrixes:
        return n, None, None

    # Pick the largest face by landmark bounding-box area.
    best_idx, best_area = 0, -1.0
    for idx, lm in enumerate(faces):
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area, best_idx = area, idx

    matrix = np.array(result.facial_transformation_matrixes[best_idx]).reshape(4, 4)
    pitch, yaw, _roll = _rotation_to_euler(matrix[:3, :3])
    return n, round(yaw, 2), round(pitch, 2)


def classify_gaze(yaw, pitch):
    """Classify head pose into center / left / right / up / down.

    Sign convention (validated empirically on webcam footage): positive yaw =>
    head turned to the subject's left (camera-right); positive pitch => head
    tilted down. Yaw is checked before pitch.
    """
    if yaw is None or pitch is None:
        return None
    if yaw >= config.GAZE_YAW_THRESHOLD_DEG:
        return "left"
    if yaw <= -config.GAZE_YAW_THRESHOLD_DEG:
        return "right"
    if pitch >= config.GAZE_PITCH_DOWN_THRESHOLD_DEG:
        return "down"
    if pitch <= -config.GAZE_PITCH_UP_THRESHOLD_DEG:
        return "up"
    return "center"


# --------------------------------------------------------------------------- #
# Collapse consecutive frames into ranges
# --------------------------------------------------------------------------- #
def _merge(times_vals: list, interval: float, gap: float) -> list:
    """Merge (t, value) samples into ranges (bridging gaps <= `gap`).

    Each range gets start, end (last_t + interval), and the max value seen.
    """
    ranges: list = []
    for t, v in sorted(times_vals):
        if ranges and t - ranges[-1]["end"] <= gap:
            ranges[-1]["end"] = round(t + interval, 3)
            ranges[-1]["_max"] = max(ranges[-1]["_max"], v)
        else:
            ranges.append({"start": t, "end": round(t + interval, 3), "_max": v})
    return ranges


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def analyze_video(video_path: str, fps: float | None = None,
                  sample_interval_sec: float | None = None) -> dict:
    import cv2

    interval = float(sample_interval_sec or config.SAMPLE_INTERVAL_SEC)
    gap = config.EVENT_MERGE_GAP_SEC
    frames = _extract_frames(video_path, interval)

    model = _get_yolo()
    _get_face()  # warm up the landmarker once before the loop
    batch = max(1, config.YOLO_BATCH)
    device = 0 if config.ON_GPU else "cpu"

    per_frame: list = []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch]
        imgs, valid = [], []
        for i, t, path in chunk:
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            imgs.append(bgr)
            valid.append((t, bgr))
        if not imgs:
            continue

        # Batched YOLO inference over the whole chunk at once.
        results = model.predict(
            imgs, conf=config.YOLO_CONF_THRESHOLD, device=device,
            imgsz=config.VISION_IMGSZ, verbose=False,
        )

        for (t, bgr), r in zip(valid, results):
            det = _parse_yolo_result(r)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            n_faces, yaw, pitch = _run_face(rgb)
            gaze = classify_gaze(yaw, pitch)
            per_frame.append({
                "t": t,
                "persons": det["persons_physical"],
                "persons_on_screen": det["persons_on_screen"],
                "faces": n_faces,
                "phone": round(det["phone_physical"], 3),
                "phone_on_screen": round(det["phone_on_screen"], 3),
                "devices": {k: round(v, 3) for k, v in det["devices"].items()},
                "screen_present": det["screen_present"],
                "yaw": yaw,
                "pitch": pitch,
                "gaze": gaze,
            })

    events = _build_events(per_frame, interval, gap)
    metrics = _build_metrics(per_frame, events, interval)
    return {
        "events": events,
        "metrics": metrics,
        "per_frame": per_frame,
        "sample_interval_sec": interval,
    }


def _build_events(per_frame: list, interval: float, gap: float) -> dict:
    # phone
    phone = [(f["t"], f["phone"]) for f in per_frame if f["phone"] > 0]
    phone_visible = [
        {"start": r["start"], "end": r["end"], "confidence": round(r["_max"], 3)}
        for r in _merge(phone, interval, gap)
    ]

    # external devices (per type)
    external_device = []
    device_types = {d for f in per_frame for d in f["devices"]}
    for dtype in sorted(device_types):
        samples = [(f["t"], f["devices"][dtype]) for f in per_frame if dtype in f["devices"]]
        for r in _merge(samples, interval, gap):
            external_device.append({
                "start": r["start"], "end": r["end"],
                "type": dtype, "confidence": round(r["_max"], 3),
            })
    external_device.sort(key=lambda e: e["start"])

    # extra person (person OR face count > 1)
    extra = [(f["t"], max(f["persons"], f["faces"])) for f in per_frame
             if max(f["persons"], f["faces"]) > 1]
    extra_person = [
        {"start": r["start"], "end": r["end"], "max_faces": int(r["_max"])}
        for r in _merge(extra, interval, gap)
    ]

    # gaze off-center (per direction)
    gaze_off_center = []
    for direction in ("left", "right", "up", "down"):
        samples = [(f["t"], 1) for f in per_frame if f["gaze"] == direction]
        for r in _merge(samples, interval, gap):
            gaze_off_center.append({
                "start": r["start"], "end": r["end"], "direction": direction,
            })
    gaze_off_center.sort(key=lambda e: e["start"])

    return {
        "phone_visible": phone_visible,
        "external_device": external_device,
        "extra_person": extra_person,
        "gaze_off_center": gaze_off_center,
    }


def _build_metrics(per_frame: list, events: dict, interval: float) -> dict:
    n = len(per_frame)
    with_face = [f for f in per_frame if f["gaze"] is not None]
    off_center = [f for f in with_face if f["gaze"] != "center"]

    gaze_breakdown = {k: 0 for k in ("center", "left", "right", "up", "down")}
    for f in with_face:
        gaze_breakdown[f["gaze"]] += 1

    device_seconds: dict = {}
    for f in per_frame:
        for dtype in f["devices"]:
            device_seconds[dtype] = round(device_seconds.get(dtype, 0.0) + interval, 1)

    phone_frames = sum(1 for f in per_frame if f["phone"] > 0)
    screen_frames = sum(1 for f in per_frame if f.get("screen_present"))
    on_screen_person_frames = sum(1 for f in per_frame if f.get("persons_on_screen", 0) > 0)
    return {
        "frames_sampled": n,
        "frames_with_face": len(with_face),
        "gaze_off_center_pct": round(100.0 * len(off_center) / len(with_face), 1) if with_face else 0.0,
        "gaze_breakdown": gaze_breakdown,
        "total_phone_seconds": round(phone_frames * interval, 1),
        "external_device_seconds": device_seconds,
        "extra_person_windows": len(events["extra_person"]),
        "max_persons": max((f["persons"] for f in per_frame), default=0),
        "max_faces": max((f["faces"] for f in per_frame), default=0),
        "sample_interval_sec": interval,
        # On-screen-content context (screen-share artifacts; lower weight).
        "screen_present_pct": round(100.0 * screen_frames / n, 1) if n else 0.0,
        "max_persons_on_screen": max((f.get("persons_on_screen", 0) for f in per_frame), default=0),
        "on_screen_person_seconds": round(on_screen_person_frames * interval, 1),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m pipeline.vision "<video>" [interval_sec]')
        raise SystemExit(2)

    interval = float(sys.argv[2]) if len(sys.argv) > 2 else None
    import time
    t0 = time.time()
    out = analyze_video(sys.argv[1], sample_interval_sec=interval)
    elapsed = time.time() - t0

    m = out["metrics"]
    print("\n" + "=" * 64)
    print(f"Analysed {m['frames_sampled']} frames in {elapsed:.1f}s "
          f"(interval {out['sample_interval_sec']}s)")
    print(f"faces seen: max {m['max_faces']}  | physical persons: max {m['max_persons']}")
    print(f"on-screen: present {m['screen_present_pct']}% | on-screen persons max "
          f"{m['max_persons_on_screen']} ({m['on_screen_person_seconds']}s)  [filtered out of flags]")
    print(f"gaze off-center: {m['gaze_off_center_pct']}%  breakdown={m['gaze_breakdown']}")
    print(f"phone seconds: {m['total_phone_seconds']}  | devices: {m['external_device_seconds']}")
    print("-" * 64)
    for name, evs in out["events"].items():
        print(f"{name}: {len(evs)} event(s)")
        for e in evs[:5]:
            print(f"    {e}")
    print("-" * 64)
    print("First 12 frames (t, faces, persons, yaw, pitch, gaze, phone):")
    for f in out["per_frame"][:12]:
        print(f"  t={f['t']:6.1f}  faces={f['faces']}  persons={f['persons']}  "
              f"yaw={f['yaw']}  pitch={f['pitch']}  gaze={f['gaze']}  phone={f['phone']}")
