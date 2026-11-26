"""Shared helpers: video frame iteration, person cropping, timing."""
import time
from contextlib import contextmanager

import cv2
import numpy as np


def iter_frames(video_path, stride=5):
    """Yield (frame_index, timestamp_s, frame_bgr) for every `stride`-th frame.

    One inference every 5 raw frames at 30 fps (~6 inferences/s), so the
    9-frame sustain rule corresponds to ~1.5 s of continuous gesture.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def video_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def crop_person(frame, box_xyxy, pad=0.3):
    """Crop a person box with `pad` (fraction of box size) extra context.

    30% padding keeps the hand and mouth region, which matters for a small object like a cigarette.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
    cx2, cy2 = min(w, x2 + pw), min(h, y2 + ph)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def timer(store, key):
    """Append elapsed milliseconds (GPU-synchronised) to store[key]."""
    _sync()
    t0 = time.perf_counter()
    yield
    _sync()
    store.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)


def summarise_ms(values):
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 95))
