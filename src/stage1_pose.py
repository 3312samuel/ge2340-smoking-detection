"""Stage 1: YOLO26 pose screening for hand-to-mouth gestures.

Scale-invariant rule (works for people near to and far from the camera):

    ratio = min over wrists ||nose - wrist|| / ||nose - shoulder midpoint||

computed in pixels. ratio < RATIO_THR means a hand is at the face. The
denominator (roughly neck length) shrinks and grows with the person, so the
same threshold works at 1.5 m and at 5 m. A wrist only counts if it is above
shoulder height. A person is flagged when ratio stays below the threshold for
SUSTAIN_S seconds (a quick touch of the face is ignored).

An earlier version used the raw nose-wrist distance in normalised image
coordinates (< 0.14). That only works when people appear small (wide CCTV
views); in our close-up clips it never fired, so it was replaced.

RATIO_THR is chosen on the dev clips (analyze.py --freeze); SUSTAIN_S is fixed
at 1.0 s in advance.
"""
import numpy as np

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 0, 5, 6, 9, 10


def mouth_ratio(kxy, kconf, kpt_conf=0.5, wrist_conf=0.3):
    """Hand-to-mouth ratio for one person, or None if it cannot be computed.

    kxy: (17, 2) pixel coordinates; kconf: (17,) keypoint confidences.
    Wrists use a lower confidence bar because a hand at the face is often
    partly occluded (by a bottle, phone, or the face itself).
    """
    if kconf[NOSE] < kpt_conf:
        return None
    nose = kxy[NOSE]
    shoulders = [kxy[s] for s in (L_SHOULDER, R_SHOULDER) if kconf[s] >= kpt_conf]
    if not shoulders:
        return None
    mid = np.mean(shoulders, axis=0)
    scale = float(np.linalg.norm(nose - mid))
    if scale < 1e-3:
        return None
    height_limit = mid[1] + 0.25 * scale  # wrist must be above (roughly) shoulder height
    best = None
    for w in (L_WRIST, R_WRIST):
        if kconf[w] >= wrist_conf and kxy[w, 1] < height_limit:
            r = float(np.linalg.norm(nose - kxy[w])) / scale
            best = r if best is None else min(best, r)
    return best


def sustained(samples, thr, sustain_s=1.0):
    """samples: list of (sample_index, time_s, ratio or None) for ONE track,
    in order. True if ratio < thr holds over consecutive samples for >= sustain_s."""
    start, prev_idx = None, None
    for idx, t, r in samples:
        if r is None or r >= thr:
            start, prev_idx = None, None
            continue
        if start is None or prev_idx != idx - 1:
            start = t  # a new run begins
        prev_idx = idx
        if t - start >= sustain_s:
            return True
    return False


class PoseScreener:
    def __init__(self, weights="yolo26n-pose.pt", det_conf=0.5, kpt_conf=0.5,
                 imgsz=640, device=None):
        self.weights = weights
        self.det_conf, self.kpt_conf = det_conf, kpt_conf
        self.imgsz, self.device = imgsz, device
        self.reset()

    def reset(self):
        """Call before each new clip: fresh tracker state."""
        from ultralytics import YOLO  # imported here so analyze.py runs without it
        self.model = YOLO(self.weights)

    def step(self, frame):
        """Process one frame. Returns [{track_id, box, ratio}] for each person."""
        res = self.model.track(frame, persist=True, conf=self.det_conf,
                               imgsz=self.imgsz, device=self.device,
                               tracker="bytetrack.yaml", verbose=False)[0]
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            return []
        ids = res.boxes.id.int().cpu().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kconf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                 else np.ones(kxy.shape[:2]))
        return [{"track_id": tid, "box": boxes[i],
                 "ratio": mouth_ratio(kxy[i], kconf[i], self.kpt_conf)}
                for i, tid in enumerate(ids)]
