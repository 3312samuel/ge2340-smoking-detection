"""Stage 1: YOLO pose screening for hand-to-mouth gestures.

Rule:
  * nose-to-wrist distance in normalised image coordinates < DIST_THR (0.14)
  * the wrist must be at or above shoulder height (+0.05 tolerance)
  * the condition must hold for SUSTAIN consecutive inference frames (9)
Keypoints come from the official YOLO26 pose weights; low-confidence keypoints
(< kpt_conf) are ignored.
"""
import numpy as np
from ultralytics import YOLO

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 0, 5, 6, 9, 10


def hand_to_mouth(kxyn, kconf, dist_thr=0.14, kpt_conf=0.5, shoulder_tol=0.05):
    """Return (is_suspicious, min_distance) for one person's 17 COCO keypoints.

    kxyn: (17, 2) normalised xy; kconf: (17,) keypoint confidences.
    """
    if kconf[NOSE] < kpt_conf:
        return False, float("inf")
    nose = kxyn[NOSE]

    shoulder_y = 1.0  # unknown shoulders -> no height restriction
    if kconf[L_SHOULDER] >= kpt_conf and kconf[R_SHOULDER] >= kpt_conf:
        shoulder_y = (kxyn[L_SHOULDER, 1] + kxyn[R_SHOULDER, 1]) / 2.0
    height_limit = shoulder_y + shoulder_tol

    best = float("inf")
    for wrist in (L_WRIST, R_WRIST):
        if kconf[wrist] >= kpt_conf and kxyn[wrist, 1] < height_limit:
            best = min(best, float(np.linalg.norm(nose - kxyn[wrist])))
    return best < dist_thr, best


class PoseScreener:
    def __init__(self, weights="yolo26n-pose.pt", dist_thr=0.14, sustain=9,
                 det_conf=0.5, kpt_conf=0.5, imgsz=640, device=None):
        self.weights = weights
        self.dist_thr, self.sustain = dist_thr, sustain
        self.det_conf, self.kpt_conf = det_conf, kpt_conf
        self.imgsz, self.device = imgsz, device
        self.reset()

    def reset(self):
        """Call before each new clip: fresh tracker state and counters."""
        self.model = YOLO(self.weights)
        self.counters = {}

    def step(self, frame):
        """Process one frame. Returns a list of per-person dicts."""
        res = self.model.track(frame, persist=True, conf=self.det_conf,
                               imgsz=self.imgsz, device=self.device,
                               tracker="bytetrack.yaml", verbose=False)[0]
        people = []
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            self.counters.clear()
            return people

        ids = res.boxes.id.int().cpu().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kxyn = res.keypoints.xyn.cpu().numpy()
        kconf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                 else np.ones(kxyn.shape[:2]))

        seen = set()
        for i, tid in enumerate(ids):
            seen.add(tid)
            sus, dist = hand_to_mouth(kxyn[i], kconf[i], self.dist_thr, self.kpt_conf)
            count = self.counters.get(tid, 0) + 1 if sus else 0
            self.counters[tid] = count
            people.append({
                "track_id": tid,
                "box": boxes[i],
                "suspicious": sus,
                "dist": dist,
                "count": count,
                "triggered": count >= self.sustain,
            })
        for tid in list(self.counters):
            if tid not in seen:
                del self.counters[tid]
        return people
