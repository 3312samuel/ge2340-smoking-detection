"""Run all three approaches on every labelled clip and log raw scores.

One GPU pass produces results/raw_scores.csv. Thresholds are NOT applied here;
analyze.py picks them on the dev split, freezes them, then scores the test split.

Per clip we log:
  A1 pose-only      : did any track satisfy the sustained hand-to-mouth rule?
  A2 detector-only  : max cigarette confidence of our YOLO26n detector over frames
  A3 pose + SAM 3   : max SAM 3 "cigarette" score over crops of triggered tracks
plus per-stage latency (ms, GPU-synchronised).

Usage (Colab):
  python src/run_eval.py --labels data/labels.csv --clips data/clips \
      --sam3 models/sam3.pt --detector models/cigarette_yolo26n_ge2340.pt
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import crop_person, iter_frames, summarise_ms, timer  # noqa: E402
from stage1_pose import PoseScreener  # noqa: E402

FIELDS = [
    "file", "person", "gesture", "distance", "is_smoking", "split", "n_frames",
    "pose_triggered", "pose_first_trigger_s", "n_sam3_calls", "sam3_max_score",
    "det_max_conf", "det_frames_hit",
    "stage1_ms_mean", "stage1_ms_p95", "sam3_ms_mean", "sam3_ms_p95",
    "det_ms_mean", "det_ms_p95",
]


def evaluate_clip(path, pose, sam3, detector, args, crop_dir=None):
    pose.reset()
    lat = {}
    pose_trig, first_t = False, ""
    sam3_scores, verified_tracks = [], set()
    det_max, det_hits, n = 0.0, 0, 0

    for fidx, t, frame in iter_frames(path, args.stride):
        n += 1
        with timer(lat, "stage1"):
            people = pose.step(frame)

        for p in people:
            if not p["triggered"]:
                continue
            if not pose_trig:
                pose_trig, first_t = True, round(t, 2)
            # Stage 2 runs once per triggered track,
            # capped per clip to bound GPU cost.
            if (sam3 is not None and p["track_id"] not in verified_tracks
                    and len(verified_tracks) < args.max_sam3_calls):
                crop = crop_person(frame, p["box"], pad=0.3)
                if crop is None:
                    continue
                with timer(lat, "sam3"):
                    s, _ = sam3.score(crop)
                sam3_scores.append(s)
                verified_tracks.add(p["track_id"])
                if crop_dir is not None:
                    cv2.imwrite(str(crop_dir / f"{Path(path).stem}_t{p['track_id']}_{s:.2f}.jpg"), crop)

        if detector is not None:
            with timer(lat, "det"):
                r = detector(frame, conf=0.01, imgsz=640, device=args.device, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                c = float(r.boxes.conf.max())
                det_max = max(det_max, c)
                det_hits += int(c >= 0.25)

    s1, s1p = summarise_ms(lat.get("stage1", [])[1:])  # drop warm-up frame
    sm, smp = summarise_ms(lat.get("sam3", []))
    dm, dmp = summarise_ms(lat.get("det", [])[1:])
    return {
        "n_frames": n,
        "pose_triggered": int(pose_trig),
        "pose_first_trigger_s": first_t,
        "n_sam3_calls": len(sam3_scores),
        "sam3_max_score": round(max(sam3_scores), 4) if sam3_scores else 0.0,
        "det_max_conf": round(det_max, 4),
        "det_frames_hit": det_hits,
        "stage1_ms_mean": round(s1, 2), "stage1_ms_p95": round(s1p, 2),
        "sam3_ms_mean": round(sm, 2), "sam3_ms_p95": round(smp, 2),
        "det_ms_mean": round(dm, 2), "det_ms_p95": round(dmp, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--out", default="results/raw_scores.csv")
    ap.add_argument("--pose", default="yolo26n-pose.pt")
    ap.add_argument("--detector", default="models/cigarette_yolo26n_ge2340.pt")
    ap.add_argument("--sam3", default="models/sam3.pt", help="'' to skip SAM 3")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--sustain", type=int, default=9)
    ap.add_argument("--dist-thr", type=float, default=0.14)
    ap.add_argument("--max-sam3-calls", type=int, default=3)
    ap.add_argument("--device", default=0)
    ap.add_argument("--save-crops", action="store_true")
    args = ap.parse_args()

    with open(args.labels, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file")]

    print(f"{len(rows)} clips in {args.labels}; loading models...", flush=True)
    pose = PoseScreener(args.pose, dist_thr=args.dist_thr, sustain=args.sustain,
                        device=args.device)
    detector = YOLO(args.detector) if args.detector else None
    sam3 = None
    if args.sam3:
        from stage2_sam3 import Sam3Verifier
        sam3 = Sam3Verifier(args.sam3, prompt="cigarette")
    print("Models loaded. Processing clips...", flush=True)

    crop_dir = None
    if args.save_crops:
        crop_dir = Path(args.out).parent / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(rows, 1):
            path = Path(args.clips) / row["file"]
            if not path.exists():
                print(f"[skip] missing {path}", flush=True)
                continue
            t0 = time.time()
            res = evaluate_clip(path, pose, sam3, detector, args, crop_dir)
            out = {k: row.get(k, "") for k in ("file", "person", "gesture", "distance",
                                                "is_smoking", "split")}
            out.update(res)
            w.writerow(out)
            f.flush()
            print(f"[{i}/{len(rows)}] {row['file']}: pose={res['pose_triggered']} "
                  f"sam3={res['sam3_max_score']:.2f} det={res['det_max_conf']:.2f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
