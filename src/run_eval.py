"""Run all three approaches on every labelled clip and log raw scores.

One GPU pass produces
  results/raw_scores.csv  one row per clip (scores + latency)
  results/pose_seqs.json  per-frame hand-to-mouth ratio for every tracked person
No decision thresholds are applied here: analyze.py chooses them on the dev
split, freezes them, then scores the test split.

Per clip:
  Stage 1 : YOLO26n-pose on every `stride`-th frame -> hand-to-mouth ratio per person
  Stage 2 : SAM 3 "cigarette" score on the person crop at the frame with the
            smallest ratio (the moment the hand is closest to the mouth)
  A2      : max confidence of our YOLO26n cigarette detector over the same frames
A1 = Stage 1 sustained trigger; A3 = A1 and Stage-2 score >= threshold.

Usage (Colab):
  python -u src/run_eval.py --labels data/labels.csv --clips /content/drive/MyDrive/ge2340/clips
"""
import argparse
import csv
import json
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
    "min_ratio", "best_frame_s", "n_sam3_calls", "sam3_score",
    "det_max_conf", "det_frames_hit",
    "stage1_ms_mean", "stage1_ms_p95", "sam3_ms_mean", "det_ms_mean", "det_ms_p95",
]


def evaluate_clip(path, pose, sam3, detector, args, crop_dir=None):
    pose.reset()
    lat, seqs = {}, {}
    best = None  # (ratio, t, crop)
    det_max, det_hits, n = 0.0, 0, 0

    for k, (fidx, t, frame) in enumerate(iter_frames(path, args.stride)):
        n += 1
        with timer(lat, "stage1"):
            people = pose.step(frame)
        for p in people:
            r = p["ratio"]
            seqs.setdefault(str(p["track_id"]), []).append(
                [k, round(t, 3), None if r is None else round(r, 4)])
            if r is not None and (best is None or r < best[0]):
                crop = crop_person(frame, p["box"], pad=0.3)
                if crop is not None:
                    best = (r, t, crop)

        if detector is not None:
            with timer(lat, "det"):
                res = detector(frame, conf=0.01, imgsz=640, device=args.device, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                c = float(res.boxes.conf.max())
                det_max = max(det_max, c)
                det_hits += int(c >= 0.25)

    sam3_score, calls = 0.0, 0
    if sam3 is not None and best is not None:
        with timer(lat, "sam3"):
            sam3_score, _ = sam3.score(best[2])
        calls = 1
        if crop_dir is not None:
            cv2.imwrite(str(crop_dir / f"{Path(path).stem}_r{best[0]:.2f}_s{sam3_score:.2f}.jpg"), best[2])

    s1, s1p = summarise_ms(lat.get("stage1", [])[1:])  # drop warm-up frame
    sm, _ = summarise_ms(lat.get("sam3", []))
    dm, dmp = summarise_ms(lat.get("det", [])[1:])
    row = {
        "n_frames": n,
        "min_ratio": round(best[0], 4) if best else "",
        "best_frame_s": round(best[1], 2) if best else "",
        "n_sam3_calls": calls,
        "sam3_score": round(sam3_score, 4),
        "det_max_conf": round(det_max, 4),
        "det_frames_hit": det_hits,
        "stage1_ms_mean": round(s1, 2), "stage1_ms_p95": round(s1p, 2),
        "sam3_ms_mean": round(sm, 2),
        "det_ms_mean": round(dm, 2), "det_ms_p95": round(dmp, 2),
    }
    return row, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--out", default="results/raw_scores.csv")
    ap.add_argument("--pose", default="yolo26n-pose.pt")
    ap.add_argument("--detector", default="models/cigarette_yolo26n_ge2340.pt")
    ap.add_argument("--sam3", default="models/sam3.pt", help="'' to skip SAM 3")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default=0)
    ap.add_argument("--save-crops", action="store_true")
    args = ap.parse_args()

    with open(args.labels, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file")]

    print(f"{len(rows)} clips in {args.labels}; loading models...", flush=True)
    pose = PoseScreener(args.pose, device=args.device)
    detector = YOLO(args.detector) if args.detector else None
    sam3 = None
    if args.sam3:
        from stage2_sam3 import Sam3Verifier
        sam3 = Sam3Verifier(args.sam3, prompt="cigarette")
    print("Models loaded. Processing clips...", flush=True)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = None
    if args.save_crops:
        crop_dir = out_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

    all_seqs = {}
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(rows, 1):
            path = Path(args.clips) / row["file"]
            if not path.exists():
                print(f"[skip] missing {path}", flush=True)
                continue
            t0 = time.time()
            res, seqs = evaluate_clip(path, pose, sam3, detector, args, crop_dir)
            all_seqs[row["file"]] = seqs
            out = {k: row.get(k, "") for k in ("file", "person", "gesture", "distance",
                                                "is_smoking", "split")}
            out.update(res)
            w.writerow(out)
            f.flush()
            print(f"[{i}/{len(rows)}] {row['file']}: min_ratio={res['min_ratio']} "
                  f"sam3={res['sam3_score']:.2f} det={res['det_max_conf']:.2f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    (out_dir / "pose_seqs.json").write_text(json.dumps(all_seqs))
    print(f"Saved {args.out} and {out_dir / 'pose_seqs.json'}", flush=True)


if __name__ == "__main__":
    main()
