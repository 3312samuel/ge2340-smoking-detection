"""Image-level check on the Roboflow validation split (complements the clip test).

Why: our own clips contain only 15 smoking clips, too few to measure recall or
the effect of cigarette size. The Roboflow "Cigarette Detection" v11 validation
split (1,212 images, 1,124 boxes) gives many positives with a wide size range.

For every image we record
  * gt_positive        : image has at least one cigarette box
  * gt_size            : COCO size bucket of the LARGEST box in the image
                         (small < 32^2 px, medium < 96^2 px, large otherwise,
                         measured at 640x640, the resolution of the export)
  * sam3_score         : best SAM 3 "cigarette" score on the full image
  * det_score          : best confidence of our YOLO26n cigarette detector
  * sam3_ms, det_ms    : per-image latency
Then, with the thresholds frozen on the clip dev split (results/frozen_thresholds.json),
we report precision / recall overall and recall per size bucket.

Caveat for the report: the YOLO26n detector used this validation split for
model selection (best.pt), so its numbers here are optimistic. SAM 3 never saw it.

Usage (Colab):
  python src/image_eval.py --data /content/Cigarette-Detection-11 --max-images 400
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

SIZE_ORDER = ["small", "medium", "large"]


def size_bucket(w_norm, h_norm, img_size=640):
    area = (w_norm * img_size) * (h_norm * img_size)
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def read_labels(label_path):
    """YOLO txt -> list of (w_norm, h_norm). Missing/empty file = background."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((float(parts[3]), float(parts[4])))
    return boxes


def collect(data_dir, split, max_images, seed):
    img_dir = Path(data_dir) / split / "images"
    lab_dir = Path(data_dir) / split / "labels"
    items = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        boxes = read_labels(lab_dir / (img.stem + ".txt"))
        largest = max(boxes, key=lambda b: b[0] * b[1]) if boxes else None
        items.append({
            "image": img.name, "path": str(img),
            "gt_positive": int(bool(boxes)), "n_boxes": len(boxes),
            "gt_size": size_bucket(*largest) if largest else "none",
        })
    if max_images and len(items) > max_images:
        # keep every background image (they are scarce), sample the positives
        rng = random.Random(seed)
        neg = [i for i in items if not i["gt_positive"]]
        pos = [i for i in items if i["gt_positive"]]
        items = neg + rng.sample(pos, max(0, max_images - len(neg)))
    return items


def run(args):
    from ultralytics import YOLO
    from common import timer
    from stage2_sam3 import Sam3Verifier

    items = collect(args.data, args.split, args.max_images, args.seed)
    print(f"{len(items)} images ({sum(i['gt_positive'] for i in items)} with cigarettes)")
    sam3 = Sam3Verifier(args.sam3, prompt="cigarette")
    det = YOLO(args.detector)

    rows = []
    for n, it in enumerate(items, 1):
        img = cv2.imread(it["path"])
        lat = {}
        with timer(lat, "sam3"):
            s, _ = sam3.score(img)
        with timer(lat, "det"):
            r = det(img, conf=0.01, imgsz=640, device=args.device, verbose=False)[0]
        d = float(r.boxes.conf.max()) if r.boxes is not None and len(r.boxes) else 0.0
        rows.append({**{k: it[k] for k in ("image", "gt_positive", "n_boxes", "gt_size")},
                     "sam3_score": round(s, 4), "det_score": round(d, 4),
                     "sam3_ms": round(lat["sam3"][0], 1), "det_ms": round(lat["det"][0], 1)})
        if n % 50 == 0:
            print(f"[{n}/{len(items)}]")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    return df


def summarise(df, out_dir, thresholds):
    out_dir = Path(out_dir)
    y = df["gt_positive"] == 1
    models = {"SAM 3 (zero-shot)": ("sam3_score", thresholds["A3"]),
              "YOLO26n detector": ("det_score", thresholds["A2"])}
    rows, size_rows = [], []
    for name, (col, t) in models.items():
        p = df[col] >= t
        tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        lat = df["sam3_ms" if col == "sam3_score" else "det_ms"].iloc[1:]  # drop warm-up
        rows.append({"model": name, "threshold": t, "TP": tp, "FP": fp, "FN": fn,
                     "precision": prec, "recall": rec,
                     "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                     "mean_ms": lat.mean()})
        for s in SIZE_ORDER:
            m = y & (df["gt_size"] == s)
            size_rows.append({"model": name, "size": s, "images": int(m.sum()),
                              "recall": float(p[m].mean()) if m.any() else float("nan")})
    summary = pd.DataFrame(rows)
    sr = pd.DataFrame(size_rows)
    by_size = sr.pivot(index="size", columns="model", values="recall").reindex(SIZE_ORDER)
    by_size.insert(0, "images", sr.groupby("size")["images"].first().reindex(SIZE_ORDER).fillna(0).astype(int))
    summary.to_csv(out_dir / "image_eval_summary.csv", index=False, float_format="%.3f")
    by_size.to_csv(out_dir / "image_eval_recall_by_size.csv", float_format="%.3f")
    print("\nImage-level results (thresholds frozen on clip dev split)\n",
          summary.round(3).to_string(index=False))
    print("\nRecall by cigarette size (largest box per image)\n", by_size.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = by_size.drop(columns="images").plot.bar(figsize=(5, 3), rot=0)
    ax.set_ylim(0, 1); ax.set_ylabel("recall"); ax.set_xlabel("cigarette size (COCO buckets)")
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(SIZE_ORDER, by_size["images"])])
    plt.tight_layout(); plt.savefig(out_dir / "fig_recall_by_size.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/Cigarette-Detection-11")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--sam3", default="models/sam3.pt")
    ap.add_argument("--detector", default="models/cigarette_yolo26n_ge2340.pt")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="results/image_eval_raw.csv")
    ap.add_argument("--summary-only", action="store_true", help="re-summarise an existing raw CSV")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    th_path = Path(args.out).parent / "frozen_thresholds.json"
    if not th_path.exists():
        raise SystemExit("Run `python src/analyze.py --freeze` on the clip dev split first.")
    thresholds = json.loads(th_path.read_text())

    df = pd.read_csv(args.out) if args.summary_only else run(args)
    summarise(df, Path(args.out).parent, thresholds)


if __name__ == "__main__":
    main()
