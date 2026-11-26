"""Pick thresholds on dev, freeze them, and report test-set results.

Step 1 (dev):   python src/analyze.py --freeze
                -> sweeps thresholds for A2 and A3 on split=="dev", keeps the
                   best-F1 value, writes results/frozen_thresholds.json.
                   Refuses to overwrite an existing file unless --force.
Step 2 (test):  python src/analyze.py
                -> loads the frozen thresholds and writes, for split=="test":
                   results/metrics_test.csv           (Table: 3 approaches)
                   results/false_alarms_by_gesture.csv (Table: confusing gestures)
                   results/recall_by_distance.csv      (Table: near vs far)
                   results/latency_summary.csv         (Table: per-stage ms)
                   results/fig_*.png                   (figures for the report)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
NAMES = {"A1": "Pose only", "A2": "YOLO26n cigarette detector",
         "A3": "Pose + SAM 3 (ours)"}


def decisions(df, th):
    return {
        "A1": df["pose_triggered"].astype(int) == 1,
        "A2": df["det_max_conf"] >= th["A2"],
        "A3": (df["pose_triggered"].astype(int) == 1) & (df["sam3_max_score"] >= th["A3"]),
    }


def prf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    tn = int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec, F1=f1)


def sweep(dev, col, gate=None):
    y = dev["is_smoking"].astype(int) == 1
    rows = []
    for t in GRID:
        p = dev[col] >= t
        if gate is not None:
            p = p & gate
        rows.append({"threshold": t, **prf(y, p)})
    table = pd.DataFrame(rows)
    best = table.sort_values(["F1", "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), table


def freeze(df, out_dir, force):
    path = out_dir / "frozen_thresholds.json"
    if path.exists() and not force:
        raise SystemExit(f"{path} exists. Thresholds are frozen; use --force only if "
                         "you are deliberately re-doing the dev step (and say so in the report).")
    dev = df[df["split"] == "dev"]
    if dev.empty:
        raise SystemExit("No rows with split == 'dev'.")
    t2, s2 = sweep(dev, "det_max_conf")
    t3, s3 = sweep(dev, "sam3_max_score", gate=dev["pose_triggered"].astype(int) == 1)
    s2.to_csv(out_dir / "dev_sweep_A2.csv", index=False)
    s3.to_csv(out_dir / "dev_sweep_A3.csv", index=False)
    path.write_text(json.dumps({"A2": t2, "A3": t3, "n_dev_clips": len(dev)}, indent=2))
    print(f"Frozen thresholds -> {path}: A2={t2}, A3={t3}")

    plt.figure(figsize=(4.5, 3))
    plt.plot(s3["threshold"], s3["precision"], label="precision")
    plt.plot(s3["threshold"], s3["recall"], label="recall")
    plt.plot(s3["threshold"], s3["F1"], label="F1")
    plt.axvline(t3, ls="--", c="grey")
    plt.xlabel("SAM 3 score threshold"); plt.ylabel("dev score"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "fig_dev_threshold_A3.png", dpi=200)


def report(df, out_dir):
    th = json.loads((out_dir / "frozen_thresholds.json").read_text())
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise SystemExit("No rows with split == 'test'.")
    y = test["is_smoking"].astype(int) == 1
    dec = decisions(test, th)

    # Table 1: overall metrics
    m = pd.DataFrame([{"approach": NAMES[k], **prf(y, v)} for k, v in dec.items()])
    m.to_csv(out_dir / "metrics_test.csv", index=False, float_format="%.3f")

    # Table 2: false alarms on non-smoking gestures
    neg = test[~y]
    fa = pd.DataFrame({NAMES[k]: v[~y].groupby(neg["gesture"]).sum() for k, v in dec.items()})
    fa.insert(0, "clips", neg.groupby("gesture").size())
    fa.to_csv(out_dir / "false_alarms_by_gesture.csv")

    # Table 3: recall by distance (small-cigarette question)
    pos = test[y]
    rd = pd.DataFrame({NAMES[k]: v[y].groupby(pos["distance"]).mean() for k, v in dec.items()})
    rd.insert(0, "clips", pos.groupby("distance").size())
    rd.to_csv(out_dir / "recall_by_distance.csv", float_format="%.3f")

    # Table 4: latency
    lat = {
        "Stage 1 pose (per frame)": test["stage1_ms_mean"],
        "Stage 2 SAM 3 (per crop)": test.loc[test["n_sam3_calls"] > 0, "sam3_ms_mean"],
        "A2 detector (per frame)": test["det_ms_mean"],
    }
    lt = pd.DataFrame([{"component": k, "mean_ms": v.mean(), "p95_of_clip_means_ms": v.quantile(0.95)}
                       for k, v in lat.items()])
    calls = test["n_sam3_calls"].sum()
    lt.to_csv(out_dir / "latency_summary.csv", index=False, float_format="%.1f")

    print(f"Thresholds (frozen on dev): {th}")
    print("\nTable 1 - test metrics\n", m.round(3).to_string(index=False))
    print("\nTable 2 - false alarms by gesture (count of clips)\n", fa.to_string())
    print("\nTable 3 - recall by distance\n", rd.round(3).to_string())
    print("\nTable 4 - latency\n", lt.round(1).to_string(index=False))
    print(f"\nSAM 3 calls on test: {calls} for {len(test)} clips "
          f"({test['n_frames'].sum()} sampled frames)")

    ax = m.set_index("approach")[["precision", "recall", "F1"]].plot.bar(figsize=(5.5, 3), rot=0)
    ax.set_ylim(0, 1); ax.set_ylabel("test score"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(out_dir / "fig_metrics_test.png", dpi=200); plt.close()

    ax = fa.drop(columns="clips").plot.bar(figsize=(6, 3), rot=20)
    ax.set_ylabel("false alarms (clips)"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(out_dir / "fig_false_alarms.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw_scores.csv")
    ap.add_argument("--freeze", action="store_true", help="choose thresholds on dev")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(args.raw)
    out_dir = Path(args.raw).parent
    if args.freeze:
        freeze(df, out_dir, args.force)
    else:
        report(df, out_dir)


if __name__ == "__main__":
    main()
