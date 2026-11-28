# AI-Powered Smoking Detection on Campus (GE2340 Phase 2)

Two-stage computer-vision pipeline that flags *suspected* smoking in camera footage:

```
video frame ──► Stage 1: YOLO26n-pose + hand-to-mouth rule ──► person crop (+30% pad)
                (every 5th frame, sustained 9 inferences ≈ 1.5 s)          │
                                                                           ▼
                                   alert ◄── score ≥ frozen threshold ◄── Stage 2: SAM 3, prompt "cigarette"
```

No face recognition or identification is performed; output is an alert with a confidence score.

## Approaches compared (from the Phase 1 proposal)

| ID | Approach | Decision rule |
|---|---|---|
| A1 | Pose only | sustained hand-to-mouth gesture |
| A2 | Object detection only | our YOLO26n cigarette detector, max confidence ≥ threshold |
| A3 | **Pose + SAM 3 (proposed)** | A1 **and** SAM 3 "cigarette" score on the person crop ≥ threshold |

Thresholds for A2 and A3 are chosen on the **dev** clips only, written to
`results/frozen_thresholds.json`, and then left unchanged for the test clips.

## GPU latency budget (Tesla T4)

| Component | Budget |
|---|---|
| Stage 1 pose, per frame | ≤ 33 ms (keeps up with 30 fps) |
| Stage 2 SAM 3, per person crop | ≤ 1500 ms (float16 autocast) |
| Gesture onset → alert | ≤ 3 s (1.5 s sustain window + Stage 1 + Stage 2) |

SAM 3 runs only once per triggered person track (max 3 per clip), not on every frame.
On a T4, SAM 3 under bfloat16 autocast took ~3.2 s per crop (bf16 is emulated on this GPU);
switching to float16 autocast gave ~1.25 s with the same score, so float16 is used.
Measured values are written to `results/latency_summary.csv`.

## Repository layout

```
src/common.py           frame sampling, person crop, GPU-synchronised timer
src/stage1_pose.py      Stage 1: pose screening rule + ByteTrack sustain counter
src/stage2_sam3.py      Stage 2: SAM 3 text-prompt verification
src/run_eval.py         one GPU pass over all clips -> results/raw_scores.csv
src/analyze.py          dev threshold selection (--freeze), then test tables + figures
src/image_eval.py       image-level recall by cigarette size on Roboflow validation images
scripts/make_labels.py  builds data/labels.csv from clip file names
scripts/train_cigarette_detector.py   training script for the A2 baseline
notebooks/colab_eval.ipynb            end-to-end Colab notebook (T4)
data/labels.csv         shot list + clip-level ground truth (clips themselves kept private)
models/cigarette_yolo26n_ge2340.pt    A2 baseline weights (5.5 MB)
results/cigarette_detector/           training curves + validation metrics of A2
```

## Setup on Colab

Open `notebooks/colab_eval.ipynb` in Colab with a **T4 GPU** runtime and run the cells in order.
Summary:

```bash
pip install "ultralytics>=8.4.0" pandas matplotlib huggingface_hub
pip install --no-deps git+https://github.com/facebookresearch/sam3.git
pip install timm ftfy regex iopath einops psutil "setuptools<81"
# SAM 3 checkpoint -> models/sam3.pt (see notebook)
python src/run_eval.py --labels data/labels.csv --clips /content/drive/MyDrive/ge2340/clips
python src/analyze.py --freeze     # dev only
python src/analyze.py              # test tables and figures
```

`yolo26n-pose.pt` is downloaded automatically by Ultralytics on first use.

## A2 baseline: cigarette detector

YOLO26n (COCO-pretrained) fine-tuned on Roboflow Universe *Cigarette Detection*
(`dpp/cigarette-detection-bow3d`, v11; 6,503 train / 1,212 val images), 50 epochs requested,
imgsz 640, batch 32, seed 0, Colab T4. Validation split (the dataset has no test split):

| Precision | Recall | mAP50 | mAP50-95 | Inference |
|---|---|---|---|---|
| 0.725 | 0.625 | 0.666 | 0.223 | 3.9 ms / image |

The large gap between mAP50 and mAP50-95 shows that cigarettes are found but poorly localised,
which is consistent with their small, thin shape.

## Credits

- Ultralytics YOLO26 (pose estimation and the A2 detector).
- SAM 3: N. Carion et al., "SAM 3: Segment Anything with Concepts" (Meta, 2025). Checkpoint
  obtained from a Hugging Face mirror of `facebook/sam3` and used under Meta's SAM license.
- Cigarette Detection dataset, Roboflow Universe (`dpp/cigarette-detection-bow3d`, v11).
