# Self-recorded evaluation clips

The clips are **not** in this repository (they show participants' faces); they are kept in a
private Google Drive folder `MyDrive/ge2340/clips/`. `labels.csv` is generated from the file names:

```
python scripts/make_labels.py <clip_dir> [<clip_dir> ...] > data/labels.csv
```

## File names
`<person>_<gesture>_<distance>_<take>.<ext>`, e.g. `p1_smoking_near_3.mp4`, `p2_phone_far_2.mov`.

| Field | Values |
|---|---|
| person | `p1`, `p2`, ... (anonymous IDs; each recorded at home with their own phone) |
| gesture | `smoking` (prop cigarette held), `smoking_mime` (smoking pose, **empty hand**), `drinking`, `phone`, `eating`, `rubbing_eye` (incl. hand on chin/mouth), `pen_mouth` |
| distance | `near` (~1.5 m), `mid`, `far` (≥ 3 m) |

## Ground truth
`is_smoking = 1` only for `smoking`. `smoking_mime` is a hard negative: the pose matches smoking
but there is no cigarette, which is exactly the case Stage 2 (SAM 3) is meant to reject.

## Dev/test split (fixed before any results were seen)
- person/gesture recorded at several distances: `mid` clips are dev, `near` and `far` are test;
- person/gesture recorded at one distance only: take 1 is dev (takes 1–2 for `smoking`), the rest test.

## Recording notes
- p1: all clips at near distance, phone on a desk pointing upwards (low camera angle).
- p2: near/mid/far for every gesture, camera at roughly chest height.
- p3: portrait video, sitting (near) and standing (mid/far); smoking with prop at all three distances, plus pen_mouth and drinking.
