# Self-recorded evaluation clips

The clips themselves are **not** in this repository (they show group members' faces).
They live in a private Google Drive folder `MyDrive/ge2340/clips/`.
`labels.csv` is the shot list *and* the ground truth.

## Shot list

| Field | Values |
|---|---|
| person | `p1`–`p4` (anonymous IDs) |
| gesture | `smoking` (unlit prop cigarette), `drinking`, `phone`, `eating`, `rubbing_eye`, `pen_mouth` |
| distance | `near` (1–2 m from camera), `far` (4–5 m: the small-cigarette case) |
| take | 1, 2, 3 |

4 people x 6 gestures x 2 distances x 3 takes = 144 clips, 5–10 s each.
File name: `<person>_<gesture>_<distance>_<take>.mp4`, e.g. `p2_drinking_far_3.mp4`.

## Recording rules
- Same phone, landscape, 1080p, 30 fps, roughly chest height; keep it fixed (tripod or on a table).
- Whole upper body in frame; the gesture starts within the first 2 s and lasts at least 2 s.
- Never light a real cigarette on campus: use an unlit prop or rolled white paper.
- Vary lighting/background between takes where possible; note anything unusual in `notes`.

## Labels and split
- `is_smoking` = 1 only for `smoking` clips (clip-level label, no boxes).
- `split`: all clips of **p1 are `dev`** (used only to choose thresholds), p2–p4 are `test`.
  Splitting by person stops the same person's mannerisms appearing in both sets.
- Do not change the split after looking at test results.
