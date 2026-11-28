"""Build data/labels.csv from clip file names.

File name: <person>_<gesture>_<distance>_<take>.<ext>
  e.g. p1_smoking_near_3.mp4, p2_smoking_mime_far_1.mov
Ground truth: is_smoking = 1 only for gesture "smoking" (a prop cigarette is held).
"smoking_mime" (smoking pose, empty hand) is a hard negative.

Dev/test split (fixed before any results are seen):
  * a person/gesture recorded at several distances -> the "mid" clips are dev;
  * a person/gesture recorded at one distance only -> take 1 is dev
    (takes 1-2 for "smoking", which has 6 takes);
  * everything else is test.

Usage: python scripts/make_labels.py <clip_dir> [<clip_dir> ...] > data/labels.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

VIDEO = {".mp4", ".mov", ".m4v", ".avi"}


def parse(name):
    parts = Path(name).stem.split("_")
    person, distance, take = parts[0], parts[-2], int(parts[-1])
    return person, "_".join(parts[1:-2]), distance, take


files = sorted(p.name for d in sys.argv[1:] for p in Path(d).iterdir()
               if p.suffix.lower() in VIDEO and not p.name.startswith(("_", ".")))
rows = [(f, *parse(f)) for f in files]
distances = defaultdict(set)
for f, person, gesture, distance, take in rows:
    distances[(person, gesture)].add(distance)

w = csv.writer(sys.stdout)
w.writerow(["file", "person", "gesture", "distance", "is_smoking", "split", "notes"])
for f, person, gesture, distance, take in rows:
    if len(distances[(person, gesture)]) > 1:
        dev = distance == "mid"
    else:
        dev = take <= (2 if gesture == "smoking" else 1)
    note = "no prop cigarette" if gesture == "smoking_mime" else ""
    w.writerow([f, person, gesture, distance, int(gesture == "smoking"),
                "dev" if dev else "test", note])
