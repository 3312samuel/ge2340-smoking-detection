"""Train the A2 baseline: a single-class YOLO26n cigarette detector.

This is the exact configuration used for models/cigarette_yolo26n_ge2340.pt
(Colab, Tesla T4). The Roboflow API key is read from the environment so it is
never committed:  export ROBOFLOW_API_KEY=...
"""
import os

from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
dataset = rf.workspace("dpp").project("cigarette-detection-bow3d").version(11).download("yolo26")

model = YOLO("yolo26n.pt")  # COCO-pretrained YOLO26 nano
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50, imgsz=640, batch=32, patience=10, seed=0, device=0,
    project="runs", name="cig_yolo26n",
)
metrics = YOLO("runs/cig_yolo26n/weights/best.pt").val(data=f"{dataset.location}/data.yaml")
print("P", metrics.box.mp, "R", metrics.box.mr,
      "mAP50", metrics.box.map50, "mAP50-95", metrics.box.map)
