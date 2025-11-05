# app_utils/config.py
import os

DEFAULT_MODELS = [
    "yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt", "yolov12x.pt"
]

PERSIST_FILE = os.environ.get("YOLOv12_MODEL_MEMO_FILE", "model_choices.json")

# 多模型比較：最多同時比較幾個
MAX_MODELS = 5
