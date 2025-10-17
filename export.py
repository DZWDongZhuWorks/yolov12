
from ultralytics import YOLO
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True, help="Path to the trained weights file (e.g., best.pt)")
parser.add_argument("--include", type=str, default="onnx", help="Export format (e.g., onnx, pb, tflite, coreml, etc.)")

args = parser.parse_args()

pt_path = args.weights
export_format = args.include
model = YOLO(pt_path)  
model.export(format=export_format)