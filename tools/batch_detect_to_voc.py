# -*- coding: utf-8 -*-
"""
批次偵測並輸出 Pascal VOC (XML) 與可視化結果圖
相依：ultralytics、opencv-python
Python 3.8+

用法：
python batch_detect_to_voc.py --source /path/to/images_dir --model yolov12m.pt \
  --imgsz 640 --conf 0.25 --out-xml ./runs/voc_xml --out-vis ./runs/vis --recursive
"""
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultralytics import YOLO

import cv2
import glob
import argparse
from ultralytics import YOLO
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(src_path: str, recursive: bool = False):
    if os.path.isfile(src_path):
        ext = os.path.splitext(src_path)[1].lower()
        return [src_path] if ext in IMG_EXTS else []
    pattern = "**/*" if recursive else "*"
    files = glob.glob(os.path.join(src_path, pattern), recursive=recursive)
    return [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]

def clamp_int(x, lo, hi):
    return max(lo, min(int(round(x)), hi))

def voc_pretty_print(elem: Element) -> str:
    rough = ElementTree(elem)
    xml_bytes = bytes()
    # ElementTree 沒有直接 pretty-print，改用 minidom
    try:
        import io
        buf = io.BytesIO()
        rough.write(buf, encoding="utf-8", xml_declaration=True)
        xml_str = buf.getvalue().decode("utf-8")
        return minidom.parseString(xml_str).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    except Exception:
        # 後備路徑：至少輸出可用的 XML
        return minidom.parseString(ElementTree.tostring(elem)).toprettyxml(indent="  ")

def make_voc_xml(
    img_path: str,
    width: int,
    height: int,
    depth: int,
    dets: list,
):
    """
    dets: List[dict] with keys: name, xmin, ymin, xmax, ymax
    """
    folder = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    abs_path = os.path.abspath(img_path)

    ann = Element("annotation")
    SubElement(ann, "folder").text = folder
    SubElement(ann, "filename").text = filename
    SubElement(ann, "path").text = abs_path

    source = SubElement(ann, "source")
    SubElement(source, "database").text = "Unknown"

    size = SubElement(ann, "size")
    SubElement(size, "width").text = str(width)
    SubElement(size, "height").text = str(height)
    SubElement(size, "depth").text = str(depth if depth in (1,3,4) else 3)

    SubElement(ann, "segmented").text = "0"

    for d in dets:
        obj = SubElement(ann, "object")
        SubElement(obj, "name").text = d["name"]
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"

        bb = SubElement(obj, "bndbox")
        SubElement(bb, "xmin").text = str(d["xmin"])
        SubElement(bb, "ymin").text = str(d["ymin"])
        SubElement(bb, "xmax").text = str(d["xmax"])
        SubElement(bb, "ymax").text = str(d["ymax"])

    return voc_pretty_print(ann)

def save_text(text: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_image(img_bgr, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img_bgr)

def annotate_image_bgr(result, label_mode: str = "name"):
    """
    使用 Ultralytics 內建的 plot() 產生有框的 BGR 圖。
    label_mode: "name" or "id"
    """
    # 先畫框與內建標籤；若想自定外觀可改成 labels=False 自行 cv2.putText
    img_bgr = result.plot(labels=True, boxes=True, masks=False)
    if label_mode == "id" and hasattr(result, "boxes") and result.boxes is not None:
        # 覆寫為 class id（可選）
        img_bgr = result.plot(labels=False, boxes=True, masks=False)
        names = getattr(result, "names", {}) or {}
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls_arr = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cid in zip(xyxy, cls_arr):
            x1, y1 = int(x1), int(y1)
            txt = str(cid)
            cv2.rectangle(img_bgr, (x1, max(0, y1-18)), (x1 + 8*len(txt) + 6, y1), (0, 0, 0), -1)
            cv2.putText(img_bgr, txt, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img_bgr

def run(args):
    model = YOLO(args.model)
    imgs = list_images(args.source, recursive=args.recursive)
    if not imgs:
        print(f"[WARN] 在 {args.source} 未找到影像（支援副檔名：{sorted(IMG_EXTS)}）")
        return

    os.makedirs(args.out_xml, exist_ok=True)
    os.makedirs(args.out_vis, exist_ok=True)

    print(f"[INFO] 共 {len(imgs)} 張影像，開始偵測...")
    for idx, img_path in enumerate(sorted(imgs)):
        try:
            # 讀尺寸
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[SKIP] 無法讀取影像：{img_path}")
                continue
            h, w = bgr.shape[:2]
            depth = 3 if bgr.ndim == 3 else 1

            # 推理
            results = model.predict(source=img_path, imgsz=args.imgsz, conf=args.conf, verbose=False)
            res = results[0]
            names = getattr(res, "names", {}) or {}

            dets = []
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls_arr = res.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), cid in zip(xyxy, cls_arr):
                    xmin = clamp_int(x1, 1, w)
                    ymin = clamp_int(y1, 1, h)
                    xmax = clamp_int(x2, 1, w)
                    ymax = clamp_int(y2, 1, h)
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    cname = names.get(cid, str(cid))
                    dets.append({
                        "name": cname,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    })

            # 1) VOC XML
            if dets or args.write_empty_xml:
                xml_str = make_voc_xml(img_path, w, h, depth, dets)
                xml_name = os.path.splitext(os.path.basename(img_path))[0] + ".xml"
                save_text(xml_str, os.path.join(args.out_xml, xml_name))

            # 2) 可視化結果圖
            vis_bgr = annotate_image_bgr(res, label_mode="name")
            vis_name = os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
            save_image(vis_bgr, os.path.join(args.out_vis, vis_name))

            print(f"[{idx+1}/{len(imgs)}] {os.path.basename(img_path)} -> dets: {len(dets)}")
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")

    print("[DONE] 全部處理完成。")

def parse_args():
    p = argparse.ArgumentParser(description="Batch detect images and export Pascal VOC XML + visuals")
    p.add_argument("--source", required=True, help="影像資料夾或影像檔")
    p.add_argument("--model", default="yolov12m.pt", help="YOLOv12 權重檔")
    p.add_argument("--imgsz", type=int, default=640, help="推理影像尺寸")
    p.add_argument("--conf", type=float, default=0.25, help="信心值門檻")
    p.add_argument("--out-xml", default="./runs/voc_xml", help="VOC XML 輸出資料夾")
    p.add_argument("--out-vis", default="./runs/vis", help="可視化結果輸出資料夾")
    p.add_argument("--recursive", action="store_true", help="遞迴處理子資料夾")
    p.add_argument("--write-empty-xml", action="store_true", help="即使無偵測也輸出空 XML")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
