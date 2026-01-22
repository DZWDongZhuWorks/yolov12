# app_utils/polygon_utils.py
from typing import Any, Dict, List, Optional
import numpy as np
import cv2


def _simplify_segment(seg: np.ndarray, mode: str, eps_ratio: float) -> np.ndarray:
    """
    seg: (N, 2) float32
    mode:
      - "none"        : 不做簡化
      - "convex_hull" : 取凸包
      - "rdp"         : 用 approxPolyDP 做 RDP 簡化
    eps_ratio: 相對於「該 segment 外接矩形的較長邊」的比例
    """
    if seg.shape[0] <= 3 or mode == "none":
        return seg

    if mode == "convex_hull":
        hull = cv2.convexHull(seg)
        return hull.reshape(-1, 2)

    # rdp / approxPolyDP
    x_min, y_min = seg.min(axis=0)
    x_max, y_max = seg.max(axis=0)
    size = max(x_max - x_min, y_max - y_min)
    eps = float(size) * eps_ratio

    approx = cv2.approxPolyDP(seg, eps, closed=True)
    return approx.reshape(-1, 2)


def build_objects_from_result(
    result,
    allowed_class_ids: Optional[List[int]] = None,
    simplify_mode: str = "none",      # "none" / "convex_hull" / "rdp"
    simplify_eps_ratio: float = 0.01, # 只對 "rdp" 有效
    split_components: bool = False,   # ★ 新增：是否拆分連通域
) -> List[Dict[str, Any]]:
    """
    從單一個 YOLO result 產生標準化的物件資訊（含 polygon）。
    之後畫面繪製 & JSON 輸出都只用這個。
    """
    names = getattr(result, "names", {}) or {}

    if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls_arr = result.boxes.cls.cpu().numpy().astype(int)

    # conf 可能不存在，保護一下
    try:
        conf_arr = result.boxes.conf.cpu().numpy()
    except Exception:
        conf_arr = None

    has_masks = getattr(result, "masks", None) is not None
    raw_polys = getattr(result.masks, "xy", None) if has_masks else None
    
    objects: List[Dict[str, Any]] = []

    for i, (box, cid) in enumerate(zip(xyxy, cls_arr)):
        # 類別篩選
        if allowed_class_ids is not None and cid not in set(allowed_class_ids):
            continue

        orig_x1, orig_y1, orig_x2, orig_y2 = [float(v) for v in box.tolist()]
        class_name = names.get(cid, str(cid))
        conf = float(conf_arr[i]) if conf_arr is not None and i < len(conf_arr) else None

        # --- 產生 polygons ---
        if has_masks:
            if split_components:
                # 【模式 A】拆分模式：使用 OpenCV 找出連通域 (修正版)
                if raw_polys is not None and i < len(raw_polys):
                    # 1. 取得原始圖片尺寸，建立全尺寸的空白 mask
                    h, w = result.orig_shape[:2]
                    full_mask = np.zeros((h, w), dtype=np.uint8)

                    # 2. 將已在正確座標系上的 polygon 畫到 full_mask 上
                    item = raw_polys[i]
                    segments = item if isinstance(item, list) else [item]
                    pts_list = [np.asarray(seg, dtype=np.int32) for seg in segments if seg is not None and len(seg) >= 3]
                    if pts_list:
                        cv2.fillPoly(full_mask, pts_list, color=255)
                    
                    # 3. 在這個準確的 full_mask 上找連通域
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)

                    # 遍歷所有找到的 component (label 0 是背景，跳過)
                    for label in range(1, num_labels):
                        # 4. 建立該 component 的 mask
                        component_mask = np.zeros_like(full_mask)
                        component_mask[labels == label] = 255
                        
                        # 5. 從 mask 找輪廓 (contour)
                        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if not contours: continue

                        for contour in contours:
                            arr = contour.reshape(-1, 2).astype(np.float32)
                            if arr.shape[0] < 3: continue

                            # 6. 重新計算該分量的 BBox
                            new_x1, new_y1, b_w, b_h = cv2.boundingRect(arr)
                            new_x2, new_y2 = new_x1 + b_w, new_y1 + b_h
                            
                            # 7. 簡化多邊形
                            arr_simplified = _simplify_segment(arr, simplify_mode, simplify_eps_ratio)
                            
                            objects.append({
                                "class_id": int(cid),
                                "class_name": class_name,
                                "confidence": conf,
                                "bbox_xyxy": [float(new_x1), float(new_y1), float(new_x2), float(new_y2)],
                                "polygons": [arr_simplified.astype(float).tolist()],
                            })
                
            else:
                # 【模式 B】不拆分模式：維持原樣，一個物件可包含多個 polygons
                if raw_polys is not None and i < len(raw_polys):
                    item = raw_polys[i]
                    segments = item if isinstance(item, list) else [item]
                    
                    polys = []
                    for seg in segments:
                        if seg is None: continue
                        arr = np.asarray(seg, dtype=np.float32)
                        if arr.ndim == 1: arr = arr.reshape(-1, 2)
                        if arr.shape[0] < 3: continue
                        
                        arr_simplified = _simplify_segment(arr, simplify_mode, simplify_eps_ratio)
                        polys.append(arr_simplified.astype(float).tolist())
                    
                    if polys:
                        objects.append({
                            "class_id": int(cid),
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox_xyxy": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "polygons": polys,
                        })
        else:
            # 沒有 mask：用 bbox 當成一個矩形 polygon
            objects.append({
                "class_id": int(cid),
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": [orig_x1, orig_y1, orig_x2, orig_y2],
                "polygons": [[ [orig_x1, orig_y1], [orig_x2, orig_y1], [orig_x2, orig_y2], [orig_x1, orig_y2] ]],
            })

    return objects
