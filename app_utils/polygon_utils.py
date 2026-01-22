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


def _close_ring(points: List[List[float]]) -> List[List[float]]:
    """
    確保 polygon ring 首尾相同（GeoJSON 需要閉合 ring）。
    """
    if len(points) < 3:
        return points
    first = points[0]
    last = points[-1]
    if len(first) >= 2 and len(last) >= 2 and first[0] == last[0] and first[1] == last[1]:
        return points
    return points + [first]


def build_objects_from_result(
    result,
    allowed_class_ids: Optional[List[int]] = None,
    simplify_mode: str = "none",      # "none" / "convex_hull" / "rdp"
    simplify_eps_ratio: float = 0.01, # 只對 "rdp" 有效
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

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        class_name = names.get(cid, str(cid))
        conf = float(conf_arr[i]) if conf_arr is not None and i < len(conf_arr) else None

        # --- 產生 polygons ---
        polys: List[List[List[float]]] = []

        if has_masks and raw_polys is not None and i < len(raw_polys):
            item = raw_polys[i]

            # YOLO 可能是 ndarray 或 list[ndarray]
            segments = item if isinstance(item, list) else [item]

            for seg in segments:
                if seg is None:
                    continue
                arr = np.asarray(seg, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 2)
                if arr.shape[0] < 3:
                    continue

                arr = _simplify_segment(arr, simplify_mode, simplify_eps_ratio)
                polys.append(_close_ring(arr.astype(float).tolist()))
        else:
            # 沒有 mask：用 bbox 當成一個矩形 polygon
            polys.append(
                _close_ring(
                    [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ]
                )
            )

        objects.append(
            {
                "class_id": int(cid),
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "polygons": polys,
            }
        )

    return objects
