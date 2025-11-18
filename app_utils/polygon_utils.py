# app_utils/polygon_utils.py
from typing import Any, Dict, List, Optional
import numpy as np
import cv2


DEFAULT_SIMPLIFY_STRATEGY = "rdp"


def _resolve_class_category(
    class_id: int,
    class_name: str,
    class_category_map: Optional[Dict[Any, str]],
) -> Optional[str]:
    """Return the category name for a class id or class name if configured."""
    if not class_category_map:
        return None

    if class_id in class_category_map:
        return class_category_map[class_id]
    if class_name in class_category_map:
        return class_category_map[class_name]
    return None


def _resolve_simplify_mode(
    simplify_mode: Optional[str],
    default_simplify_strategy: str,
    category: Optional[str],
    category_strategy_map: Optional[Dict[str, str]],
) -> str:
    """Pick a simplify strategy based on category mapping and fallbacks."""
    if category_strategy_map and category:
        mapped = category_strategy_map.get(category)
        if mapped:
            return mapped

    return simplify_mode or default_simplify_strategy


def _simplify_short_line(
    seg: np.ndarray,
    aspect_ratio_threshold: float,
) -> np.ndarray:
    """Simplify thin line-like polygons into a skinny bounding box.

    If the segment is extremely elongated (major/minor ratio exceeds
    ``aspect_ratio_threshold``), we collapse it to an axis-aligned rectangle with
    just four points. This keeps the overall extent while drastically reducing
    the vertex count for short lines.
    """
    if seg.shape[0] <= 3:
        return seg

    x_min, y_min = seg.min(axis=0)
    x_max, y_max = seg.max(axis=0)
    width = float(x_max - x_min)
    height = float(y_max - y_min)

    major = max(width, height)
    minor = min(width, height)

    # Degenerate: collapse to endpoints along the dominant axis
    if minor == 0:
        axis = 0 if width >= height else 1
        min_idx = int(np.argmin(seg[:, axis]))
        max_idx = int(np.argmax(seg[:, axis]))
        return seg[[min_idx, max_idx]].reshape(-1, 2)

    if major / minor < aspect_ratio_threshold:
        return seg

    return np.asarray(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )


def _simplify_segment(
    seg: np.ndarray,
    mode: str,
    eps_ratio: float,
    short_line_aspect_ratio: float,
) -> np.ndarray:
    """
    seg: (N, 2) float32
    mode:
      - "none"        : 不做簡化
      - "convex_hull" : 取凸包
      - "rdp"         : 用 approxPolyDP 做 RDP 簡化
      - "short_line"  : 將極度狹長的 polygon 簡化為 4 點矩形
    eps_ratio: 相對於「該 segment 外接矩形的較長邊」的比例
    """
    if seg.shape[0] <= 3 or mode == "none":
        return seg

    if mode == "convex_hull":
        hull = cv2.convexHull(seg)
        return hull.reshape(-1, 2)

    if mode == "short_line":
        return _simplify_short_line(seg, aspect_ratio_threshold=short_line_aspect_ratio)

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
    simplify_mode: Optional[str] = DEFAULT_SIMPLIFY_STRATEGY,  # fallback strategy
    simplify_eps_ratio: float = 0.01,  # 只對 "rdp" 有效
    *,
    class_category_map: Optional[Dict[Any, str]] = None,
    category_strategy_map: Optional[Dict[str, str]] = None,
    default_simplify_strategy: str = DEFAULT_SIMPLIFY_STRATEGY,
    short_line_aspect_ratio: float = 8.0,
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
        category = _resolve_class_category(cid, class_name, class_category_map)
        chosen_mode = _resolve_simplify_mode(
            simplify_mode,
            default_simplify_strategy,
            category,
            category_strategy_map,
        )
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

                arr = _simplify_segment(
                    arr,
                    chosen_mode,
                    simplify_eps_ratio,
                    short_line_aspect_ratio,
                )
                polys.append(arr.astype(float).tolist())
        else:
            # 沒有 mask：用 bbox 當成一個矩形 polygon
            polys.append(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ]
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
