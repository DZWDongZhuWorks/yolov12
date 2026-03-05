# app_utils/polygon_utils.py
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

DEFAULT_SIMPLIFY_EPS_RATIO = 0.01


def _triangle_area2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    回傳三角形面積的兩倍（避免不必要的 sqrt），用於 Visvalingam-Whyatt。
    """
    return float(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))


def _visvalingam_whyatt_open(points: np.ndarray, threshold_area2: float) -> np.ndarray:
    """
    對開放折線套用 Visvalingam-Whyatt。
    首尾點固定保留。
    """
    pts = [p.copy() for p in points]
    if len(pts) <= 3 or threshold_area2 <= 0:
        return np.asarray(pts, dtype=np.float32)

    while len(pts) > 3:
        min_idx = -1
        min_area2 = float("inf")
        for i in range(1, len(pts) - 1):
            area2 = _triangle_area2(pts[i - 1], pts[i], pts[i + 1])
            if area2 < min_area2:
                min_area2 = area2
                min_idx = i

        if min_area2 > threshold_area2 or min_idx < 0:
            break

        pts.pop(min_idx)

    return np.asarray(pts, dtype=np.float32)


def _visvalingam_whyatt_closed(seg: np.ndarray, eps_coeff: float) -> np.ndarray:
    """
    對封閉 polygon ring（不含重複結尾點）套用 Visvalingam-Whyatt。
    """
    n = seg.shape[0]
    if n <= 3:
        return seg

    x_min, y_min = seg.min(axis=0)
    x_max, y_max = seg.max(axis=0)
    size = max(x_max - x_min, y_max - y_min)
    threshold_area2 = (float(size) * DEFAULT_SIMPLIFY_EPS_RATIO * eps_coeff) ** 2

    # 把 ring 轉成開放折線（在尾端附加首點）再簡化，最後去掉附加點
    open_pts = np.vstack([seg, seg[0:1]])
    simplified_open = _visvalingam_whyatt_open(open_pts, threshold_area2)

    # 最少保留 3 點（不含結尾重複點）
    if simplified_open.shape[0] <= 4:
        core = simplified_open[:-1]
        if core.shape[0] < 3:
            return seg
        return core

    return simplified_open[:-1]


def _simplify_segment(seg: np.ndarray, mode: str, eps_coeff: float) -> np.ndarray:
    """
    seg: (N, 2) float32
    mode:
      - "none"               : 不做簡化
      - "convex_hull"        : 取凸包
      - "rdp"                : 用 approxPolyDP 做 RDP 簡化
      - "visvalingam_whyatt" : 用 Visvalingam-Whyatt 依面積簡化
    eps_coeff: eps 係數（會乘上預設比例）
    """
    if seg.shape[0] <= 3 or mode == "none":
        return seg

    if mode == "convex_hull":
        hull = cv2.convexHull(seg)
        return hull.reshape(-1, 2)

    if mode == "visvalingam_whyatt":
        return _visvalingam_whyatt_closed(seg, eps_coeff)

    # rdp / approxPolyDP
    x_min, y_min = seg.min(axis=0)
    x_max, y_max = seg.max(axis=0)
    size = max(x_max - x_min, y_max - y_min)
    eps = float(size) * DEFAULT_SIMPLIFY_EPS_RATIO * eps_coeff

    approx = cv2.approxPolyDP(seg, eps, closed=True)
    return approx.reshape(-1, 2)


def _polygon_to_min_rect_and_line(
    seg: np.ndarray,
    min_aspect: float,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if seg.shape[0] < 3:
        return seg, None, None

    rect = cv2.minAreaRect(seg.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return seg, None, None

    long_side = max(w, h)
    short_side = min(w, h)
    if min_aspect > 0:
        aspect = long_side / (short_side + 1e-6)
        if aspect < min_aspect:
            return seg, None, None

    rect_poly = cv2.boxPoints(rect).astype(np.float32)

    if w >= h:
        theta = np.deg2rad(angle)
        length = w
    else:
        theta = np.deg2rad(angle + 90.0)
        length = h

    ux, uy = np.cos(theta), np.sin(theta)
    half_len = 0.5 * length
    dx, dy = ux * half_len, uy * half_len
    p1 = np.array([cx - dx, cy - dy], dtype=np.float32)
    p2 = np.array([cx + dx, cy + dy], dtype=np.float32)
    line_seg = np.vstack([p1, p2]).astype(np.float32)
    line_vec = (p2 - p1).astype(np.float32)
    return rect_poly, line_seg, line_vec


def _apply_polygon_steps(
    seg: np.ndarray,
    steps: Optional[List[Dict[str, Any]]],
    class_id: int,
) -> tuple[np.ndarray, Optional[List[float]]]:
    if seg.shape[0] <= 3 or not steps:
        return seg, None

    out = seg
    line_vector: Optional[List[float]] = None
    for step in steps:
        name = str(step.get("name", "")).strip().lower()
        if name not in {"convex_hull", "rdp", "visvalingam_whyatt", "min_rect", "min_line"}:
            continue
        class_filter = step.get("class_filter")
        if class_filter is not None and class_id not in class_filter:
            continue
        count = max(1, int(step.get("count", 1)))
        eps_coeff = float(step.get("eps_coeff", 1.0))
        for _ in range(count):
            if name == "min_rect":
                out, _, _ = _polygon_to_min_rect_and_line(out, min_aspect=eps_coeff)
            elif name == "min_line":
                _, line_seg, line_vec = _polygon_to_min_rect_and_line(out, min_aspect=eps_coeff)
                if line_seg is not None and line_vec is not None:
                    out = line_seg
                    line_vector = line_vec.astype(float).tolist()
            else:
                out = _simplify_segment(out, name, eps_coeff)
            if out.shape[0] <= 3:
                break
    return out, line_vector


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
    simplify_mode: str = "none",      # "none" / "convex_hull" / "rdp" / "visvalingam_whyatt"
    simplify_eps_coeff: float = 1.0, # 對 "rdp" / "visvalingam_whyatt" 有效
    polygon_opt_steps: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    從單一個 YOLO result 產生標準化的物件資訊（含 polygon）。
    之後畫面繪製 & JSON 輸出都只用這個。
    """
    names = getattr(result, "names", {}) or {}

    resolved_polygon_steps: List[Dict[str, Any]] = []
    if polygon_opt_steps:
        name_to_id = {str(name).lower(): int(idx) for idx, name in (names or {}).items()}
        for step in polygon_opt_steps:
            class_tokens = step.get("classes")
            class_filter = None
            if class_tokens is not None:
                if len(class_tokens) == 0:
                    class_filter = set()
                else:
                    resolved = set()
                    for token in class_tokens:
                        if isinstance(token, (int, np.integer)):
                            resolved.add(int(token))
                            continue
                        token_str = str(token).strip()
                        if not token_str:
                            continue
                        try:
                            resolved.add(int(token_str))
                            continue
                        except Exception:
                            pass
                        matched = name_to_id.get(token_str.lower())
                        if matched is not None:
                            resolved.add(matched)
                    class_filter = resolved or set()
            resolved_polygon_steps.append({**step, "class_filter": class_filter})

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
        line_vectors: List[Optional[List[float]]] = []

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

                arr = _simplify_segment(arr, simplify_mode, simplify_eps_coeff)
                arr, line_vector = _apply_polygon_steps(arr, resolved_polygon_steps, int(cid))
                if arr.shape[0] == 2:
                    polys.append(arr.astype(float).tolist())
                else:
                    polys.append(_close_ring(arr.astype(float).tolist()))
                line_vectors.append(line_vector)
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

        obj = {
            "class_id": int(cid),
            "class_name": class_name,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "polygons": polys,
        }
        if any(v is not None for v in line_vectors):
            obj["line_vectors"] = line_vectors
        objects.append(obj)

    return objects
