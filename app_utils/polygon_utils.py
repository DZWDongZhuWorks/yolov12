# app_utils/polygon_utils.py
from typing import Any, Dict, Iterable, List, Optional, Set
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


def _normalize_options(options: Optional[Iterable[str]]) -> Set[str]:
    return {str(opt) for opt in (options or []) if opt}


def _remove_neighbor_duplicates(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    cleaned = [points[0]]
    for pt in points[1:]:
        if not np.allclose(pt, cleaned[-1]):
            cleaned.append(pt)
    return np.asarray(cleaned, dtype=points.dtype)


def _remove_collinear(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if points.shape[0] < 3:
        return points
    pts = points
    cleaned = []
    n = pts.shape[0]
    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % n]
        v1 = curr_pt - prev_pt
        v2 = next_pt - curr_pt
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) <= eps:
            continue
        cleaned.append(curr_pt)
    if not cleaned:
        return points
    return np.asarray(cleaned, dtype=points.dtype)


def _snap_to_grid(points: np.ndarray, grid_size: float) -> np.ndarray:
    if grid_size <= 0:
        return points
    return np.round(points / grid_size) * grid_size


def _apply_mask_filters(binary: np.ndarray, options: Set[str]) -> np.ndarray:
    mask = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if "mask_open_close" in options:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if "mask_blur" in options:
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
        mask = (blurred > 0.5).astype(np.uint8)

    if "mask_remove_small" in options:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= 30:
                cv2.drawContours(cleaned, [contour], -1, 1, thickness=-1)
        mask = cleaned

    if "mask_fill_holes" in options:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for idx, contour in enumerate(contours):
                parent = hierarchy[0][idx][3]
                if parent != -1 and cv2.contourArea(contour) < 30:
                    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)

    return mask


def _extract_contours_from_mask(
    binary: np.ndarray,
    options: Set[str],
    eps_ratio: float,
) -> List[np.ndarray]:
    retrieval = cv2.RETR_EXTERNAL if "contour_external" in options else cv2.RETR_CCOMP
    chain = cv2.CHAIN_APPROX_SIMPLE if "contour_simple" in options else cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(binary, retrieval, chain)
    if not contours:
        return []

    if "contour_rdp" in options:
        simplified = []
        for contour in contours:
            arc = cv2.arcLength(contour, True)
            eps = float(arc) * eps_ratio
            approx = cv2.approxPolyDP(contour, eps, True)
            simplified.append(approx)
        contours = simplified

    return [contour.reshape(-1, 2).astype(np.float32) for contour in contours if contour.shape[0] >= 3]


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
    optimize_options: Optional[List[str]] = None,
    snap_grid_size: float = 0.0,
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
    mask_data = getattr(result.masks, "data", None) if has_masks else None
    mask_count = int(len(mask_data)) if mask_data is not None else 0
    options = _normalize_options(optimize_options)
    use_contour_extract = bool(
        options
        & {
            "mask_open_close",
            "mask_blur",
            "mask_remove_small",
            "mask_fill_holes",
            "contour_simple",
            "contour_rdp",
            "contour_external",
        }
    )

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

        if has_masks and i < mask_count:
            if use_contour_extract:
                mask_tensor = mask_data[i]
                mask_np = mask_tensor.detach().cpu().numpy()
                binary = (mask_np > 0.5).astype(np.uint8)
                binary = _apply_mask_filters(binary, options)
                segments = _extract_contours_from_mask(binary, options, simplify_eps_ratio)
            else:
                if raw_polys is None or i >= len(raw_polys):
                    segments = []
                else:
                    item = raw_polys[i]
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
                if "polygon_collinear" in options:
                    arr = _remove_collinear(arr)
                if "polygon_snap" in options:
                    arr = _snap_to_grid(arr, snap_grid_size)
                arr = _remove_neighbor_duplicates(arr)
                if arr.shape[0] >= 3:
                    polys.append(_close_ring(arr.astype(float).tolist()))
        if not polys:
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
