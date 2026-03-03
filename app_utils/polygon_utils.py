# app_utils/polygon_utils.py
from typing import Any, Dict, List, Optional, Set
import numpy as np
import cv2

DEFAULT_SIMPLIFY_EPS_RATIO = 0.01
DEFAULT_POLYGON_EPS_COEFF = 1.0


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


def _coerce_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _normalize_class_filter(raw_value) -> Optional[List[object]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple, set)):
        tokens = [t for t in raw_value if t not in (None, "")]
    else:
        text = str(raw_value).strip()
        if not text or text.lower() in {"all", "*", "any"}:
            return None
        tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    if not tokens:
        return None

    normalized: List[object] = []
    for token in tokens:
        if isinstance(token, (int, np.integer)):
            normalized.append(int(token))
            continue
        token_str = str(token).strip()
        if ":" in token_str:
            prefix = token_str.split(":", 1)[0].strip()
            try:
                normalized.append(int(prefix))
                continue
            except Exception:
                pass
        try:
            normalized.append(int(token_str))
        except Exception:
            normalized.append(token_str)
    return normalized or None


def _resolve_class_filter(class_filter: Optional[List[object]], names: Dict[int, str]) -> Optional[Set[int]]:
    if not class_filter:
        return None
    name_to_id = {str(name).lower(): int(idx) for idx, name in (names or {}).items()}
    resolved: Set[int] = set()
    for token in class_filter:
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
    return resolved or None


def _build_polygon_step(name: str, count: int, eps_coeff: float = DEFAULT_POLYGON_EPS_COEFF, classes=None) -> Dict[str, Any]:
    return {
        "name": name,
        "count": max(1, int(count)),
        "eps_coeff": max(0.01, _coerce_float(eps_coeff, DEFAULT_POLYGON_EPS_COEFF)),
        "classes": _normalize_class_filter(classes),
    }


def parse_polygon_steps(steps_input) -> List[Dict[str, Any]]:
    if steps_input is None:
        return []
    if hasattr(steps_input, "empty") and steps_input.empty:
        return []
    if not hasattr(steps_input, "empty") and not steps_input:
        return []

    allowed = {"convex_hull", "rdp", "visvalingam_whyatt"}
    steps: List[Dict[str, Any]] = []
    if isinstance(steps_input, str):
        for item in [p.strip() for p in steps_input.replace("\n", ",").split(",") if p.strip()]:
            if ":" in item:
                name, count_text = item.split(":", 1)
            else:
                name, count_text = item, "1"
            name = str(name).strip().lower()
            if name not in allowed:
                continue
            count = _coerce_int(count_text, 1)
            if count <= 0:
                continue
            steps.append(_build_polygon_step(name=name, count=count))
        return steps

    iterable = steps_input
    if hasattr(steps_input, "values") and hasattr(steps_input, "tolist"):
        try:
            iterable = steps_input.values.tolist()
        except Exception:
            pass

    for row in iterable:
        if not row or len(row) < 1:
            continue
        name = str(row[0]).strip().lower()
        if name not in allowed:
            continue
        count = _coerce_int(row[1] if len(row) > 1 else None, 1)
        if count <= 0:
            continue
        eps_coeff = row[2] if len(row) > 2 else DEFAULT_POLYGON_EPS_COEFF
        classes = row[3] if len(row) > 3 else None
        steps.append(_build_polygon_step(name=name, count=count, eps_coeff=eps_coeff, classes=classes))
    return steps


def _apply_polygon_step_pipeline(
    seg: np.ndarray,
    class_id: int,
    names: Dict[int, str],
    polygon_opt_enabled: bool,
    polygon_opt_steps,
    fallback_simplify_mode: str,
    fallback_simplify_eps_coeff: float,
) -> np.ndarray:
    if not polygon_opt_enabled:
        return _simplify_segment(seg, fallback_simplify_mode, fallback_simplify_eps_coeff)

    steps = parse_polygon_steps(polygon_opt_steps)
    if not steps:
        return _simplify_segment(seg, fallback_simplify_mode, fallback_simplify_eps_coeff)

    resolved_steps: List[Dict[str, Any]] = []
    for step in steps:
        resolved_steps.append({
            **step,
            "class_filter": _resolve_class_filter(step.get("classes"), names),
        })

    out = seg
    for step in resolved_steps:
        class_filter = step.get("class_filter")
        if class_filter is not None and class_id not in class_filter:
            continue
        for _ in range(step["count"]):
            out = _simplify_segment(out, step["name"], step["eps_coeff"])
    return out


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
    polygon_opt_enabled: bool = False,
    polygon_opt_steps=None,
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

                arr = _apply_polygon_step_pipeline(
                    arr,
                    class_id=int(cid),
                    names=names,
                    polygon_opt_enabled=polygon_opt_enabled,
                    polygon_opt_steps=polygon_opt_steps,
                    fallback_simplify_mode=simplify_mode,
                    fallback_simplify_eps_coeff=simplify_eps_coeff,
                )
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
