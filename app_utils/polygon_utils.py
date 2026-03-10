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


def _polygon_to_min_area_rect(seg: np.ndarray, min_aspect: float = 0.0) -> np.ndarray:
    if seg.shape[0] < 3:
        return seg

    rect = cv2.minAreaRect(seg.astype(np.float32))
    (w, h) = rect[1]
    if w <= 0 or h <= 0:
        return seg

    long_side = max(w, h)
    short_side = min(w, h)
    if min_aspect > 0:
        aspect = float(long_side / (short_side + 1e-6))
        if aspect < min_aspect:
            return seg

    box = cv2.boxPoints(rect)
    return box.reshape(-1, 2)


def _polygon_to_long_axis_line(seg: np.ndarray, min_aspect: float = 0.0) -> np.ndarray:
    if seg.shape[0] < 3:
        return seg

    rect = cv2.minAreaRect(seg.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return seg

    long_side = max(w, h)
    short_side = min(w, h)
    if min_aspect > 0:
        aspect = float(long_side / (short_side + 1e-6))
        if aspect < min_aspect:
            return seg

    if w >= h:
        theta = np.deg2rad(angle)
        length = w
    else:
        theta = np.deg2rad(angle + 90.0)
        length = h

    ux, uy = np.cos(theta), np.sin(theta)
    half = 0.5 * length
    p1 = np.array([cx - ux * half, cy - uy * half], dtype=np.float32)
    p2 = np.array([cx + ux * half, cy + uy * half], dtype=np.float32)
    return np.vstack([p1, p2])


def _point_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    line = line_end - line_start
    denom = float(np.dot(line, line))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - line_start))
    t = float(np.dot(point - line_start, line) / denom)
    proj = line_start + t * line
    return float(np.linalg.norm(point - proj))


def _detect_turn_point_indices(points: np.ndarray, angle_threshold_deg: float = 25.0) -> set:
    if points.ndim != 2 or points.shape[0] < 3:
        return set()

    protected = set()
    threshold = np.deg2rad(angle_threshold_deg)
    for i in range(1, points.shape[0] - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cos_theta = float(np.dot(v1 / n1, v2 / n2))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = float(np.arccos(cos_theta))
        if angle >= threshold:
            protected.add(i)
    return protected


def _simplify_open_polyline_with_protection(points: np.ndarray, epsilon: float, protected_indices: set) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] <= 2 or epsilon <= 0:
        return points

    n = points.shape[0]
    must_keep = {idx for idx in protected_indices if 0 <= idx < n}
    must_keep.add(0)
    must_keep.add(n - 1)

    def _rdp(i: int, j: int) -> List[int]:
        if j <= i + 1:
            return [i, j]

        max_dist = -1.0
        max_idx = -1
        for k in range(i + 1, j):
            dist = _point_line_distance(points[k], points[i], points[j])
            if dist > max_dist:
                max_dist = dist
                max_idx = k

        inner_protected = [k for k in sorted(must_keep) if i < k < j]
        if max_idx > 0 and max_dist > epsilon:
            left = _rdp(i, max_idx)
            right = _rdp(max_idx, j)
            return left[:-1] + right

        if inner_protected:
            out = [i]
            prev = i
            for p in inner_protected:
                part = _rdp(prev, p)
                out.extend(part[1:])
                prev = p
            tail = _rdp(prev, j)
            out.extend(tail[1:])
            return out

        return [i, j]

    kept = _rdp(0, n - 1)
    unique_kept = sorted(set(kept))
    return points[unique_kept]


def _polygon_to_lane_line(seg: np.ndarray, eps_coeff: float = 1.0, min_aspect: float = 0.0) -> np.ndarray:
    if seg.shape[0] < 4:
        return seg

    ring = seg.astype(np.float32)
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 4:
        return _polygon_to_long_axis_line(seg, min_aspect=min_aspect)

    rect = cv2.minAreaRect(ring)
    (_, _), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return seg

    long_side = max(w, h)
    short_side = min(w, h)
    aspect = float(long_side / (short_side + 1e-6))
    if min_aspect > 0 and aspect < min_aspect:
        return seg

    if w >= h:
        theta = np.deg2rad(angle)
    else:
        theta = np.deg2rad(angle + 90.0)

    axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    center = ring.mean(axis=0)

    rel = ring - center
    t = rel @ axis
    s = rel @ normal
    t_min, t_max = float(t.min()), float(t.max())
    t_span = max(1e-6, t_max - t_min)

    simplify_strength = max(0.1, float(eps_coeff))
    keep_ratio = max(0.15, min(0.9, 1.0 / (1.0 + 0.6 * simplify_strength)))
    target_points = int(max(4, min(ring.shape[0], round(ring.shape[0] * keep_ratio))))

    bins = np.linspace(t_min, t_max, num=target_points)
    lane_points: List[np.ndarray] = []
    for idx in range(len(bins) - 1):
        left, right = bins[idx], bins[idx + 1]
        if idx == len(bins) - 2:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        if not np.any(mask):
            continue
        t_center = float(t[mask].mean())
        s_center = float(np.median(s[mask]))
        lane_points.append(center + axis * t_center + normal * s_center)

    if len(lane_points) < 2:
        return _polygon_to_long_axis_line(seg, min_aspect=min_aspect)

    line = np.asarray(lane_points, dtype=np.float32)
    order = np.argsort(line @ axis)
    line = line[order]

    dedup = [line[0]]
    for p in line[1:]:
        if float(np.linalg.norm(p - dedup[-1])) > 1e-3:
            dedup.append(p)
    line = np.asarray(dedup, dtype=np.float32)
    if line.shape[0] < 2:
        return _polygon_to_long_axis_line(seg, min_aspect=min_aspect)

    protected = _detect_turn_point_indices(line, angle_threshold_deg=25.0)
    epsilon = t_span * DEFAULT_SIMPLIFY_EPS_RATIO * max(0.1, float(eps_coeff))
    return _simplify_open_polyline_with_protection(line, epsilon=epsilon, protected_indices=protected)


def _apply_polygon_steps(seg: np.ndarray, steps: Optional[List[Dict[str, Any]]], class_id: int) -> np.ndarray:
    if seg.shape[0] <= 3 or not steps:
        return seg

    out = seg
    for step in steps:
        name = str(step.get("name", "")).strip().lower()
        if name not in {"convex_hull", "rdp", "visvalingam_whyatt", "min_area_rect", "export_line", "pca", "trans_lane_line_polygon"}:
            continue
        class_filter = step.get("class_filter")
        if class_filter is not None and class_id not in class_filter:
            continue
        count = max(1, int(step.get("count", 1)))
        eps_coeff = float(step.get("eps_coeff", 1.0))
        min_aspect = max(0.0, float(step.get("min_aspect", 0.0)))
        for _ in range(count):
            if name == "min_area_rect":
                out = _polygon_to_min_area_rect(out, min_aspect=min_aspect)
            elif name == "export_line":
                out = _polygon_to_long_axis_line(out, min_aspect=min_aspect)
            elif name == "trans_lane_line_polygon":
                out = _polygon_to_lane_line(out, eps_coeff=eps_coeff, min_aspect=min_aspect)
            elif name == "pca":
                # pca 為 class-wise 後處理（需跨物件統計方向），此處先略過
                continue
            else:
                out = _simplify_segment(out, name, eps_coeff)
            if out.shape[0] <= 3:
                break
    return out



def _line_endpoints(seg: np.ndarray):
    if seg.ndim != 2 or seg.shape[0] < 2 or seg.shape[1] < 2:
        return None
    if seg.shape[0] == 2:
        return seg.astype(np.float32)
    # 封閉 ring（首尾相同）則拿前兩點代表線段會有偏差，因此只對 2 點線段做 PCA 對齊
    return None


def _line_unit_direction(line: np.ndarray) -> Optional[np.ndarray]:
    if line.ndim != 2 or line.shape[0] != 2:
        return None
    v = (line[1] - line[0]).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n <= 1e-6:
        return None
    return (v / n).astype(np.float32)


def _dominant_axis_from_units(units: List[np.ndarray]) -> Optional[np.ndarray]:
    if not units:
        return None

    cov = np.zeros((2, 2), dtype=np.float64)
    for u in units:
        cov += np.outer(u, u)

    if float(cov.sum()) <= 1e-9:
        return None

    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    n = float(np.linalg.norm(axis))
    if n <= 1e-9:
        return None
    return (axis / n).astype(np.float32)


def _cluster_line_orientations(units: List[np.ndarray], min_cosine: float = 0.94) -> List[Dict[str, Any]]:
    if not units:
        return []

    clusters: List[Dict[str, Any]] = []

    for idx, u in enumerate(units):
        best_i = -1
        best_score = -1.0
        for ci, cluster in enumerate(clusters):
            axis = cluster["axis"]
            score = float(abs(np.dot(u, axis)))
            if score > best_score:
                best_score = score
                best_i = ci

        if best_i >= 0 and best_score >= min_cosine:
            clusters[best_i]["members"].append(idx)
            member_units = [units[i] for i in clusters[best_i]["members"]]
            axis = _dominant_axis_from_units(member_units)
            if axis is not None:
                clusters[best_i]["axis"] = axis
        else:
            clusters.append({"members": [idx], "axis": u.copy()})

    # 再做一次指派，讓 greedy 初始分群更穩定
    if len(clusters) <= 1:
        return clusters

    refined = [{"members": [], "axis": c["axis"].copy()} for c in clusters]
    for idx, u in enumerate(units):
        scores = [float(abs(np.dot(u, c["axis"]))) for c in refined]
        best_i = int(np.argmax(scores)) if scores else -1
        if best_i >= 0 and scores[best_i] >= min_cosine:
            refined[best_i]["members"].append(idx)
        else:
            refined.append({"members": [idx], "axis": u.copy()})

    output: List[Dict[str, Any]] = []
    for cluster in refined:
        if not cluster["members"]:
            continue
        member_units = [units[i] for i in cluster["members"]]
        axis = _dominant_axis_from_units(member_units)
        if axis is None:
            axis = cluster["axis"]
        output.append({"members": cluster["members"], "axis": axis})
    return output


def _apply_pca_alignment(objects: List[Dict[str, Any]], steps: Optional[List[Dict[str, Any]]]):
    if not objects or not steps:
        return

    pca_steps = [st for st in steps if str(st.get("name", "")).strip().lower() == "pca"]
    if not pca_steps:
        return

    for step in pca_steps:
        class_filter = step.get("class_filter")
        count = max(1, int(step.get("count", 1)))
        alpha = float(step.get("eps_coeff", 1.0))
        alpha = max(0.0, min(1.0, alpha))

        for _ in range(count):
            cross_class = bool(step.get("pca_cross_class", False))
            grouped_entries: Dict[str, List[Dict[str, Any]]] = {}
            for obj_idx, obj in enumerate(objects):
                cid = int(obj.get("class_id", -1))
                if class_filter is not None and cid not in class_filter:
                    continue

                for poly_idx, poly in enumerate(obj.get("polygons", [])):
                    seg = np.asarray(poly, dtype=np.float32)
                    line = _line_endpoints(seg)
                    if line is None:
                        continue
                    unit = _line_unit_direction(line)
                    if unit is None:
                        continue
                    key = "__cross_class__" if cross_class else str(cid)
                    grouped_entries.setdefault(key, []).append(
                        {
                            "obj_idx": obj_idx,
                            "poly_idx": poly_idx,
                            "line": line,
                            "unit": unit,
                        }
                    )

            for _, entries in grouped_entries.items():
                if not entries:
                    continue

                units = [e["unit"] for e in entries]
                pca_min_cosine = float(step.get("pca_min_cosine", 0.94))
                pca_min_cosine = max(0.0, min(1.0, pca_min_cosine))
                clusters = _cluster_line_orientations(units, min_cosine=pca_min_cosine)

                for cluster in clusters:
                    axis = cluster.get("axis")
                    if axis is None:
                        continue

                    for member_idx in cluster["members"]:
                        entry = entries[member_idx]
                        line = entry["line"]
                        p1, p2 = line[0], line[1]
                        center = 0.5 * (p1 + p2)
                        v = p2 - p1
                        length = float(np.linalg.norm(v))
                        if length <= 1e-6:
                            continue

                        u = v / length
                        target = axis if float(np.dot(u, axis)) >= 0 else -axis
                        blended = (1.0 - alpha) * u + alpha * target
                        bn = float(np.linalg.norm(blended))
                        if bn <= 1e-6:
                            continue

                        d = (blended / bn) * (0.5 * length)
                        n1 = (center - d).astype(float).tolist()
                        n2 = (center + d).astype(float).tolist()
                        objects[entry["obj_idx"]]["polygons"][entry["poly_idx"]] = [n1, n2]


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


def _step_exports_open_polyline(steps: Optional[List[Dict[str, Any]]], class_id: int) -> bool:
    if not steps:
        return False
    for step in steps:
        name = str(step.get("name", "")).strip().lower()
        if name not in {"export_line", "trans_lane_line_polygon"}:
            continue
        class_filter = step.get("class_filter")
        if class_filter is None or class_id in class_filter:
            return True
    return False


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
                arr = _apply_polygon_steps(arr, resolved_polygon_steps, int(cid))
                arr_list = arr.astype(float).tolist()
                if arr.shape[0] > 2 and not _step_exports_open_polyline(resolved_polygon_steps, int(cid)):
                    arr_list = _close_ring(arr_list)
                polys.append(arr_list)
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

    _apply_pca_alignment(objects, resolved_polygon_steps)

    return objects
