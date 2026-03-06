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


def _polygon_to_lane_centerline(seg: np.ndarray, min_aspect: float = 0.0, eps_coeff: float = 1.0) -> np.ndarray:
    """
    將長條狀 polygon 轉成多點中心線（LineString）。
    方法：以最長軸方向切片，取每個切片的上下邊界中點。
    """
    if seg.shape[0] < 3:
        return seg

    rect = cv2.minAreaRect(seg.astype(np.float32))
    (cx, cy), (w, h), angle = rect
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

    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)  # 長軸
    v = np.array([-u[1], u[0]], dtype=np.float32)  # 法向量
    c = np.array([cx, cy], dtype=np.float32)

    rel = seg.astype(np.float32) - c
    ts = rel @ u
    ns = rel @ v

    # 長條狀物件切片越密，中心線越平順
    sample_count = int(np.clip(max(8.0, aspect * 4.0), 8, 256))
    t_samples = np.linspace(float(ts.min()), float(ts.max()), sample_count, dtype=np.float32)

    # 封閉邊集合（最後一點連回第一點）
    poly_t = np.concatenate([ts, ts[:1]])
    poly_n = np.concatenate([ns, ns[:1]])

    centers_t: List[float] = []
    centers_n: List[float] = []
    for t0 in t_samples:
        intersections: List[float] = []
        for i in range(seg.shape[0]):
            t1, n1 = float(poly_t[i]), float(poly_n[i])
            t2, n2 = float(poly_t[i + 1]), float(poly_n[i + 1])

            dt = t2 - t1
            if abs(dt) <= 1e-6:
                if abs(t0 - t1) <= 1e-6:
                    intersections.extend([n1, n2])
                continue

            t_min, t_max = (t1, t2) if t1 <= t2 else (t2, t1)
            if t0 < t_min or t0 > t_max:
                continue

            ratio = (float(t0) - t1) / dt
            intersections.append(n1 + ratio * (n2 - n1))

        if len(intersections) < 2:
            continue

        intersections = sorted(intersections)
        centers_t.append(float(t0))
        centers_n.append(0.5 * (intersections[0] + intersections[-1]))

    if len(centers_t) < 2:
        return _polygon_to_long_axis_line(seg, min_aspect=min_aspect)

    line_local = np.stack([np.asarray(centers_t, dtype=np.float32), np.asarray(centers_n, dtype=np.float32)], axis=1)
    # 開放折線簡化（eps_coeff 越大，點越少）
    threshold_area2 = (float(long_side) * DEFAULT_SIMPLIFY_EPS_RATIO * max(0.1, eps_coeff)) ** 2
    line_local = _visvalingam_whyatt_open(line_local, threshold_area2)
    if line_local.shape[0] < 2:
        return _polygon_to_long_axis_line(seg, min_aspect=min_aspect)

    line_xy = c + np.outer(line_local[:, 0], u) + np.outer(line_local[:, 1], v)
    return line_xy.astype(np.float32)


def _apply_polygon_steps(seg: np.ndarray, steps: Optional[List[Dict[str, Any]]], class_id: int):
    if seg.shape[0] <= 3 or not steps:
        return seg, False

    out = seg
    is_line = False
    for step in steps:
        name = str(step.get("name", "")).strip().lower()
        if name not in {"convex_hull", "rdp", "visvalingam_whyatt", "min_area_rect", "export_line", "lane_centerline", "pca"}:
            continue
        if is_line and name != "pca":
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
                is_line = True
            elif name == "lane_centerline":
                out = _polygon_to_lane_centerline(out, min_aspect=min_aspect, eps_coeff=eps_coeff)
                is_line = True
            elif name == "pca":
                # pca 為 class-wise 後處理（需跨物件統計方向），此處先略過
                continue
            else:
                out = _simplify_segment(out, name, eps_coeff)
            if out.shape[0] <= 2:
                break
    return out, is_line



def _line_endpoints(seg: np.ndarray):
    if seg.ndim != 2 or seg.shape[0] < 2 or seg.shape[1] < 2:
        return None
    if seg.shape[0] == 2:
        return seg.astype(np.float32)
    # 開放折線（多點）可用首尾點代表方向；封閉 ring 則忽略
    first, last = seg[0], seg[-1]
    if float(np.linalg.norm(first - last)) <= 1e-6:
        return None
    return np.vstack([first, last]).astype(np.float32)


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
                arr, is_line = _apply_polygon_steps(arr, resolved_polygon_steps, int(cid))
                if is_line:
                    polys.append(arr.astype(float).tolist())
                else:
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

    _apply_pca_alignment(objects, resolved_polygon_steps)

    return objects
