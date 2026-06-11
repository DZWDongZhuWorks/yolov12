# app_utils/polygon_utils.py
import time
from typing import Any, Dict, List, Optional, Tuple
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
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr

    is_closed = arr.shape[0] >= 4 and np.allclose(arr[0], arr[-1])
    core = arr[:-1] if is_closed else arr
    if not is_closed or core.shape[0] < 3:
        return arr

    rect = cv2.minAreaRect(core.astype(np.float32))
    (w, h) = rect[1]
    if w <= 0 or h <= 0:
        return arr

    long_side = max(w, h)
    short_side = min(w, h)
    if min_aspect > 0:
        aspect = float(long_side / (short_side + 1e-6))
        if aspect < min_aspect:
            return arr

    box = cv2.boxPoints(rect)
    box = box.reshape(-1, 2)
    return np.vstack([box, box[0:1]])


def _polygon_to_long_axis_line(seg: np.ndarray, min_aspect: float = 0.0) -> np.ndarray:
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr

    is_closed = arr.shape[0] >= 4 and np.allclose(arr[0], arr[-1])
    core = arr[:-1] if is_closed else arr
    if not is_closed or core.shape[0] < 3:
        return arr

    rect = cv2.minAreaRect(core.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return arr

    long_side = max(w, h)
    short_side = min(w, h)
    if min_aspect > 0:
        aspect = float(long_side / (short_side + 1e-6))
        if aspect < min_aspect:
            return arr

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


def _points_to_lane_line(points: np.ndarray, eps_coeff: float = 1.0) -> Optional[np.ndarray]:
    """
    將一組 2D 邊界點擬合成 lane line（多點 LineString）。
    作法：
      1) PCA 求主軸 u 與法向 v。
      2) 將點投影到 (t, s) 座標（t 沿主軸、s 沿法向）。
      3) 沿 t 分箱，對每箱以 s 的 min/max 中點估計中心線點。
      4) 依 eps_coeff 對開放折線做輕量簡化。
    """
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None

    pts = points[:, :2].astype(np.float32)
    center = pts.mean(axis=0)
    centered = pts - center

    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return None
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))].astype(np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return None
    u = axis / axis_norm
    v = np.array([-u[1], u[0]], dtype=np.float32)

    t = centered @ u
    s = centered @ v
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    t_range = t_max - t_min
    s_range = s_max - s_min
    if t_range <= 1e-6:
        return None

    aspect = t_range / max(s_range, 1e-3)
    n_bins = int(np.clip(round(aspect * 8), 8, 128))
    if n_bins < 2:
        n_bins = 2

    edges = np.linspace(t_min, t_max, n_bins + 1)
    centers = []

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        if not np.any(mask):
            continue

        t_bin = t[mask]
        s_bin = s[mask]
        t_mid = float(np.mean(t_bin))
        s_mid = 0.5 * (float(np.min(s_bin)) + float(np.max(s_bin)))
        p = center + t_mid * u + s_mid * v
        centers.append([float(t_mid), float(p[0]), float(p[1])])

    if len(centers) < 2:
        p1 = (center + t_min * u).astype(np.float32)
        p2 = (center + t_max * u).astype(np.float32)
        return np.vstack([p1, p2])

    centers.sort(key=lambda x: x[0])
    line = np.asarray([[c[1], c[2]] for c in centers], dtype=np.float32)

    # 開放折線簡化：eps_coeff 越大，簡化越強
    threshold_area2 = (max(t_range, s_range) * DEFAULT_SIMPLIFY_EPS_RATIO * max(0.1, float(eps_coeff))) ** 2
    line = _visvalingam_whyatt_open(line, threshold_area2)

    if line.shape[0] < 2:
        p1 = (center + t_min * u).astype(np.float32)
        p2 = (center + t_max * u).astype(np.float32)
        return np.vstack([p1, p2])

    return line


def _polyline_length(line: np.ndarray) -> float:
    if line.ndim != 2 or line.shape[0] < 2:
        return 0.0
    diffs = np.diff(line[:, :2], axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _resample_polyline(line: np.ndarray, n_samples: int) -> Optional[np.ndarray]:
    if line.ndim != 2 or line.shape[0] < 2 or n_samples < 2:
        return None

    pts = line[:, :2].astype(np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-6:
        return None

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, int(n_samples))
    out = []

    j = 0
    for t in targets:
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        d = seg[j]
        if d <= 1e-6:
            out.append(pts[j].copy())
            continue
        ratio = float((t - cum[j]) / d)
        ratio = max(0.0, min(1.0, ratio))
        p = pts[j] * (1.0 - ratio) + pts[j + 1] * ratio
        out.append(p)

    return np.asarray(out, dtype=np.float32)


def _extract_centerline_delaunay(arr: np.ndarray, eps_coeff: float = 1.0) -> Optional[np.ndarray]:
    """
    使用 Delaunay 三角網格式的 Medial Axis 演算法，從封閉多邊形中萃取中心線。
    能自動適應狹長、U型、甚至 O型 內凹幾何形狀。
    """
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None

    arr_closed = arr if np.allclose(arr[0], arr[-1]) else np.vstack([arr, arr[0]])
    diffs = np.diff(arr_closed, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    poly_len = float(np.sum(seg_lens))
    
    if poly_len <= 1e-6:
        return None
        
    step = 15.0 # pixels
    n_samples = int(np.clip(poly_len / step, 10, 500))
    resampled = _resample_polyline(arr_closed, n_samples)
    if resampled is None or resampled.shape[0] < 3:
        return None
    resampled = resampled[:-1]
    
    x_min, y_min = resampled.min(axis=0)
    x_max, y_max = resampled.max(axis=0)
    
    pad = 20.0
    w = int(x_max - x_min + 2 * pad)
    h = int(y_max - y_min + 2 * pad)
    subdiv = cv2.Subdiv2D((int(x_min - pad), int(y_min - pad), w, h))
    
    pts_inserted = []
    for p in resampled:
        if not pts_inserted or float(np.linalg.norm(p - pts_inserted[-1])) > 0.1:
            subdiv.insert((float(p[0]), float(p[1])))
            pts_inserted.append(p)
            
    if len(pts_inserted) < 3:
        return None
        
    try:
        triangles = subdiv.getTriangleList()
    except Exception:
        return None
        
    arr_float32 = arr_closed.astype(np.float32)
    
    inside_tris = []
    for t in triangles:
        p1 = np.array([t[0], t[1]], dtype=np.float32)
        p2 = np.array([t[2], t[3]], dtype=np.float32)
        p3 = np.array([t[4], t[5]], dtype=np.float32)
        centroid = (p1 + p2 + p3) / 3.0
        
        if cv2.pointPolygonTest(arr_float32, (float(centroid[0]), float(centroid[1])), True) >= -3.0:
            inside_tris.append((p1, p2, p3, centroid))
            
    if not inside_tris:
        return None
        
    pts_inserted_np = np.asarray(pts_inserted, dtype=np.float32)
    
    def get_vertex_indices(tri_pts):
        diff = pts_inserted_np[np.newaxis, :, :] - tri_pts[:, np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        return tuple(sorted(np.argmin(dist_sq, axis=1).tolist()))
        
    edge_to_tri_idx = {}
    for i, t in enumerate(inside_tris):
        pts = np.array([t[0], t[1], t[2]])
        vid = get_vertex_indices(pts)
        if len(set(vid)) < 3:
            continue
        edges = [(vid[0], vid[1]), (vid[1], vid[2]), (vid[0], vid[2])]
        for e in edges:
            edge_to_tri_idx.setdefault(e, []).append(i)
            
    n_nodes = len(inside_tris)
    adj = {i: [] for i in range(n_nodes)}
    for e, t_indices in edge_to_tri_idx.items():
        if len(t_indices) == 2:
            u, v = t_indices
            adj[u].append(v)
            adj[v].append(u)
            
    visited_dfs = set()
    parent_dfs = {}
    cycle_path = []
    
    def dfs_cycle(start):
        stack = [(start, None)]
        while stack:
            curr, par = stack.pop()
            if curr in visited_dfs:
                path = [curr]
                c = par
                while c is not None and c != curr:
                    path.append(c)
                    c = parent_dfs.get(c)
                if c == curr:
                    path.append(curr)
                    return path
                continue
            visited_dfs.add(curr)
            parent_dfs[curr] = par
            for neighbor in adj[curr]:
                if neighbor != par:
                    stack.append((neighbor, curr))
        return []

    for i in range(n_nodes):
        if i not in visited_dfs:
            cycle = dfs_cycle(i)
            if len(cycle) > len(cycle_path):
                cycle_path = cycle

    main_path_indices = []
    if len(cycle_path) > max(n_nodes * 0.3, 5):
        main_path_indices = cycle_path
    else:
        def bfs_farthest(start_node):
            q = [(start_node, [start_node])]
            vis = {start_node}
            max_path = [start_node]
            while q:
                curr, path = q.pop(0)
                if len(path) > len(max_path):
                    max_path = path
                for neighbor in adj[curr]:
                    if neighbor not in vis:
                        vis.add(neighbor)
                        q.append((neighbor, path + [neighbor]))
            return max_path
            
        for i in range(n_nodes):
            path = bfs_farthest(i)
            if len(path) > len(main_path_indices):
                main_path_indices = path

    if len(main_path_indices) < 2:
        return None
        
    centerline = np.asarray([inside_tris[i][3] for i in main_path_indices], dtype=np.float32)
    
    x_min_l, y_min_l = centerline.min(axis=0)
    x_max_l, y_max_l = centerline.max(axis=0)
    size = max(float(x_max_l - x_min_l), float(y_max_l - y_min_l))
    threshold_area2 = (size * DEFAULT_SIMPLIFY_EPS_RATIO * max(0.1, float(eps_coeff))) ** 2
    smoothed_line = _visvalingam_whyatt_open(centerline, threshold_area2)
    
    return smoothed_line if smoothed_line.shape[0] >= 2 else centerline


def _ring_to_lane_line(ring: np.ndarray, eps_coeff: float = 1.0) -> Optional[np.ndarray]:
    """
    將單一封閉 ring 轉成可跟隨急彎的多點中心線。
    核心：使用 Delaunay 三角網格之內接中軸 (Medial Axis) 演算法。
    """
    return _extract_centerline_delaunay(ring, eps_coeff)


def _nearest_point_on_ring(p: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """點到閉合折線的最近點（逐線段投影）。"""
    best = ring[0]
    best_d = float("inf")
    for i in range(ring.shape[0] - 1):
        a = ring[i]
        b = ring[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom <= 1e-12 else max(0.0, min(1.0, float((p - a) @ ab) / denom))
        q = a + t * ab
        d = float(np.sum((p - q) ** 2))
        if d < best_d:
            best_d = d
            best = q
    return best


def _annulus_centerline(outer: np.ndarray, hole: np.ndarray, eps_coeff: float = 1.0) -> Optional[np.ndarray]:
    """甜甜圈（外環 + 單一洞）-> 閉合環形中心線。

    作法：對內環等距取樣，每個樣本點找外環最近點，取中點串成閉合折線。
    比 Delaunay 中軸法穩定：無三角化、無圖搜尋。eps_coeff 越大取樣越疏（線越簡化）。
    """
    if outer.ndim != 2 or outer.shape[0] < 4 or hole.ndim != 2 or hole.shape[0] < 4:
        return None

    def _close(r: np.ndarray) -> np.ndarray:
        return r if np.allclose(r[0], r[-1]) else np.vstack([r, r[0]])

    outer_closed = _close(outer[:, :2].astype(np.float32))
    hole_closed = _close(hole[:, :2].astype(np.float32))

    hole_len = float(np.linalg.norm(np.diff(hole_closed, axis=0), axis=1).sum())
    if hole_len <= 1e-6:
        return None

    step = 15.0 * max(0.1, float(eps_coeff))
    n_samples = int(np.clip(hole_len / step, 12, 500))
    samples = _resample_polyline(hole_closed, n_samples)
    if samples is None or samples.shape[0] < 4:
        return None
    samples = samples[:-1]

    mids = np.asarray(
        [(p + _nearest_point_on_ring(p, outer_closed)) / 2.0 for p in samples],
        dtype=np.float32,
    )
    return np.vstack([mids, mids[:1]])  # 閉合環形中心線（首尾相同）


def _is_closed_ring(seg: np.ndarray) -> bool:
    return bool(seg.ndim == 2 and seg.shape[0] >= 4 and np.allclose(seg[0], seg[-1]))


def _strip_closing_point(seg: np.ndarray) -> np.ndarray:
    return seg[:-1] if _is_closed_ring(seg) else seg


def _restore_geometry(seg: np.ndarray, was_closed: bool) -> np.ndarray:
    arr = np.asarray(seg, dtype=np.float32)
    if was_closed and arr.ndim == 2 and arr.shape[0] >= 3 and not np.allclose(arr[0], arr[-1]):
        return np.vstack([arr, arr[0:1]]).astype(np.float32)
    return arr.astype(np.float32)


def _geometry_scale(seg: np.ndarray) -> float:
    arr = _strip_closing_point(np.asarray(seg, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 0.0
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    return float(max(x_max - x_min, y_max - y_min))


def _simplify_open_line(seg: np.ndarray, mode: str, eps_coeff: float) -> np.ndarray:
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or mode == "none":
        return arr
    if mode == "convex_hull":
        return arr

    size = _geometry_scale(arr)
    if size <= 1e-6:
        return arr

    if mode == "visvalingam_whyatt":
        threshold_area2 = (size * DEFAULT_SIMPLIFY_EPS_RATIO * max(0.1, float(eps_coeff))) ** 2
        return _visvalingam_whyatt_open(arr, threshold_area2)

    eps = size * DEFAULT_SIMPLIFY_EPS_RATIO * float(eps_coeff)
    approx = cv2.approxPolyDP(arr, eps, closed=False)
    return approx.reshape(-1, 2).astype(np.float32)


def _simplify_geometry(seg: np.ndarray, mode: str, eps_coeff: float) -> np.ndarray:
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or mode == "none":
        return arr

    was_closed = _is_closed_ring(arr)
    core = _strip_closing_point(arr)
    if not was_closed:
        return _simplify_open_line(core, mode, eps_coeff)

    if core.shape[0] <= 3:
        return _restore_geometry(core, True)

    if mode == "convex_hull":
        hull = cv2.convexHull(core.astype(np.float32))
        return _restore_geometry(hull.reshape(-1, 2), True)

    if mode == "visvalingam_whyatt":
        return _restore_geometry(_visvalingam_whyatt_closed(core, eps_coeff), True)

    size = _geometry_scale(core)
    if size <= 1e-6:
        return _restore_geometry(core, True)

    eps = size * DEFAULT_SIMPLIFY_EPS_RATIO * float(eps_coeff)
    approx = cv2.approxPolyDP(core.astype(np.float32), eps, closed=True)
    return _restore_geometry(approx.reshape(-1, 2), True)


def _polygon_fit_small_object(
    seg: np.ndarray,
    max_area_px: float,
    target_vertices: int,
    min_vertices: int = 3,
) -> np.ndarray:
    """
    為小物件（菱形、倒三角形、箭頭、道路標字等）設計的形狀貼合簡化。

    流程：
      - 若 bbox 面積 > max_area_px（且 max_area_px > 0）→ 原樣回傳，避免動到大物件。
      - 否則在 [0, bbox_diag * 0.5] 範圍二分搜尋 cv2.approxPolyDP 的 epsilon，
        把點數壓到 [min_vertices, target_vertices]。
      - 不做 convex hull，保留凹凸結構（箭頭凹口、文字凹點得以存活）。
    """
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return arr

    was_closed = _is_closed_ring(arr)
    core = _strip_closing_point(arr)
    n = core.shape[0]

    target_vertices = max(3, int(target_vertices))
    min_vertices = max(3, int(min_vertices))
    if min_vertices > target_vertices:
        min_vertices = target_vertices

    if n <= min_vertices:
        return _restore_geometry(core, was_closed)

    x_min, y_min = core.min(axis=0)
    x_max, y_max = core.max(axis=0)
    bbox_w = float(x_max - x_min)
    bbox_h = float(y_max - y_min)
    bbox_area = bbox_w * bbox_h
    if max_area_px > 0.0 and bbox_area > max_area_px:
        return _restore_geometry(core, was_closed)

    diag = float(np.hypot(bbox_w, bbox_h))
    if diag <= 1e-6:
        return _restore_geometry(core, was_closed)

    if n <= target_vertices:
        return _restore_geometry(core, was_closed)

    pts_f32 = core.astype(np.float32)
    lo, hi = 0.0, diag * 0.5
    best_pts: Optional[np.ndarray] = None
    best_count = n
    tol = diag * 1e-4

    for _ in range(24):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(pts_f32, mid, closed=True).reshape(-1, 2)
        cnt = approx.shape[0]
        if cnt > target_vertices:
            lo = mid
            if cnt < best_count and cnt >= min_vertices:
                best_count = cnt
                best_pts = approx
        elif cnt < min_vertices:
            hi = mid
        else:
            best_pts = approx
            best_count = cnt
            hi = mid
        if (hi - lo) <= tol:
            break

    if best_pts is None or best_pts.shape[0] < 3:
        return _restore_geometry(core, was_closed)

    return _restore_geometry(best_pts.astype(np.float32), was_closed)


def _apply_polygon_steps(seg: np.ndarray, steps: Optional[List[Dict[str, Any]]], class_id: int) -> np.ndarray:
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or not steps:
        return arr

    out = arr
    for step in steps:
        name = str(step.get("name", "")).strip().lower()
        if name not in {"convex_hull", "rdp", "visvalingam_whyatt", "min_area_rect", "export_line", "small_object_fit"}:
            continue
        class_filter = step.get("class_filter")
        if class_filter is not None and class_id not in class_filter:
            continue
        count = max(1, int(step.get("count", 1)))
        eps_coeff = float(step.get("eps_coeff", 1.0))
        min_aspect = max(0.0, float(step.get("min_aspect", 0.0)))
        max_area_px = max(0.0, float(step.get("max_area_px", 5000.0)))
        target_vertices = max(3, int(step.get("target_vertices", 8)))
        for _ in range(count):
            if name in {"convex_hull", "rdp", "visvalingam_whyatt"}:
                out = _simplify_geometry(out, name, eps_coeff)
            elif name == "min_area_rect":
                out = _polygon_to_min_area_rect(out, min_aspect=min_aspect)
            elif name == "export_line":
                out = _polygon_to_long_axis_line(out, min_aspect=min_aspect)
            elif name == "small_object_fit":
                out = _polygon_fit_small_object(out, max_area_px, target_vertices)
    return out.astype(np.float32)


def _apply_object_lane_line(objects: List[Dict[str, Any]], step: Optional[Dict[str, Any]]):
    if not objects or not step:
        return

    if str(step.get("name", "")).strip().lower() != "polygon_to_lane_line":
        return

    class_filter = step.get("class_filter")
    count = max(1, int(step.get("count", 1)))

    for _ in range(count):
        for obj in objects:
            cid = int(obj.get("class_id", -1))
            if class_filter is not None and cid not in class_filter:
                continue

            polys = obj.get("polygons", [])
            if not polys:
                continue

            poly_holes_aligned = _aligned_holes(obj, len(polys))
            eps_coeff = float(step.get("eps_coeff", 1.0))
            extracted_lines: List[np.ndarray] = []
            all_points: List[List[float]] = []

            for poly, poly_holes in zip(polys, poly_holes_aligned):
                arr = np.asarray(poly, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue

                is_closed = _is_closed_ring(arr)
                if is_closed:
                    ring_line = None
                    if len(poly_holes) == 1:
                        # 甜甜圈（外環 + 單一洞）→ 中點配對法產生閉合環形中心線
                        hole_arr = np.asarray(poly_holes[0], dtype=np.float32)
                        ring_line = _annulus_centerline(arr, hole_arr, eps_coeff=eps_coeff)
                    if ring_line is None:
                        ring_line = _ring_to_lane_line(arr, eps_coeff=eps_coeff)
                    if ring_line is not None and ring_line.shape[0] >= 2:
                        extracted_lines.append(ring_line)
                    arr = arr[:-1]

                if arr.shape[0] >= 2:
                    all_points.extend(arr[:, :2].astype(float).tolist())

            if not extracted_lines and len(all_points) >= 2:
                fallback = _points_to_lane_line(np.asarray(all_points, dtype=np.float32), eps_coeff=eps_coeff)
                if fallback is not None and fallback.shape[0] >= 2:
                    extracted_lines.append(fallback)

            if not extracted_lines:
                continue

            obj["polygons"] = [ln.astype(float).tolist() for ln in extracted_lines]
            obj["holes"] = [[] for _ in extracted_lines]
            # 標記給 _sync_line_string_payloads：閉合的環形中心線應視為 LineString 而非面
            obj["_lane_line_applied"] = True


def _sync_line_string_payloads(objects: List[Dict[str, Any]]):
    for obj in objects:
        polys = obj.get("polygons", [])
        holes = _aligned_holes(obj, len(polys))
        lane_applied = bool(obj.pop("_lane_line_applied", False))

        normalized_polys: List[List[List[float]]] = []
        normalized_holes: List[List[List[List[float]]]] = []
        line_strings: List[Dict[str, Any]] = []

        for poly, poly_holes in zip(polys, holes):
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue

            if _is_closed_ring(arr):
                coords = _close_ring(_strip_closing_point(arr).astype(float).tolist())
                if lane_applied:
                    # 環形 lane line：閉合 LineString（首尾相同），語義是線不是面
                    line_strings.append({"type": "LineString", "coordinates": coords})
            else:
                coords = arr.astype(float).tolist()
                line_strings.append({"type": "LineString", "coordinates": coords})

            normalized_polys.append(coords)
            group_holes: List[List[List[float]]] = []
            for hole in poly_holes:
                hole_arr = np.asarray(hole, dtype=np.float32)
                if hole_arr.ndim == 2 and hole_arr.shape[0] >= 3:
                    group_holes.append(_close_ring(_strip_closing_point(hole_arr).astype(float).tolist()))
            normalized_holes.append(group_holes)

        obj["polygons"] = normalized_polys
        obj["holes"] = normalized_holes
        if line_strings:
            obj["line_strings"] = line_strings
        else:
            obj.pop("line_strings", None)


def _apply_ordered_polygon_steps(objects: List[Dict[str, Any]], steps: Optional[List[Dict[str, Any]]]):
    if not objects or not steps:
        return

    step_timings: List[Tuple[str, float]] = []
    for step in steps:
        _t_step = time.perf_counter()
        name = str(step.get("name", "")).strip().lower()
        if name in {"convex_hull", "rdp", "visvalingam_whyatt", "min_area_rect", "export_line", "small_object_fit"}:
            # 洞語義：點數簡化（rdp/visvalingam）同步套用到洞；
            # export_line 轉成線後洞失去意義 → 丟棄；其餘形狀替換步驟保留洞不動。
            # 注意：洞的處理必須跟 _apply_polygon_steps 一樣尊重類別篩選，
            # 否則 export_line 只作用於部分類別時，其他類別（如甜甜圈）的洞會被誤刪
            simplifies_holes = name in {"rdp", "visvalingam_whyatt"}
            drops_holes = name == "export_line"
            step_class_filter = step.get("class_filter")
            for obj in objects:
                cid = int(obj.get("class_id", -1))
                step_applies = step_class_filter is None or cid in step_class_filter
                polys = obj.get("polygons", [])
                holes = _aligned_holes(obj, len(polys))
                updated_polys: List[List[List[float]]] = []
                updated_holes: List[List[List[List[float]]]] = []
                for poly, poly_holes in zip(polys, holes):
                    arr = np.asarray(poly, dtype=np.float32)
                    if arr.ndim != 2 or arr.shape[0] < 2:
                        continue
                    updated = _apply_polygon_steps(arr, [step], cid)
                    updated_polys.append(updated.astype(float).tolist())
                    if drops_holes and step_applies:
                        updated_holes.append([])
                    elif simplifies_holes and step_applies and poly_holes:
                        new_holes: List[List[List[float]]] = []
                        for hole in poly_holes:
                            hole_arr = np.asarray(hole, dtype=np.float32)
                            if hole_arr.ndim != 2 or hole_arr.shape[0] < 2:
                                continue
                            new_holes.append(_apply_polygon_steps(hole_arr, [step], cid).astype(float).tolist())
                        updated_holes.append(new_holes)
                    else:
                        updated_holes.append(list(poly_holes))
                obj["polygons"] = updated_polys
                obj["holes"] = updated_holes
        elif name == "polygon_to_lane_line":
            _apply_object_lane_line(objects, step)
        elif name == "pca":
            _apply_pca_alignment(objects, step)
        count = max(1, int(step.get("count", 1) or 1))
        step_timings.append((f"{name} x{count}", (time.perf_counter() - _t_step) * 1000))

    _t0 = time.perf_counter()
    _sync_line_string_payloads(objects)
    sync_ms = (time.perf_counter() - _t0) * 1000

    steps_detail = " | ".join(f"{name} {ms:.1f}ms" for name, ms in step_timings)
    total_ms = sync_ms + sum(ms for _, ms in step_timings)
    print(
        f"[Timing] polygon_opt ({len(objects)} objects): total {total_ms:.1f}ms"
        f" | {steps_detail} | sync {sync_ms:.1f}ms"
    )



def _line_endpoints(seg: np.ndarray):
    if seg.ndim != 2 or seg.shape[0] < 2 or seg.shape[1] < 2:
        return None
    if np.allclose(seg[0], seg[-1]):
        # 封閉 ring 不視為線段
        return None
    if seg.shape[0] == 2:
        return seg.astype(np.float32)
    # 多點開放折線取首尾點作為主方向
    return np.vstack([seg[0], seg[-1]]).astype(np.float32)


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


def _apply_pca_alignment(objects: List[Dict[str, Any]], step: Optional[Dict[str, Any]]):
    if not objects or not step:
        return

    if str(step.get("name", "")).strip().lower() != "pca":
        return

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

        for entries in grouped_entries.values():
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


# 小於此面積（px²）的洞視為遮罩雜訊，直接忽略
MIN_HOLE_AREA_PX = 4.0


def _mask_to_ring_groups(binary_mask: np.ndarray) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """二值遮罩 -> [(外環, [洞環...]), ...]（遮罩座標，未縮放）。

    取代 ultralytics masks2segments 的 RETR_EXTERNAL：
    - RETR_CCOMP 保留兩層階層（外環 + 洞），解決甜甜圈空洞遺失問題
    - 多個獨立外環各自成組，不再被 merge_multi_segment 橋接成單一折線
    """
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # (n, 4): [next, prev, first_child, parent]

    groups: List[Tuple[np.ndarray, List[np.ndarray]]] = []
    contour_to_group: Dict[int, int] = {}
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue  # 洞，第二輪處理
        ring = contour.reshape(-1, 2).astype(np.float32)
        if ring.shape[0] < 3:
            continue
        contour_to_group[i] = len(groups)
        groups.append((ring, []))

    for i, contour in enumerate(contours):
        parent = int(hierarchy[i][3])
        if parent == -1 or parent not in contour_to_group:
            continue
        hole = contour.reshape(-1, 2).astype(np.float32)
        if hole.shape[0] < 3 or cv2.contourArea(hole) < MIN_HOLE_AREA_PX:
            continue
        groups[contour_to_group[parent]][1].append(hole)

    return groups


def _aligned_holes(obj: Dict[str, Any], n: int) -> List[List[Any]]:
    """取出與 polygons 索引對齊的 holes（缺漏補空清單）。"""
    holes = obj.get("holes") or []
    out = [list(h) if h else [] for h in holes[:n]]
    while len(out) < n:
        out.append([])
    return out


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
    masks_data = getattr(result.masks, "data", None) if has_masks else None
    mask_shape = tuple(masks_data.shape[1:]) if masks_data is not None else None
    if masks_data is not None:
        from ultralytics.utils import ops as _ultra_ops  # 延遲載入，避免單純使用本模組時引入 ultralytics

    objects: List[Dict[str, Any]] = []

    for i, (box, cid) in enumerate(zip(xyxy, cls_arr)):
        # 類別篩選
        if allowed_class_ids is not None and cid not in set(allowed_class_ids):
            continue

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        class_name = names.get(cid, str(cid))
        conf = float(conf_arr[i]) if conf_arr is not None and i < len(conf_arr) else None

        # --- 產生 polygons（外環）與 holes（洞，索引與 polygons 對齊） ---
        polys: List[List[List[float]]] = []
        holes: List[List[List[List[float]]]] = []

        if masks_data is not None and i < len(masks_data):
            m = masks_data[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            binary = (m > 0.5).astype(np.uint8)

            for ext_ring, hole_rings in _mask_to_ring_groups(binary):
                ext = _ultra_ops.scale_coords(mask_shape, ext_ring, result.orig_shape, normalize=False)
                ext = _simplify_segment(ext, simplify_mode, simplify_eps_coeff)
                if ext.shape[0] < 3:
                    continue
                polys.append(_close_ring(ext.astype(float).tolist()))

                group_holes: List[List[List[float]]] = []
                for hole_ring in hole_rings:
                    hole = _ultra_ops.scale_coords(mask_shape, hole_ring, result.orig_shape, normalize=False)
                    hole = _simplify_segment(hole, simplify_mode, simplify_eps_coeff)
                    if hole.shape[0] >= 3:
                        group_holes.append(_close_ring(hole.astype(float).tolist()))
                holes.append(group_holes)
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
            holes.append([])

        objects.append(
            {
                "class_id": int(cid),
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "polygons": polys,
                "holes": holes,
            }
        )

    _apply_ordered_polygon_steps(objects, resolved_polygon_steps)

    return objects
