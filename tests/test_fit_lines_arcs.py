"""fit_lines_arcs step 的回歸測試（層次 2：基元擬合，直線+圓弧）。

涵蓋：
1. 鋸齒直線 → 收斂成乾淨直線（少數點、低殘差、端點保留）
2. 稠密圓環 → 圓弧重建（半徑 std 小、環精確閉合）
3. L 形 → 真實 90° 銳角存活、兩臂各自筆直（不被磨圓）
4. 帶洞物件 → holes 與 polygons 對齊、半徑帶保留
"""
import numpy as np
import cv2

from app_utils import polygon_utils


class _ArrayWrapper:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _ArrayWrapper(xyxy)
        self.cls = _ArrayWrapper(cls)
        self.conf = _ArrayWrapper(conf)

    def __len__(self):
        return int(self.xyxy.numpy().shape[0])


class _Masks:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class _Result:
    def __init__(self, mask, names=None):
        mask = np.asarray(mask, dtype=np.uint8)
        ys, xs = np.where(mask > 0)
        self.names = names or {0: "ring"}
        self.orig_shape = mask.shape
        self.boxes = _Boxes(
            [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]],
            [0],
            [0.9],
        )
        self.masks = _Masks(mask[None])


def _donut_mask(size=200, center=(100, 100), r_outer=80, r_inner=40):
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, center, r_outer, 1, -1)
    cv2.circle(mask, center, r_inner, 0, -1)
    return mask


def _radius(points, center=(100.0, 100.0)):
    arr = np.asarray(points, dtype=np.float32)
    return np.linalg.norm(arr - np.asarray(center, dtype=np.float32), axis=1)


def _jagged_line(n=60, length=100.0, amp=0.4):
    xs = np.linspace(0.0, length, n)
    ys = amp * np.sin(np.arange(n) * 1.7)  # 確定性的次像素抖動
    return np.column_stack([xs, ys]).astype(np.float32)


def _circle_ring(n=200, r=60.0, center=(100.0, 100.0), wobble=0.5):
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rad = r + wobble * np.sin(ang * 7.0)  # 確定性微抖動
    xs = center[0] + rad * np.cos(ang)
    ys = center[1] + rad * np.sin(ang)
    ring = np.column_stack([xs, ys]).astype(np.float32)
    return np.vstack([ring, ring[:1]])


def _l_shape(per_leg=25, leg=50.0, corner=(50.0, 0.0), amp=0.15):
    # leg1: (0,0)->(50,0)；leg2: (50,0)->(50,50)；轉角在 (50,0)
    t = np.linspace(0.0, 1.0, per_leg)
    leg1 = np.column_stack([t * corner[0], amp * np.sin(np.arange(per_leg) * 2.1)])
    leg2 = np.column_stack([
        corner[0] + amp * np.sin(np.arange(per_leg) * 1.3),
        t * leg,
    ])
    return np.vstack([leg1, leg2[1:]]).astype(np.float32)


def _apply_steps(polygons, steps):
    objects = [{"class_id": 0, "polygons": [[list(pt) for pt in poly] for poly in polygons]}]
    polygon_utils._apply_ordered_polygon_steps(objects, steps)
    return objects


def _max_dev_to_line(pts, a, b):
    """點集到線段 ab 的最大垂距。"""
    pts = np.asarray(pts, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return float(np.linalg.norm(pts - a, axis=1).max())
    t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return float(np.linalg.norm(pts - proj, axis=1).max())


def test_fit_lines_arcs_straightens_jagged_line():
    original = _jagged_line()
    objects = _apply_steps(
        [original],
        [{"name": "fit_lines_arcs", "count": 1, "eps_coeff": 1.0, "class_filter": None}],
    )
    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)

    # 鋸齒直線收斂成弦（極少數點）
    assert out.shape[0] <= 3
    # 端點保留
    assert np.allclose(out[0], original[0], atol=1e-3)
    assert np.allclose(out[-1], original[-1], atol=1e-3)
    # 原始點都貼近輸出直線
    assert _max_dev_to_line(original, out[0], out[-1]) < 1.0


def test_fit_lines_arcs_recovers_circle():
    original = _circle_ring()
    objects = _apply_steps(
        [original],
        [{"name": "fit_lines_arcs", "count": 1, "eps_coeff": 2.0, "class_filter": None}],
    )
    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)

    # 精確閉合
    assert np.allclose(out[0], out[-1])
    # 圓弧重建：半徑緊（比 RDP 的弦折更平滑）
    radii = _radius(out[:-1])
    assert abs(float(radii.mean()) - 60.0) < 2.0
    assert float(radii.std()) < 1.0


def test_fit_lines_arcs_preserves_real_corner():
    original = _l_shape()
    objects = _apply_steps(
        [original],
        [{"name": "fit_lines_arcs", "count": 1, "eps_coeff": 1.0, "class_filter": None}],
    )
    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)

    # 兩臂 → 三點（起點、轉角、終點）
    assert out.shape[0] <= 4
    # 真實 90° 轉角存活（輸出有一點貼近 (50,0)）
    corner = np.array([50.0, 0.0], dtype=np.float32)
    assert float(np.linalg.norm(out - corner, axis=1).min()) < 1.0
    # 端點保留
    assert np.allclose(out[0], original[0], atol=1e-3)
    assert np.allclose(out[-1], original[-1], atol=1e-3)


def test_fit_lines_arcs_keeps_holes_aligned():
    objects = polygon_utils.build_objects_from_result(
        _Result(_donut_mask()),
        polygon_opt_steps=[{"name": "fit_lines_arcs", "count": 1, "eps_coeff": 1.0, "classes": None}],
    )
    obj = objects[0]

    assert len(obj["polygons"]) == 1
    assert len(obj["holes"]) == 1 and len(obj["holes"][0]) == 1

    outer_r = _radius(obj["polygons"][0])
    hole_r = _radius(obj["holes"][0][0])
    assert 75 <= outer_r.mean() <= 85
    assert 35 <= hole_r.mean() <= 45
