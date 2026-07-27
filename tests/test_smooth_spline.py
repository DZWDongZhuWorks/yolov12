"""smooth_spline step 的回歸測試（層次 1：通用平滑）。

涵蓋：
1. 開放折線：鋸齒被抹平（曲率下降）、端點錨定、仍為開放線
2. 封閉環：輸出仍精確閉合、半徑 std 下降
3. 帶洞物件：holes 與 polygons 對齊、外環/洞半徑帶保留
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


def _mean_abs_curvature(pts):
    """以二階差分大小近似折線的鋸齒程度。"""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 3:
        return 0.0
    second = np.diff(arr, n=2, axis=0)
    return float(np.linalg.norm(second, axis=1).mean())


def _sawtooth_open_line(n=21, amp=3.0, step=5.0):
    xs = np.arange(n) * step
    ys = amp * ((-1.0) ** np.arange(n))
    return np.column_stack([xs, ys]).astype(np.float32)


def _noisy_closed_ring(n=72, r=60.0, center=(100.0, 100.0), noise=4.0):
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    radial = r + noise * ((-1.0) ** np.arange(n))  # 交錯徑向雜訊 → 鋸齒環
    xs = center[0] + radial * np.cos(ang)
    ys = center[1] + radial * np.sin(ang)
    ring = np.column_stack([xs, ys]).astype(np.float32)
    return np.vstack([ring, ring[:1]])  # 閉合（首尾相同）


def _apply_steps(polygons, steps):
    objects = [{"class_id": 0, "polygons": [[list(pt) for pt in poly] for poly in polygons]}]
    polygon_utils._apply_ordered_polygon_steps(objects, steps)
    return objects


def test_smooth_spline_open_reduces_jaggedness():
    original = _sawtooth_open_line()
    objects = _apply_steps(
        [original],
        [{"name": "smooth_spline", "count": 1, "eps_coeff": 3.0, "class_filter": None}],
    )
    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)

    # 鋸齒程度明顯下降
    assert _mean_abs_curvature(out) < 0.5 * _mean_abs_curvature(original)
    # 端點錨定
    assert np.allclose(out[0], original[0], atol=1e-3)
    assert np.allclose(out[-1], original[-1], atol=1e-3)
    # 仍是開放線
    assert not np.allclose(out[0], out[-1])


def test_smooth_spline_closed_ring_stays_closed():
    original = _noisy_closed_ring()
    objects = _apply_steps(
        [original],
        [{"name": "smooth_spline", "count": 1, "eps_coeff": 2.0, "class_filter": None}],
    )
    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)

    # 精確閉合（byte 相同，給 GeoJSON 匯出的 == 判斷用）
    assert np.allclose(out[0], out[-1])
    assert out.shape[0] >= 4
    # 半徑抖動下降（鋸齒被抹平）
    assert _radius(out[:-1]).std() < _radius(original[:-1]).std()
    # 形狀大致保留（平均半徑仍接近 60）
    assert abs(float(_radius(out[:-1]).mean()) - 60.0) < 4.0


def test_smooth_spline_keeps_holes_aligned():
    objects = polygon_utils.build_objects_from_result(
        _Result(_donut_mask()),
        polygon_opt_steps=[{"name": "smooth_spline", "count": 1, "eps_coeff": 1.0, "classes": None}],
    )
    obj = objects[0]

    assert len(obj["polygons"]) == 1
    assert len(obj["holes"]) == 1 and len(obj["holes"][0]) == 1

    outer_r = _radius(obj["polygons"][0])
    hole_r = _radius(obj["holes"][0][0])
    assert 75 <= outer_r.mean() <= 85
    assert 35 <= hole_r.mean() <= 45
