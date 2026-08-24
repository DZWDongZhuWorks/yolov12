"""pca step 的 Manhattan 正則化回歸測試（層次 3：方向正則化）。

涵蓋：
1. pca_manhattan=True：近垂直線被吸附成與主導(水平)方向精確正交，水平線彼此平行，中點不變
2. pca_manhattan 預設關閉：單一朝向的線維持原方向（既有 pca 行為不變）
"""
import numpy as np

from app_utils import polygon_utils


def _hline(y, x0=0.0, x1=10.0):
    return [[x0, y], [x1, y]]


def _line_off_vertical(deg=3.0, x=5.0, length=10.0):
    rad = np.deg2rad(deg)
    return [[x, 0.0], [x + length * np.sin(rad), length * np.cos(rad)]]


def _apply_steps(polygons_per_object, steps):
    objects = [
        {"class_id": 0, "polygons": [[list(pt) for pt in poly]]}
        for poly in polygons_per_object
    ]
    polygon_utils._apply_ordered_polygon_steps(objects, steps)
    return objects


def _direction(poly):
    p = np.asarray(poly, dtype=np.float64)
    v = p[1] - p[0]
    return v / np.linalg.norm(v)


def _midpoint(poly):
    p = np.asarray(poly, dtype=np.float64)
    return 0.5 * (p[0] + p[1])


def _pca_step(manhattan):
    return {
        "name": "pca",
        "count": 1,
        "eps_coeff": 1.0,          # alpha=1 → 完全吸附到軸
        "pca_min_cosine": 0.94,
        "pca_cross_class": False,
        "pca_manhattan": manhattan,
        "class_filter": None,
    }


def test_pca_manhattan_snaps_near_perpendicular_pair():
    polys = [_hline(0.0), _hline(20.0), _hline(40.0), _line_off_vertical(3.0)]
    original_mids = [_midpoint(p) for p in polys]

    objects = _apply_steps(polys, [_pca_step(manhattan=True)])
    out = [obj["polygons"][0] for obj in objects]

    h0, h1, h2 = _direction(out[0]), _direction(out[1]), _direction(out[2])
    v = _direction(out[3])

    # 三條水平線彼此平行
    assert abs(abs(float(np.dot(h0, h1))) - 1.0) < 1e-4
    assert abs(abs(float(np.dot(h0, h2))) - 1.0) < 1e-4
    # 近垂直線被吸附成與水平方向精確正交
    assert abs(float(np.dot(h0, v))) < 1e-4
    # 中點不變（僅旋轉，不平移）
    for poly, mid0 in zip(out, original_mids):
        assert np.allclose(_midpoint(poly), mid0, atol=1e-3)


def test_pca_manhattan_off_is_unchanged_behavior():
    polys = [_hline(0.0), _hline(20.0), _hline(40.0), _line_off_vertical(3.0)]
    in_dirs = [_direction(p) for p in polys]

    objects = _apply_steps(polys, [_pca_step(manhattan=False)])
    out_dirs = [_direction(obj["polygons"][0]) for obj in objects]

    # 預設關閉 → 每條線方向維持不變（單一朝向群只朝自身軸吸附 = 不動）
    for d_in, d_out in zip(in_dirs, out_dirs):
        assert abs(abs(float(np.dot(d_in, d_out))) - 1.0) < 1e-4

    # 且近垂直線「沒有」被扳成正交（與 manhattan=True 形成對比）
    assert abs(float(np.dot(out_dirs[0], out_dirs[3]))) > 0.02
