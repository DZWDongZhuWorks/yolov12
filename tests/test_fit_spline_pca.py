"""fit_spline_pca step 的回歸測試（教授法:PCA 直線閘門 + RDP 控制點 + 內插三次樣條）。

涵蓋：
1. 近直線(在容差內)→ PCA 閘門 → 輸出兩點直線
2. 鋸齒弧線 → 平滑曲線：**保留弧形**(不被壓直)、轉角平順、端點保留
3. 封閉環 → 仍精確閉合、半徑 std 下降
"""
import numpy as np

from app_utils import polygon_utils


def _apply(poly, eps_coeff=1.0):
    objects = [{"class_id": 0, "polygons": [[list(p) for p in poly]]}]
    polygon_utils._apply_ordered_polygon_steps(
        objects, [{"name": "fit_spline_pca", "count": 1, "eps_coeff": eps_coeff, "class_filter": None}]
    )
    return np.asarray(objects[0]["polygons"][0], dtype=np.float32)


def _total_turning(pts):
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    if p.shape[0] < 3:
        return 0.0
    d = np.diff(p, axis=0); a = np.arctan2(d[:, 1], d[:, 0]); da = np.diff(a)
    da = (da + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(da).sum())


def _dev_from_chord(pts):
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    a, b = p[0], p[-1]; ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-9:
        return 0.0
    t = np.clip((p - a) @ ab / denom, 0, 1)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return float(np.linalg.norm(p - proj, axis=1).max())


def test_fit_spline_pca_straight_line_gate():
    n = 40
    xs = np.linspace(0, 200, n)
    ys = 0.5 * np.sin(np.arange(n) * 1.3)  # 抖動 < 容差
    line = np.column_stack([xs, ys]).astype(np.float32)
    out = _apply(line, eps_coeff=1.0)  # 容差 = 200*0.01*1 = 2 > 0.5

    assert out.shape[0] == 2                       # PCA 閘門 → 直線
    assert abs(out[0, 1]) < 1.0 and abs(out[-1, 1]) < 1.0
    assert abs(out[:, 0].max() - 200) < 2.0


def test_fit_spline_pca_preserves_arc():
    t = np.linspace(0, np.pi / 2, 40)
    arc = np.column_stack([100 * np.cos(t), 100 * np.sin(t)]).astype(np.float32)
    arc[:, 1] += 0.6 * np.sin(np.arange(40) * 2.1)  # 加鋸齒
    out = _apply(arc, eps_coeff=1.0)

    # 弧形被保留(沒被壓直)：弧高仍接近真實 sagitta ~29px
    assert _dev_from_chord(out) > 20.0
    # 端點保留
    assert np.allclose(out[0], arc[0], atol=1.5)
    assert np.allclose(out[-1], arc[-1], atol=1.5)
    # 平順:總轉角接近理想四分之一圓(pi/2),遠低於鋸齒輸入
    assert _total_turning(out) < _total_turning(arc)
    assert _total_turning(out) < 2.2


def test_fit_spline_pca_closed_ring_stays_closed():
    n = 80
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = 60 + 3.0 * ((-1.0) ** np.arange(n))      # 鋸齒環
    ring = np.column_stack([100 + rad * np.cos(ang), 100 + rad * np.sin(ang)]).astype(np.float32)
    ring = np.vstack([ring, ring[:1]])
    out = _apply(ring, eps_coeff=1.0)

    assert np.allclose(out[0], out[-1])            # 精確閉合
    assert out.shape[0] >= 4
    r = np.linalg.norm(out[:-1] - np.array([100.0, 100.0]), axis=1)
    assert abs(float(r.mean()) - 60.0) < 4.0
    assert float(r.std()) < 3.0 * ((-1.0) ** 0)    # 抖動下降(輸入 std=3)
    assert float(r.std()) < 3.0
