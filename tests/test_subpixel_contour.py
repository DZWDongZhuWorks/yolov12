"""sub-pixel 輪廓抽取的回歸測試（層次 0：治本去鋸齒，opt-in 預設 OFF）。

涵蓋：
1. 傾斜/圓形邊界：sub-pixel 抽取比 hard-threshold 更平滑（半徑 std 較小）
2. 甜甜圈：sub-pixel 開啟時仍正確偵測到洞、外環/洞半徑帶成立
3. 預設關閉 → 既有 hard-threshold 路徑不受影響（見 tests/test_donut_polygon.py）
"""
import numpy as np

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
        self.data = np.asarray(data, dtype=np.float32)  # 浮點機率圖 (n, h, w)


class _Result:
    def __init__(self, prob_mask, names=None):
        prob = np.asarray(prob_mask, dtype=np.float32)
        ys, xs = np.where(prob > 0.5)
        self.names = names or {0: "ring"}
        self.orig_shape = prob.shape
        self.boxes = _Boxes(
            [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]],
            [0],
            [0.9],
        )
        self.masks = _Masks(prob[None])


def _radius(points, center=(100.0, 100.0)):
    arr = np.asarray(points, dtype=np.float32)
    return np.linalg.norm(arr - np.asarray(center, dtype=np.float32), axis=1)


def _aa_disk(size=200, center=(100, 100), r=70):
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return np.clip(0.5 + (r - dist), 0.0, 1.0).astype(np.float32)  # 1px 反鋸齒邊


def _aa_donut(size=200, center=(100, 100), r_out=80, r_in=40):
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    outer = np.clip(0.5 + (r_out - dist), 0.0, 1.0)
    inner = np.clip(0.5 + (dist - r_in), 0.0, 1.0)
    return np.minimum(outer, inner).astype(np.float32)


def test_subpixel_contour_smoother_than_hard_threshold():
    prob = _aa_disk()
    hard = polygon_utils.build_objects_from_result(_Result(prob))
    sub = polygon_utils.build_objects_from_result(
        _Result(prob), subpixel_contour=True, subpixel_scale=4
    )

    r_hard = _radius(hard[0]["polygons"][0])
    r_sub = _radius(sub[0]["polygons"][0])

    # sub-pixel 邊界更貼近真實圓（半徑 70）：最大徑向偏差更小
    # 註：不用 std——CHAIN_APPROX_SIMPLE 會把 hard 的共線階梯收成少數點，
    # 使頂點 std 受點數分佈干擾；對「真實形狀」的偏差才是乾淨的階梯量度。
    assert np.abs(r_sub - 70.0).max() < np.abs(r_hard - 70.0).max()
    # 平均半徑也更接近真值（量化偏差更小）
    assert abs(float(r_sub.mean()) - 70.0) < abs(float(r_hard.mean()) - 70.0)


def test_subpixel_preserves_donut_hole():
    prob = _aa_donut()
    objects = polygon_utils.build_objects_from_result(
        _Result(prob), subpixel_contour=True, subpixel_scale=3
    )
    obj = objects[0]

    assert len(obj["polygons"]) == 1
    assert len(obj["holes"]) == 1 and len(obj["holes"][0]) == 1

    outer_r = _radius(obj["polygons"][0])
    hole_r = _radius(obj["holes"][0][0])
    assert 75 <= outer_r.mean() <= 85
    assert 35 <= hole_r.mean() <= 45


def test_subpixel_scale_one_matches_hard_threshold():
    # scale=1 退化為單純 0.5 閾值，應與預設 hard-threshold 路徑一致
    prob = _aa_disk()
    hard = polygon_utils.build_objects_from_result(_Result(prob))
    sub1 = polygon_utils.build_objects_from_result(
        _Result(prob), subpixel_contour=True, subpixel_scale=1
    )
    assert len(sub1[0]["polygons"][0]) == len(hard[0]["polygons"][0])
