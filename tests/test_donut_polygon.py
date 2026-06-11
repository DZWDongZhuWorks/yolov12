"""甜甜圈（中空環形遮罩）問題的回歸測試。

涵蓋：
1. mask -> polygon 轉換保留洞（RETR_CCOMP 階層提取）
2. 多個獨立外環不再被橋接成單一折線
3. polygon_to_lane_line 對「外環 + 單一洞」輸出閉合環形中心線（中點配對法）
4. 優化步驟與 sync 維持 holes 對齊
5. GeoJSON 轉換：帶洞 Polygon 與環形 LineString
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
        self.data = np.asarray(data, dtype=np.float32)  # (n, h, w)


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


# ---------------------------------------------------------------------------
# 1. 洞保留
# ---------------------------------------------------------------------------


def test_donut_mask_keeps_hole():
    objects = polygon_utils.build_objects_from_result(_Result(_donut_mask()))

    assert len(objects) == 1
    obj = objects[0]
    assert len(obj["polygons"]) == 1
    assert len(obj["holes"]) == 1 and len(obj["holes"][0]) == 1

    outer_r = _radius(obj["polygons"][0])
    hole_r = _radius(obj["holes"][0][0])
    assert 75 <= outer_r.mean() <= 85
    assert 35 <= hole_r.mean() <= 45


def test_tiny_hole_is_ignored_as_noise():
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 80, 1, -1)
    mask[100, 100] = 0  # 1px 雜訊洞，面積 < MIN_HOLE_AREA_PX
    objects = polygon_utils.build_objects_from_result(_Result(mask))
    assert objects[0]["holes"] == [[]]


# ---------------------------------------------------------------------------
# 2. 多個獨立外環不被橋接
# ---------------------------------------------------------------------------


def test_multi_component_mask_not_bridged():
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (50, 100), 30, 1, -1)
    cv2.circle(mask, (150, 100), 30, 1, -1)
    objects = polygon_utils.build_objects_from_result(_Result(mask))

    obj = objects[0]
    assert len(obj["polygons"]) == 2
    assert obj["holes"] == [[], []]


# ---------------------------------------------------------------------------
# 3. 環形 lane line
# ---------------------------------------------------------------------------


def _lane_steps():
    return [{"name": "polygon_to_lane_line", "count": 1, "eps_coeff": 1.0, "classes": None}]


def test_annulus_lane_line_is_closed_ring():
    objects = polygon_utils.build_objects_from_result(
        _Result(_donut_mask()), polygon_opt_steps=_lane_steps()
    )

    obj = objects[0]
    assert len(obj["polygons"]) == 1
    line = np.asarray(obj["polygons"][0], dtype=np.float32)

    # 閉合（首尾相同）且半徑貼近理想中心線 r=60
    assert np.allclose(line[0], line[-1])
    radii = _radius(line[:-1])
    assert abs(float(radii.mean()) - 60.0) < 3.0
    assert float(radii.std()) < 3.0

    # 環形 lane line 必須以 LineString 形式輸出（語義是線不是面）
    assert "line_strings" in obj and len(obj["line_strings"]) == 1
    ls = obj["line_strings"][0]
    assert ls["type"] == "LineString"
    assert ls["coordinates"][0] == ls["coordinates"][-1]

    # 轉換後洞已被消耗
    assert obj["holes"] == [[]]


def test_solid_ring_lane_line_still_works():
    """無洞的實心遮罩走原本 medial axis 路徑，不應因 holes 改動而壞掉。"""
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (20, 90), (180, 110), 1, -1)  # 細長條
    objects = polygon_utils.build_objects_from_result(
        _Result(mask), polygon_opt_steps=_lane_steps()
    )
    obj = objects[0]
    assert len(obj["polygons"]) >= 1
    assert "line_strings" in obj


# ---------------------------------------------------------------------------
# 4. 步驟與 sync 的 holes 對齊
# ---------------------------------------------------------------------------


def test_rdp_step_keeps_and_simplifies_holes():
    objects = polygon_utils.build_objects_from_result(
        _Result(_donut_mask()),
        polygon_opt_steps=[{"name": "rdp", "count": 1, "eps_coeff": 1.0, "classes": None}],
    )
    obj = objects[0]
    assert len(obj["holes"]) == len(obj["polygons"]) == 1
    assert len(obj["holes"][0]) == 1
    hole_r = _radius(obj["holes"][0][0])
    assert 35 <= hole_r.mean() <= 45


def test_export_line_step_drops_holes():
    objects = polygon_utils.build_objects_from_result(
        _Result(_donut_mask()),
        polygon_opt_steps=[{"name": "export_line", "count": 1, "eps_coeff": 1.0,
                            "min_aspect": 0.0, "classes": None}],
    )
    obj = objects[0]
    assert obj["holes"] == [[] for _ in obj["polygons"]]


# ---------------------------------------------------------------------------
# 5. GeoJSON 轉換
# ---------------------------------------------------------------------------


def _identity_converter_args():
    from app_utils.tool.tfw2lonlat import WorldFile
    from pyproj import Transformer

    wf = WorldFile(A=1.0, D=0.0, B=0.0, E=1.0, C=0.0, F=0.0)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    return wf, transformer


def _payload(objects):
    return {
        "image": {"file_name": "t.png", "width": 200, "height": 200},
        "model": {"name": "test"},
        "objects": objects,
    }


def test_geojson_polygon_with_hole():
    from app_utils.tool.tfw2lonlat import convert_yolo_json_to_geojson

    objects = polygon_utils.build_objects_from_result(_Result(_donut_mask()))
    wf, transformer = _identity_converter_args()
    geojson = convert_yolo_json_to_geojson(_payload(objects), wf, transformer, 0.0, 0.0)

    geometry = geojson["features"][0]["geometry"]
    assert geometry["type"] == "Polygon"
    assert len(geometry["coordinates"]) == 2  # [外環, 洞]
    outer_r = _radius([c[:2] for c in geometry["coordinates"][0]])
    hole_r = _radius([c[:2] for c in geometry["coordinates"][1]])
    assert outer_r.mean() > hole_r.mean()


def test_geojson_ring_lane_line_is_linestring():
    from app_utils.tool.tfw2lonlat import convert_yolo_json_to_geojson

    objects = polygon_utils.build_objects_from_result(
        _Result(_donut_mask()), polygon_opt_steps=_lane_steps()
    )
    wf, transformer = _identity_converter_args()
    geojson = convert_yolo_json_to_geojson(_payload(objects), wf, transformer, 0.0, 0.0)

    geometry = geojson["features"][0]["geometry"]
    assert geometry["type"] == "LineString"
    assert geometry["coordinates"][0] == geometry["coordinates"][-1]  # 閉合環形線


def test_geojson_legacy_object_without_holes_field():
    """舊版 JSON（無 holes 欄位）必須維持原行為。"""
    from app_utils.tool.tfw2lonlat import convert_yolo_json_to_geojson

    legacy_obj = {
        "class_id": 0,
        "class_name": "ring",
        "confidence": 0.9,
        "bbox_xyxy": [0, 0, 10, 10],
        "polygons": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]],
    }
    wf, transformer = _identity_converter_args()
    geojson = convert_yolo_json_to_geojson(_payload([legacy_obj]), wf, transformer, 0.0, 0.0)

    geometry = geojson["features"][0]["geometry"]
    assert geometry["type"] == "Polygon"
    assert len(geometry["coordinates"]) == 1
