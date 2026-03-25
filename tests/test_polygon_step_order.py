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
    def __init__(self, polygons):
        self.xy = polygons


class _Result:
    def __init__(self, polygon):
        arr = np.asarray(polygon, dtype=np.float32)
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        self.names = {0: "lane"}
        self.boxes = _Boxes([[x_min, y_min, x_max, y_max]], [0], [0.95])
        self.masks = _Masks([arr[:-1]])


def _closed_ribbon_polygon():
    return [
        [0.0, 5.0],
        [10.0, 8.0],
        [20.0, 2.0],
        [30.0, 7.0],
        [40.0, 3.0],
        [50.0, 5.0],
        [50.0, -5.0],
        [40.0, -3.0],
        [30.0, -7.0],
        [20.0, -2.0],
        [10.0, -8.0],
        [0.0, -5.0],
        [0.0, 5.0],
    ]


def _open_wavy_line():
    return [
        [0.0, 0.0],
        [10.0, 0.2],
        [20.0, -0.1],
        [30.0, 0.1],
        [40.0, 0.0],
    ]


def _apply_steps(polygons, steps):
    objects = [{"class_id": 0, "polygons": [[list(pt) for pt in poly] for poly in polygons]}]
    polygon_utils._apply_ordered_polygon_steps(objects, steps)
    return objects


def test_polygon_to_lane_line_can_be_followed_by_visvalingam(monkeypatch):
    fake_line = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.2],
            [20.0, -0.1],
            [30.0, 0.1],
            [40.0, -0.2],
            [50.0, 0.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        polygon_utils,
        "_ring_to_lane_line",
        lambda ring, eps_coeff=1.0: fake_line.copy(),
    )

    objects = _apply_steps(
        [_closed_ribbon_polygon()],
        [
            {"name": "polygon_to_lane_line", "count": 1, "eps_coeff": 1.0, "class_filter": None},
            {"name": "visvalingam_whyatt", "count": 1, "eps_coeff": 10.0, "class_filter": None},
        ],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    assert out.shape[0] < fake_line.shape[0]
    assert np.allclose(out[0], fake_line[0])
    assert np.allclose(out[-1], fake_line[-1])
    assert not np.allclose(out[0], out[-1])


def test_rdp_simplifies_open_linestring():
    original = np.asarray(_open_wavy_line(), dtype=np.float32)
    objects = _apply_steps(
        [_open_wavy_line()],
        [{"name": "rdp", "count": 1, "eps_coeff": 5.0, "class_filter": None}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    assert out.shape[0] < original.shape[0]
    assert np.allclose(out[0], original[0])
    assert np.allclose(out[-1], original[-1])
    assert not np.allclose(out[0], out[-1])


def test_visvalingam_simplifies_open_linestring():
    original = np.asarray(_open_wavy_line(), dtype=np.float32)
    objects = _apply_steps(
        [_open_wavy_line()],
        [{"name": "visvalingam_whyatt", "count": 1, "eps_coeff": 10.0, "class_filter": None}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    assert out.shape[0] < original.shape[0]
    assert np.allclose(out[0], original[0])
    assert np.allclose(out[-1], original[-1])
    assert not np.allclose(out[0], out[-1])


def test_build_objects_from_result_uses_ordered_polygon_steps(monkeypatch):
    fake_line = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.2],
            [20.0, -0.1],
            [30.0, 0.1],
            [40.0, -0.2],
            [50.0, 0.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        polygon_utils,
        "_ring_to_lane_line",
        lambda ring, eps_coeff=1.0: fake_line.copy(),
    )

    objects = polygon_utils.build_objects_from_result(
        _Result(_closed_ribbon_polygon()),
        polygon_opt_steps=[
            {"name": "polygon_to_lane_line", "count": 1, "eps_coeff": 1.0, "classes": None},
            {"name": "rdp", "count": 1, "eps_coeff": 5.0, "classes": None},
        ],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    assert out.shape[0] < fake_line.shape[0]
    assert np.allclose(out[0], fake_line[0])
    assert np.allclose(out[-1], fake_line[-1])
    assert "line_strings" in objects[0]
