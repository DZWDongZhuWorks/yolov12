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


# ---------------------------------------------------------------------------
# small_object_fit
# ---------------------------------------------------------------------------


def _densify_closed_ring(corners, points_per_edge: int = 15):
    """在每條邊插值，產出密化後的閉合 ring（不含結尾重複點）。"""
    corners = np.asarray(corners, dtype=np.float32)
    n = corners.shape[0]
    out = []
    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        for j in range(points_per_edge):
            t = j / points_per_edge
            out.append(a * (1.0 - t) + b * t)
    return [pt.tolist() for pt in out] + [corners[0].tolist()]


def _rhombus_corners(half_w=10.0, half_h=10.0):
    return [(0.0, half_h), (half_w, 0.0), (0.0, -half_h), (-half_w, 0.0)]


def _arrow_corners():
    """7 點箭頭：往右指。"""
    return [
        (0.0, -2.0),
        (8.0, -2.0),
        (8.0, -5.0),
        (15.0, 0.0),
        (8.0, 5.0),
        (8.0, 2.0),
        (0.0, 2.0),
    ]


def _t_letter_corners():
    """凹形 'T' 的 8 個角點。"""
    return [
        (0.0, 10.0),
        (10.0, 10.0),
        (10.0, 7.0),
        (6.5, 7.0),
        (6.5, 0.0),
        (3.5, 0.0),
        (3.5, 7.0),
        (0.0, 7.0),
    ]


def test_small_object_fit_rhombus_yields_four_vertices():
    poly = _densify_closed_ring(_rhombus_corners(), points_per_edge=20)
    objects = _apply_steps(
        [poly],
        [{"name": "small_object_fit", "count": 1, "max_area_px": 5000.0,
          "target_vertices": 4, "class_filter": None}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    # 4 corners + 1 closing point
    assert 4 <= out.shape[0] <= 5
    core = out[:-1] if np.allclose(out[0], out[-1]) else out
    expected = np.asarray(_rhombus_corners(), dtype=np.float32)
    for ec in expected:
        d = np.linalg.norm(core - ec, axis=1).min()
        assert d < 1.5, f"corner {ec} not preserved (min dist={d})"


def test_small_object_fit_arrow_preserves_concavity():
    poly = _densify_closed_ring(_arrow_corners(), points_per_edge=10)
    objects = _apply_steps(
        [poly],
        [{"name": "small_object_fit", "count": 1, "max_area_px": 5000.0,
          "target_vertices": 8, "class_filter": None}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    core = out[:-1] if np.allclose(out[0], out[-1]) else out
    assert 5 <= core.shape[0] <= 8

    notches = [np.asarray(_arrow_corners()[2], dtype=np.float32),
               np.asarray(_arrow_corners()[4], dtype=np.float32)]
    for notch in notches:
        d = np.linalg.norm(core - notch, axis=1).min()
        assert d < 1.5, f"arrow notch {notch} lost (min dist={d})"


def test_small_object_fit_large_polygon_unchanged():
    # bbox = 200x100 = 20000 px²，遠大於 max_area_px=5000
    corners = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]
    poly = _densify_closed_ring(corners, points_per_edge=10)
    objects = _apply_steps(
        [poly],
        [{"name": "small_object_fit", "count": 1, "max_area_px": 5000.0,
          "target_vertices": 4, "class_filter": None}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    original = np.asarray(poly, dtype=np.float32)
    assert out.shape[0] == original.shape[0]
    assert np.allclose(out, original, atol=1e-3)


def test_small_object_fit_zero_max_area_always_applies():
    corners = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]
    poly = _densify_closed_ring(corners, points_per_edge=10)
    objects = _apply_steps(
        [poly],
        [{"name": "small_object_fit", "count": 1, "max_area_px": 0.0,
          "target_vertices": 6, "class_filter": None}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    core = out[:-1] if np.allclose(out[0], out[-1]) else out
    assert core.shape[0] <= 6


def test_small_object_fit_text_outline_kept_under_limit():
    poly = _densify_closed_ring(_t_letter_corners(), points_per_edge=8)
    objects = _apply_steps(
        [poly],
        [{"name": "small_object_fit", "count": 1, "max_area_px": 5000.0,
          "target_vertices": 10, "class_filter": None}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    core = out[:-1] if np.allclose(out[0], out[-1]) else out
    assert 6 <= core.shape[0] <= 10


def test_small_object_fit_class_filter_skips_other_classes():
    poly = _densify_closed_ring(_rhombus_corners(), points_per_edge=20)
    objects = [{"class_id": 1, "polygons": [[list(pt) for pt in poly]]}]
    polygon_utils._apply_ordered_polygon_steps(
        objects,
        [{"name": "small_object_fit", "count": 1, "max_area_px": 5000.0,
          "target_vertices": 4, "class_filter": {0}}],
    )

    out = np.asarray(objects[0]["polygons"][0], dtype=np.float32)
    assert out.shape[0] == len(poly)


def test_parse_polygon_steps_handles_10col_schema():
    from app_utils.inference_optimizations import parse_polygon_steps

    rows = [["small_object_fit", True, 1, 1.0, 0.0, 0.94, False, 5000.0, 8, ""]]
    steps = parse_polygon_steps(rows)
    assert len(steps) == 1
    assert steps[0]["name"] == "small_object_fit"
    assert steps[0]["max_area_px"] == 5000.0
    assert steps[0]["target_vertices"] == 8


def test_parse_polygon_steps_handles_legacy_8col_schema():
    from app_utils.inference_optimizations import (
        parse_polygon_steps,
        DEFAULT_POLYGON_SMALL_OBJECT_MAX_AREA,
        DEFAULT_POLYGON_SMALL_OBJECT_TARGET_VERTICES,
    )

    rows = [["rdp", True, 1, 1.0, 0.0, 0.94, False, "0, 1"]]
    steps = parse_polygon_steps(rows)
    assert len(steps) == 1
    assert steps[0]["name"] == "rdp"
    assert steps[0]["max_area_px"] == DEFAULT_POLYGON_SMALL_OBJECT_MAX_AREA
    assert steps[0]["target_vertices"] == DEFAULT_POLYGON_SMALL_OBJECT_TARGET_VERTICES
