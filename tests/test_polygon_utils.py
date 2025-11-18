import numpy as np

from app_utils.polygon_utils import build_objects_from_result, _simplify_short_line


class DummyArray:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


class DummyBoxes:
    def __init__(self, xyxy, cls, conf=None):
        self.xyxy = DummyArray(xyxy)
        self.cls = DummyArray(cls)
        self.conf = DummyArray(conf) if conf is not None else None

    def __len__(self):
        return len(self.xyxy)


class DummyMasks:
    def __init__(self, segments):
        self.xy = segments

    def __len__(self):
        return len(self.xy)


class DummyResult:
    def __init__(self, names, boxes, masks=None):
        self.names = names
        self.boxes = boxes
        self.masks = masks


def test_default_uses_rdp_simplification():
    """RDP should be the default simplifier when none is specified."""
    elongated_square = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = DummyResult(
        names={0: "box"},
        boxes=DummyBoxes([[0, 0, 2, 2]], [0], [0.8]),
        masks=DummyMasks([[elongated_square]]),
    )

    objects = build_objects_from_result(result, simplify_eps_ratio=0.5)

    simplified = np.asarray(objects[0]["polygons"][0])
    assert len(simplified) == 4


def test_category_mapping_enables_short_line_strategy():
    """Class-to-category mapping should drive the strategy selection."""
    thin_segment = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.8],
            [0.0, 0.6],
        ],
        dtype=np.float32,
    )

    result = DummyResult(
        names={1: "wire"},
        boxes=DummyBoxes([[0, 0, 10, 1]], [1], [0.9]),
        masks=DummyMasks([[thin_segment]]),
    )

    objects = build_objects_from_result(
        result,
        class_category_map={1: "short_line"},
        category_strategy_map={"short_line": "short_line"},
        short_line_aspect_ratio=5.0,
    )

    simplified = np.asarray(objects[0]["polygons"][0])
    assert simplified.shape[0] == 4
    assert simplified[1, 0] > simplified[0, 0]


def test_multiple_contours_are_preserved():
    """Each contour should yield an independent polygon entry."""
    seg_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    seg_b = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0]], dtype=np.float32)

    result = DummyResult(
        names={2: "double"},
        boxes=DummyBoxes([[0, 0, 3, 3]], [2], [0.5]),
        masks=DummyMasks([[seg_a, seg_b]]),
    )

    objects = build_objects_from_result(result)

    polygons = objects[0]["polygons"]
    assert len(polygons) == 2
    assert np.allclose(polygons[0][0], [0.0, 0.0])


def test_short_line_helper_collapses_thin_segments():
    thin_seg = np.array(
        [[0.0, 0.0], [5.0, 0.0], [5.0, 0.2], [0.0, 0.1]],
        dtype=np.float32,
    )

    simplified = _simplify_short_line(thin_seg, aspect_ratio_threshold=5.0)

    assert simplified.shape[0] == 4
    assert simplified[0, 0] == 0.0
    assert simplified[1, 0] == 5.0

