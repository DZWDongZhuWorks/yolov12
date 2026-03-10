import numpy as np
import torch

from app_utils.inference import parse_polygon_steps
from app_utils.polygon_utils import build_objects_from_result


class _DummyBoxes:
    def __init__(self):
        self.xyxy = torch.tensor([[0.0, 0.0, 40.0, 20.0]], dtype=torch.float32)
        self.cls = torch.tensor([0], dtype=torch.float32)
        self.conf = torch.tensor([0.95], dtype=torch.float32)

    def __len__(self):
        return int(self.xyxy.shape[0])


class _DummyMasks:
    def __init__(self, polygon):
        self.xy = [np.asarray(polygon, dtype=np.float32)]


class _DummyResult:
    def __init__(self, polygon):
        self.names = {0: "lane"}
        self.boxes = _DummyBoxes()
        self.masks = _DummyMasks(polygon)


def _max_turn_angle_deg(points):
    if len(points) < 3:
        return 0.0
    arr = np.asarray(points, dtype=np.float32)
    max_angle = 0.0
    for i in range(1, len(arr) - 1):
        v1 = arr[i] - arr[i - 1]
        v2 = arr[i + 1] - arr[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cos_theta = float(np.dot(v1 / n1, v2 / n2))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        ang = float(np.degrees(np.arccos(cos_theta)))
        max_angle = max(max_angle, ang)
    return max_angle


def test_parse_polygon_steps_accepts_trans_lane_line_polygon():
    steps = parse_polygon_steps("trans_lane_line_polygon:1:1.8:2.0")
    assert len(steps) == 1
    assert steps[0]["name"] == "trans_lane_line_polygon"
    assert steps[0]["eps_coeff"] == 1.8
    assert steps[0]["min_aspect"] == 2.0


def test_trans_lane_line_polygon_outputs_open_linestring_with_turn():
    bent_lane_polygon = np.array(
        [
            [0.0, -1.0],
            [10.0, -1.0],
            [20.0, 2.0],
            [30.0, 12.0],
            [32.0, 14.0],
            [28.0, 16.0],
            [18.0, 6.0],
            [8.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    result = _DummyResult(bent_lane_polygon)
    steps = [
        {
            "name": "trans_lane_line_polygon",
            "count": 1,
            "eps_coeff": 1.6,
            "min_aspect": 2.0,
            "pca_min_cosine": 0.94,
            "pca_cross_class": False,
            "class_filter": None,
        }
    ]

    objects = build_objects_from_result(result, polygon_opt_steps=steps)
    assert len(objects) == 1
    line = objects[0]["polygons"][0]

    assert len(line) >= 2
    assert line[0] != line[-1], "LineString 不應閉合成 polygon ring"
    assert _max_turn_angle_deg(line) >= 12.0, "轉彎點應保留，不應被過度簡化"
