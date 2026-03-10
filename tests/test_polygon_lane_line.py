import numpy as np

from app_utils.inference import parse_polygon_steps
from app_utils.polygon_utils import _polygon_to_trsnd_lane_line


def _make_curved_lane_polygon(n=40, width=6.0):
    t = np.linspace(0.0, 1.0, n)
    x = 20 + 120 * t
    y = 30 + 40 * np.sin(t * np.pi * 0.8)
    center = np.stack([x, y], axis=1)

    tangents = np.gradient(center, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.clip(norms, 1e-6, None)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    left = center + normals * (width * 0.5)
    right = center - normals * (width * 0.5)
    poly = np.vstack([left, right[::-1]])
    return poly.astype(np.float32)


def test_parse_polygon_steps_accepts_trsnd_lane_line_polygon():
    steps = parse_polygon_steps("trsnd_lane_line_polygon:1:1.0:2.5")
    assert len(steps) == 1
    assert steps[0]["name"] == "trsnd_lane_line_polygon"
    assert steps[0]["min_aspect"] == 2.5


def test_trsnd_lane_line_polygon_outputs_polyline_with_endpoints():
    poly = _make_curved_lane_polygon()
    line = _polygon_to_trsnd_lane_line(poly, min_aspect=2.0, eps_coeff=1.0)

    assert line.shape[0] >= 3
    assert line.shape[1] == 2

    # Endpoints should stay near polygon longitudinal extremes.
    x_min, x_max = float(poly[:, 0].min()), float(poly[:, 0].max())
    line_x = line[:, 0]
    assert abs(float(line_x.min()) - x_min) < 12.0
    assert abs(float(line_x.max()) - x_max) < 12.0


def test_trsnd_lane_line_polygon_keeps_turn_information():
    poly = _make_curved_lane_polygon()
    line = _polygon_to_trsnd_lane_line(poly, min_aspect=2.0, eps_coeff=1.0)

    diffs = np.diff(line, axis=0)
    assert diffs.shape[0] >= 2
    turns = []
    for i in range(diffs.shape[0] - 1):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        c = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
        turns.append(np.degrees(np.arccos(c)))

    assert turns
    assert max(turns) > 5.0
