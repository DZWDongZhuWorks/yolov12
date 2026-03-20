import numpy as np

from app_utils.inference import parse_polygon_steps
from app_utils.polygon_utils import _apply_polygon_steps


def test_parse_polygon_steps_supports_polygon_to_linestring():
    steps = parse_polygon_steps("polygon_to_linestring:1:1.0:2.0")
    assert len(steps) == 1
    assert steps[0]["name"] == "polygon_to_linestring"
    assert steps[0]["min_aspect"] == 2.0


def test_polygon_to_linestring_returns_multipoint_line_for_elongated_polygon():
    # 狹長矩形（順時針，不重複結尾）
    seg = np.array(
        [
            [0.0, 0.0],
            [12.0, 0.0],
            [12.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    steps = [{"name": "polygon_to_linestring", "count": 1, "min_aspect": 2.0}]
    out = _apply_polygon_steps(seg, steps, class_id=0)
    assert out.shape[0] > 2
    assert not np.allclose(out[0], out[-1])
