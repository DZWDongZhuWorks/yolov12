import numpy as np

from app_utils.inference import _merge_instances_by_iou


def _rect_mask(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def test_merge_by_iou_avoids_transitive_chain_merge():
    # A 與 B 重疊，B 與 C 重疊，但 A 與 C 不重疊。
    # 期望不要鏈式合併成單一 instance。
    a = _rect_mask(32, 32, 2, 10, 10, 18)
    b = _rect_mask(32, 32, 6, 10, 14, 18)
    c = _rect_mask(32, 32, 10, 10, 18, 18)

    instances = [
        {"binary": a, "cls_id": 5, "source_idx": 0, "source_indices": [0], "conf": 0.9},
        {"binary": b, "cls_id": 5, "source_idx": 1, "source_indices": [1], "conf": 0.8},
        {"binary": c, "cls_id": 5, "source_idx": 2, "source_indices": [2], "conf": 0.7},
    ]

    merged = _merge_instances_by_iou(instances, iou_threshold=0.1)

    assert len(merged) == 2
    source_groups = sorted(tuple(item["source_indices"]) for item in merged)
    assert source_groups == [(0, 1), (2,)]
