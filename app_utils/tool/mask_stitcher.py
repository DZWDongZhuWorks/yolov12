"""跨切片 mask 縫合（mask 層級去重）。

流程：各切片推論後只抽出「原始 instance mask」（不做任何優化），
以稀疏形式（全域 bbox 原點 + 裁切後 binary mask）收集；全片完成後，
同類別且像素相交的 instance 以 union-find 合併 —— 跨切片被切斷的物件
自然癒合、重疊區的重複偵測自然合併 —— 之後才對合併後的完整 mask
跑 mask 優化與 polygon 優化鏈（pca 等跨物件步驟因此以整張圖幅為單位運作）。

instance dict 欄位：
    cls_id: int / conf: float / x0, y0: 全域像素原點 / mask: uint8 2D（裁切）/
    tile_seqs: List[int] 來源切片序號
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

Instance = Dict[str, Any]


def extract_instances_from_result(
    result,
    tile_x: int,
    tile_y: int,
    tile_seq: int = 0,
    allowed_class_ids: Optional[Sequence[int]] = None,
) -> List[Instance]:
    """從單一 YOLO result 抽出稀疏全域 instance（原始 mask，不做優化）。

    masks.data 尺寸與 orig_shape 不同時（letterbox / proto 尺寸），
    以 ultralytics ops.scale_image 還原回切片像素空間。
    """
    if getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return []
    if result.masks.data is None or len(result.boxes) == 0:
        return []

    masks = result.masks.data
    masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None]

    orig_h, orig_w = int(result.orig_shape[0]), int(result.orig_shape[1])
    if masks_np.shape[1:] != (orig_h, orig_w):
        from ultralytics.utils import ops as _ultra_ops

        stacked = _ultra_ops.scale_image(masks_np.transpose(1, 2, 0), (orig_h, orig_w))
        masks_np = np.asarray(stacked).transpose(2, 0, 1)

    cls_arr = result.boxes.cls.cpu().numpy().astype(int)
    try:
        conf_arr = result.boxes.conf.cpu().numpy()
    except Exception:
        conf_arr = None

    allowed = set(int(c) for c in allowed_class_ids) if allowed_class_ids is not None else None
    instances: List[Instance] = []
    for i, m in enumerate(masks_np):
        cid = int(cls_arr[i]) if i < len(cls_arr) else -1
        if allowed is not None and cid not in allowed:
            continue
        binary = (m > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(binary)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        instances.append(
            {
                "cls_id": cid,
                "conf": float(conf_arr[i]) if conf_arr is not None and i < len(conf_arr) else 0.0,
                "x0": tile_x + x1,
                "y0": tile_y + y1,
                "mask": binary[y1:y2, x1:x2].copy(),
                "tile_seqs": [tile_seq],
            }
        )
    return instances


def _bbox(inst: Instance) -> Tuple[int, int, int, int]:
    h, w = inst["mask"].shape[:2]
    return inst["x0"], inst["y0"], inst["x0"] + w, inst["y0"] + h


def _paste(canvas: np.ndarray, cx0: int, cy0: int, inst: Instance) -> None:
    x1, y1, x2, y2 = _bbox(inst)
    np.logical_or(
        canvas[y1 - cy0 : y2 - cy0, x1 - cx0 : x2 - cx0],
        inst["mask"],
        out=canvas[y1 - cy0 : y2 - cy0, x1 - cx0 : x2 - cx0],
    )


def _masks_intersect(a: Instance, b: Instance, gap: int) -> bool:
    """兩個稀疏 mask 是否像素相交（容差 gap px：對交集視窗做 dilate）。"""
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)
    x1 = max(ax1 - gap, bx1 - gap)
    y1 = max(ay1 - gap, by1 - gap)
    x2 = min(ax2 + gap, bx2 + gap)
    y2 = min(ay2 + gap, by2 + gap)
    if x1 >= x2 or y1 >= y2:
        return False

    def _window(inst: Instance) -> np.ndarray:
        ix1, iy1, ix2, iy2 = _bbox(inst)
        win = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        sx1, sy1 = max(x1, ix1), max(y1, iy1)
        sx2, sy2 = min(x2, ix2), min(y2, iy2)
        if sx1 < sx2 and sy1 < sy2:
            win[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = inst["mask"][
                sy1 - iy1 : sy2 - iy1, sx1 - ix1 : sx2 - ix1
            ]
        return win

    win_a = _window(a)
    if gap > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * gap + 1, 2 * gap + 1))
        win_a = cv2.dilate(win_a, kernel, iterations=1)
    return bool(np.logical_and(win_a, _window(b)).any())


def merge_instances(instances: List[Instance], gap: int = 1) -> List[Instance]:
    """同類別且像素相交（含 gap px 容差）的 instance 以 union-find 合併。

    合併後 conf 取最大值、tile_seqs 取聯集；不同類別互不影響。
    """
    parent = list(range(len(instances)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_class: Dict[int, List[int]] = {}
    for idx, inst in enumerate(instances):
        by_class.setdefault(int(inst["cls_id"]), []).append(idx)

    for indices in by_class.values():
        for pos, i in enumerate(indices):
            bi = _bbox(instances[i])
            for j in indices[pos + 1 :]:
                if find(i) == find(j):
                    continue
                bj = _bbox(instances[j])
                if (
                    bi[0] - gap >= bj[2]
                    or bj[0] - gap >= bi[2]
                    or bi[1] - gap >= bj[3]
                    or bj[1] - gap >= bi[3]
                ):
                    continue
                if _masks_intersect(instances[i], instances[j], gap):
                    parent[find(j)] = find(i)

    groups: Dict[int, List[Instance]] = {}
    for idx, inst in enumerate(instances):
        groups.setdefault(find(idx), []).append(inst)

    merged: List[Instance] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        x1 = min(_bbox(m)[0] for m in members)
        y1 = min(_bbox(m)[1] for m in members)
        x2 = max(_bbox(m)[2] for m in members)
        y2 = max(_bbox(m)[3] for m in members)
        canvas = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        for m in members:
            _paste(canvas, x1, y1, m)
        merged.append(
            {
                "cls_id": int(members[0]["cls_id"]),
                "conf": max(float(m.get("conf", 0.0)) for m in members),
                "x0": x1,
                "y0": y1,
                "mask": canvas,
                "tile_seqs": sorted({s for m in members for s in m.get("tile_seqs", [])}),
            }
        )
    return merged


def instances_to_objects(
    instances: List[Instance],
    names: Dict[int, str],
    mask_steps: Optional[List[Dict[str, Any]]] = None,
    polygon_steps: Optional[List[Dict[str, Any]]] = None,
    simplify_mode: str = "none",
    simplify_eps_coeff: float = 1.0,
    subpixel_contour: bool = False,
    subpixel_scale: int = 3,
    pad: int = 32,
) -> List[Dict[str, Any]]:
    """合併後 instance -> mask 優化 -> 輪廓抽取（全域座標）-> polygon 優化鏈。

    產出 schema 與 build_objects_from_result 相同（class_id / class_name /
    confidence / bbox_xyxy / polygons / holes [+ line_strings]），另附 tile_seqs。
    座標為原始大圖全域像素，可直接餵 convert_yolo_json_to_geojson(crop=0,0)。

    pad：mask 優化前在稀疏畫布四周補的邊界，避免 dilate/blur 貼邊被裁。
    限制：mask 優化的 "merge" 步驟只在單一 instance 拆出的元件間有效，
    不會跨越縫合後的不同 instance（縫合階段已完成像素相交合併）。
    """
    from app_utils.inference_optimizations import (
        apply_mask_steps_to_instances,
        resolve_mask_step_filters,
    )
    from app_utils.polygon_utils import (
        _apply_ordered_polygon_steps,
        _close_ring,
        _mask_to_ring_groups,
        _mask_to_ring_groups_subpixel,
        _simplify_segment,
        resolve_polygon_step_filters,
    )

    resolved_mask_steps = resolve_mask_step_filters(mask_steps, names) if mask_steps else None

    objects: List[Dict[str, Any]] = []
    for inst in instances:
        canvas = np.pad(inst["mask"], pad)
        origin_x = int(inst["x0"]) - pad
        origin_y = int(inst["y0"]) - pad
        work_items: List[Dict[str, Any]] = [
            {
                "binary": canvas,
                "cls_id": int(inst["cls_id"]),
                "conf": float(inst.get("conf", 0.0)),
                "source_idx": 0,
                "source_indices": [0],
            }
        ]
        if resolved_mask_steps:
            work_items, _ = apply_mask_steps_to_instances(work_items, resolved_mask_steps)

        for item in work_items:
            binary = item["binary"]
            ys, xs = np.nonzero(binary)
            if xs.size == 0:
                continue

            if subpixel_contour:
                ring_groups = _mask_to_ring_groups_subpixel(binary.astype(np.float32), subpixel_scale)
            else:
                ring_groups = _mask_to_ring_groups(binary)

            offset = np.array([origin_x, origin_y], dtype=np.float32)
            polys: List[List[List[float]]] = []
            holes: List[List[List[List[float]]]] = []
            for ext_ring, hole_rings in ring_groups:
                ext = _simplify_segment(ext_ring + offset, simplify_mode, simplify_eps_coeff)
                if ext.shape[0] < 3:
                    continue
                polys.append(_close_ring(ext.astype(float).tolist()))
                group_holes: List[List[List[float]]] = []
                for hole_ring in hole_rings:
                    hole = _simplify_segment(hole_ring + offset, simplify_mode, simplify_eps_coeff)
                    if hole.shape[0] >= 3:
                        group_holes.append(_close_ring(hole.astype(float).tolist()))
                holes.append(group_holes)
            if not polys:
                continue

            cid = int(item["cls_id"])
            objects.append(
                {
                    "class_id": cid,
                    "class_name": (names or {}).get(cid, str(cid)),
                    "confidence": float(item.get("conf", 0.0)),
                    "bbox_xyxy": [
                        float(origin_x + xs.min()),
                        float(origin_y + ys.min()),
                        float(origin_x + xs.max()),
                        float(origin_y + ys.max()),
                    ],
                    "polygons": polys,
                    "holes": holes,
                    "tile_seqs": list(inst.get("tile_seqs", [])),
                }
            )

    resolved_polygon_steps = resolve_polygon_step_filters(polygon_steps, names or {})
    _apply_ordered_polygon_steps(objects, resolved_polygon_steps)
    return objects
