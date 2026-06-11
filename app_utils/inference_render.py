import copy
from typing import List, Optional

import cv2
import numpy as np
from ultralytics.utils.plotting import colors as ucolors

from .polygon_utils import build_objects_from_result


def filter_result_by_classes(result, allowed_class_ids: Optional[List[int]]):
    """Return shallow-copied result filtered by class IDs."""
    if allowed_class_ids is None:
        return result

    r = copy.copy(result)
    keep_idx: List[int] = []

    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        cls_arr = result.boxes.cls.cpu().numpy().astype(int)
        allow = set(allowed_class_ids)
        keep_idx = [i for i, cid in enumerate(cls_arr) if cid in allow]
        r.boxes = result.boxes[keep_idx] if len(keep_idx) > 0 else None
    else:
        r.boxes = None

    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        cls_arr = result.obb.cls.cpu().numpy().astype(int)
        allow = set(allowed_class_ids)
        keep_idx = [i for i, cid in enumerate(cls_arr) if cid in allow]
        r.obb = result.obb[keep_idx] if len(keep_idx) > 0 else None
    else:
        r.obb = None

    if hasattr(result, "masks") and result.masks is not None:
        r.masks = result.masks[keep_idx] if len(keep_idx) > 0 else None
    else:
        r.masks = None

    return r


def annotate_from_results(
    result,
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_points: bool,
    show_confidence: bool,
    simplify_mode: str,
    simplify_eps_coeff: float,
    allowed_class_ids: Optional[List[int]] = None,
    polygon_opt_steps=None,
):
    """Render model result with optional class filtering, polygons, and labels."""
    filtered = filter_result_by_classes(result, allowed_class_ids)
    base = filtered.plot(labels=False, boxes=show_boxes, masks=show_masks)

    if (show_polygons or show_points) and getattr(filtered, "masks", None) is not None:
        objects = build_objects_from_result(
            result,
            allowed_class_ids=allowed_class_ids,
            simplify_mode=simplify_mode,
            simplify_eps_coeff=simplify_eps_coeff,
            polygon_opt_steps=polygon_opt_steps,
        )

        for obj in objects:
            cid = obj["class_id"]
            color = tuple(int(v) for v in ucolors(cid, bgr=True))
            # 洞（內環）邊界與外環同色一併繪製
            hole_rings = [h for group in (obj.get("holes") or []) for h in group]
            for seg in list(obj["polygons"]) + hole_rings:
                seg_arr = np.asarray(seg, dtype=np.float32)
                if seg_arr.ndim != 2 or seg_arr.shape[0] < 2:
                    continue
                pts = seg_arr.astype(np.int32).reshape(-1, 1, 2)
                is_closed = seg_arr.shape[0] >= 3 and np.allclose(seg_arr[0], seg_arr[-1])
                if show_polygons:
                    cv2.polylines(base, [pts], isClosed=is_closed, color=color, thickness=2)
                if show_points:
                    for x, y in seg_arr.astype(np.int32):
                        cv2.circle(base, (int(x), int(y)), radius=3, color=(255, 255, 255), thickness=1)

    if label_mode == "隱藏":
        return base

    has_boxes = hasattr(filtered, "boxes") and filtered.boxes is not None and len(filtered.boxes) > 0
    has_obb = hasattr(filtered, "obb") and filtered.obb is not None and len(filtered.obb) > 0
    if not has_boxes and not has_obb:
        return base

    names = getattr(result, "names", None) or {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    pad = 3

    def _draw_label(x1: int, y1: int, cid: int, conf_val):
        text = f"{cid}" if label_mode == "顯示 class id" else names.get(cid, str(cid))
        if show_confidence and conf_val is not None:
            text = f"{text} {conf_val:.2f}"

        c_bgr = tuple(int(v) for v in ucolors(cid, bgr=True))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        y_top = max(0, y1 - th - 2 * pad)
        cv2.rectangle(base, (x1, y_top), (x1 + tw + 2 * pad, y1), c_bgr, -1)
        b, g, r = c_bgr
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
        cv2.putText(base, text, (x1 + pad, y1 - pad), font, font_scale, text_color, thickness, cv2.LINE_AA)

    if has_boxes:
        xyxy = filtered.boxes.xyxy.cpu().numpy()
        cls_arr = filtered.boxes.cls.cpu().numpy().astype(int)
        conf_arr = None
        try:
            conf_arr = filtered.boxes.conf.cpu().numpy()
        except Exception:
            pass

        for i, ((x1, y1, _, _), cid) in enumerate(zip(xyxy, cls_arr)):
            conf_val = conf_arr[i] if conf_arr is not None and i < len(conf_arr) else None
            _draw_label(int(x1), int(y1), cid, conf_val)

    if has_obb:
        obb_xyxy = filtered.obb.xyxy.cpu().numpy()
        cls_arr = filtered.obb.cls.cpu().numpy().astype(int)
        conf_arr = None
        try:
            conf_arr = filtered.obb.conf.cpu().numpy()
        except Exception:
            pass

        for i, ((x1, y1, _, _), cid) in enumerate(zip(obb_xyxy, cls_arr)):
            conf_val = conf_arr[i] if conf_arr is not None and i < len(conf_arr) else None
            _draw_label(int(x1), int(y1), cid, conf_val)

    return base
