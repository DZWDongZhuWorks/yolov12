from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
import torch
from ultralytics.utils import ops

DEFAULT_MORPH_KERNEL = 3
DEFAULT_BLUR_KERNEL = 3
DEFAULT_BLUR_THRESHOLD = 0.5
DEFAULT_MIN_COMPONENT_AREA = 0
DEFAULT_MAX_HOLE_AREA = 0
DEFAULT_POLYGON_EPS_COEFF = 1.0
DEFAULT_POLYGON_MIN_ASPECT = 0.0
DEFAULT_POLYGON_PCA_MIN_COSINE = 0.94


def _coerce_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _normalize_class_filter(raw_value) -> Optional[List[object]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple, set)):
        if len(raw_value) == 0:
            return []
        tokens = [t for t in raw_value if t not in (None, "")]
    else:
        text = str(raw_value).strip()
        if text.lower() in {"all", "*", "any"}:
            return None
        if not text:
            return []
        tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    if not tokens:
        return []

    normalized: List[object] = []
    for token in tokens:
        if isinstance(token, (int, np.integer)):
            normalized.append(int(token))
            continue
        token_str = str(token).strip()
        if ":" in token_str:
            prefix = token_str.split(":", 1)[0].strip()
            try:
                normalized.append(int(prefix))
                continue
            except Exception:
                pass
        try:
            normalized.append(int(token_str))
        except Exception:
            normalized.append(token_str)
    return normalized or None


def _resolve_class_filter(class_filter: Optional[List[object]], names: Dict[int, str]) -> Optional[Set[int]]:
    if class_filter is None:
        return None
    if len(class_filter) == 0:
        return set()
    name_to_id = {str(name).lower(): int(idx) for idx, name in (names or {}).items()}
    resolved: Set[int] = set()
    for token in class_filter:
        if isinstance(token, (int, np.integer)):
            resolved.add(int(token))
            continue
        token_str = str(token).strip()
        if not token_str:
            continue
        try:
            resolved.add(int(token_str))
            continue
        except Exception:
            pass
        matched = name_to_id.get(token_str.lower())
        if matched is not None:
            resolved.add(matched)
    return resolved or None


def _build_step(name: str, count: int, morph_kernel=3, blur_kernel=3, blur_threshold=0.5,
                min_component_area=0, max_hole_area=0, merge_iou_threshold=0.1, classes=None) -> Dict[str, Any]:
    return {
        "name": name,
        "count": max(1, int(count)),
        "morph_kernel": _coerce_int(morph_kernel, DEFAULT_MORPH_KERNEL),
        "blur_kernel": _coerce_int(blur_kernel, DEFAULT_BLUR_KERNEL),
        "blur_threshold": _coerce_float(blur_threshold, DEFAULT_BLUR_THRESHOLD),
        "min_component_area": _coerce_int(min_component_area, DEFAULT_MIN_COMPONENT_AREA),
        "max_hole_area": _coerce_int(max_hole_area, DEFAULT_MAX_HOLE_AREA),
        "merge_iou_threshold": min(1.0, max(0.0, _coerce_float(merge_iou_threshold, 0.1))),
        "classes": _normalize_class_filter(classes),
    }


def parse_mask_steps(steps_input) -> List[Dict[str, Any]]:
    if steps_input is None:
        return []
    if hasattr(steps_input, "empty"):
        if steps_input.empty:
            return []
    elif not steps_input:
        return []

    valid_names = {"erode", "dilate", "distance_erode", "distance_dilate", "blur", "remove_small", "fill_holes", "split", "merge"}
    steps: List[Dict[str, Any]] = []

    if isinstance(steps_input, str):
        raw_parts = [part.strip() for part in steps_input.replace("\n", ",").split(",") if part.strip()]
        for item in raw_parts:
            name, count_text = item.split(":", 1) if ":" in item else (item, "1")
            name = name.strip().lower().replace("contour_split", "split")
            count = _coerce_int(count_text.strip(), 1)
            if name in valid_names and count > 0:
                steps.append(_build_step(name=name, count=count))
        return steps

    iterable = steps_input
    if hasattr(steps_input, "values") and hasattr(steps_input, "tolist"):
        try:
            iterable = steps_input.values.tolist()
        except Exception:
            pass

    try:
        for row in iterable:
            if not row or len(row) < 1:
                continue
            name = str(row[0]).strip().lower().replace("contour_split", "split")
            if name not in valid_names:
                continue
            enabled = True
            count_index = 1
            if len(row) > 1 and isinstance(row[1], (bool, np.bool_)):
                enabled = bool(row[1])
                count_index = 2
            if not enabled:
                continue
            count = _coerce_int(row[count_index] if len(row) > count_index else None, 1)
            if count <= 0:
                continue
            steps.append(_build_step(
                name=name,
                count=count,
                morph_kernel=row[count_index + 1] if len(row) > count_index + 1 else DEFAULT_MORPH_KERNEL,
                blur_kernel=row[count_index + 2] if len(row) > count_index + 2 else DEFAULT_BLUR_KERNEL,
                blur_threshold=row[count_index + 3] if len(row) > count_index + 3 else DEFAULT_BLUR_THRESHOLD,
                min_component_area=row[count_index + 4] if len(row) > count_index + 4 else DEFAULT_MIN_COMPONENT_AREA,
                max_hole_area=row[count_index + 5] if len(row) > count_index + 5 else DEFAULT_MAX_HOLE_AREA,
                merge_iou_threshold=row[count_index + 6] if len(row) > count_index + 6 else 0.1,
                classes=row[count_index + 7] if len(row) > count_index + 7 else None,
            ))
    except Exception:
        return []

    return steps


def _build_polygon_step(name: str, count: int, eps_coeff=1.0, min_aspect=0.0,
                        pca_min_cosine=0.94, pca_cross_class=False, classes=None) -> Dict[str, Any]:
    pca_cos = max(0.0, min(1.0, _coerce_float(pca_min_cosine, DEFAULT_POLYGON_PCA_MIN_COSINE)))
    return {
        "name": name,
        "count": max(1, int(count)),
        "eps_coeff": _coerce_float(eps_coeff, DEFAULT_POLYGON_EPS_COEFF),
        "min_aspect": max(0.0, _coerce_float(min_aspect, DEFAULT_POLYGON_MIN_ASPECT)),
        "pca_min_cosine": pca_cos,
        "pca_cross_class": bool(pca_cross_class),
        "classes": _normalize_class_filter(classes),
    }


def parse_polygon_steps(steps_input) -> List[Dict[str, Any]]:
    if steps_input is None:
        return []
    if hasattr(steps_input, "empty"):
        if steps_input.empty:
            return []
    elif not steps_input:
        return []

    valid_names = {
        "convex_hull",
        "rdp",
        "visvalingam_whyatt",
        "min_area_rect",
        "export_line",
        "pca",
        "polygon_to_lane_line",
        "polygon_merge",
    }
    steps: List[Dict[str, Any]] = []

    if isinstance(steps_input, str):
        raw_parts = [part.strip() for part in steps_input.replace("\n", ",").split(",") if part.strip()]
        for item in raw_parts:
            chunks = [p.strip() for p in item.split(":") if p.strip()]
            if not chunks:
                continue
            name = chunks[0].lower()
            if name not in valid_names:
                continue
            count = _coerce_int(chunks[1] if len(chunks) > 1 else 1, 1)
            if count <= 0:
                continue
            steps.append(_build_polygon_step(
                name=name,
                count=count,
                eps_coeff=_coerce_float(chunks[2] if len(chunks) > 2 else DEFAULT_POLYGON_EPS_COEFF, DEFAULT_POLYGON_EPS_COEFF),
                min_aspect=_coerce_float(chunks[3] if len(chunks) > 3 else DEFAULT_POLYGON_MIN_ASPECT, DEFAULT_POLYGON_MIN_ASPECT),
                pca_min_cosine=_coerce_float(chunks[4] if len(chunks) > 4 else DEFAULT_POLYGON_PCA_MIN_COSINE, DEFAULT_POLYGON_PCA_MIN_COSINE),
                pca_cross_class=str(chunks[5]).lower() in {"1", "true", "yes", "y", "on"} if len(chunks) > 5 else False,
            ))
        return steps

    iterable = steps_input
    if hasattr(steps_input, "values") and hasattr(steps_input, "tolist"):
        try:
            iterable = steps_input.values.tolist()
        except Exception:
            pass

    try:
        for row in iterable:
            if not row or len(row) < 1:
                continue
            name = str(row[0]).strip().lower()
            if name not in valid_names:
                continue
            enabled = True
            count_index = 1
            if len(row) > 1 and isinstance(row[1], (bool, np.bool_)):
                enabled = bool(row[1])
                count_index = 2
            if not enabled:
                continue

            count = _coerce_int(row[count_index] if len(row) > count_index else 1, 1)
            if count <= 0:
                continue
            eps_coeff = row[count_index + 1] if len(row) > count_index + 1 else DEFAULT_POLYGON_EPS_COEFF
            if len(row) > count_index + 5:
                min_aspect = row[count_index + 2]
                pca_min_cosine = row[count_index + 3]
                pca_cross_class = bool(row[count_index + 4]) if row[count_index + 4] is not None else False
                classes = row[count_index + 5]
            elif len(row) > count_index + 4:
                min_aspect = row[count_index + 2]
                pca_min_cosine = row[count_index + 3]
                pca_cross_class = False
                classes = row[count_index + 4]
            else:
                min_aspect = DEFAULT_POLYGON_MIN_ASPECT
                pca_min_cosine = DEFAULT_POLYGON_PCA_MIN_COSINE
                pca_cross_class = False
                classes = row[count_index + 2] if len(row) > count_index + 2 else None
            steps.append(_build_polygon_step(name, count, eps_coeff, min_aspect, pca_min_cosine, pca_cross_class, classes))
    except Exception:
        return []

    return steps


def _ensure_odd(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _split_components(binary: np.ndarray) -> List[np.ndarray]:
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) <= 1:
        return [binary]
    components: List[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        component = np.zeros_like(binary)
        cv2.drawContours(component, [contour], -1, 1, thickness=-1)
        if component.sum() > 0:
            components.append(component)
    return components or [binary]


def split_connection_contours(result):
    if not hasattr(result, "masks") or result.masks is None:
        return result
    if not hasattr(result, "boxes") or result.boxes is None:
        return result

    masks = result.masks.data
    boxes_data = result.boxes.data
    if masks is None or len(masks) == 0 or boxes_data is None or len(boxes_data) == 0:
        return result

    orig_shape = result.orig_shape
    mask_shape = masks.shape[1:]
    new_masks: List[np.ndarray] = []
    new_boxes: List[List[float]] = []
    had_split = False

    boxes_np = boxes_data.detach().cpu().numpy()
    is_track = result.boxes.is_track
    for i, mask_tensor in enumerate(masks):
        mask_np = mask_tensor.detach().cpu().numpy()
        binary = (mask_np > 0.5).astype(np.uint8)
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(contours) <= 1:
            new_masks.append(mask_np)
            new_boxes.append(boxes_np[i].tolist())
            continue
        had_split = True
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            component = np.zeros_like(binary)
            cv2.drawContours(component, [contour], -1, 1, thickness=-1)
            new_masks.append(component.astype(mask_np.dtype))
            coords = ops.scale_coords(mask_shape, contour.reshape(-1, 2).astype(np.float32), orig_shape, normalize=False)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            if is_track:
                track_id = float(boxes_np[i][4])
                conf = float(boxes_np[i][5])
                cls = float(boxes_np[i][6])
                new_boxes.append([x_min, y_min, x_max, y_max, track_id, conf, cls])
            else:
                conf = float(boxes_np[i][4])
                cls = float(boxes_np[i][5])
                new_boxes.append([x_min, y_min, x_max, y_max, conf, cls])

    if not new_boxes or not had_split:
        return result

    updated = result.new()
    updated.update(
        boxes=torch.tensor(new_boxes, device=boxes_data.device, dtype=boxes_data.dtype),
        masks=torch.tensor(np.stack(new_masks, axis=0), device=masks.device, dtype=masks.dtype),
    )
    updated.names = result.names
    updated.path = result.path
    return updated


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    output = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 1
    return output


def _distance_dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    return (cv2.distanceTransform((1 - binary).astype(np.uint8), cv2.DIST_L2, 3) <= radius).astype(np.uint8)


def _distance_erode(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    return (cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3) > radius).astype(np.uint8)


def _fill_small_holes(binary: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0:
        return binary
    h, w = binary.shape[:2]
    inverted = (1 - binary).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if num_labels <= 1:
        return binary
    output = binary.copy()
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if x == 0 or y == 0 or (x + width) == w or (y + height) == h:
            continue
        if area <= max_hole_area:
            output[labels == label] = 1
    return output


def _merge_instances_by_iou(instances: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    if not instances:
        return []
    pending = [dict(item) for item in instances]
    merged_instances: List[Dict[str, Any]] = []
    while pending:
        base = pending.pop(0)
        base_mask = (base["binary"] > 0).astype(np.uint8)
        base_sources = list(base.get("source_indices", [])) or [base.get("source_idx", 0)]
        base_conf = float(base.get("conf", 0.0))

        changed = True
        while changed:
            changed = False
            remained: List[Dict[str, Any]] = []
            for candidate in pending:
                cand_mask = (candidate["binary"] > 0).astype(np.uint8)
                inter = np.logical_and(base_mask > 0, cand_mask > 0).sum()
                if inter == 0:
                    remained.append(candidate)
                    continue
                union = np.logical_or(base_mask > 0, cand_mask > 0).sum()
                if union <= 0:
                    remained.append(candidate)
                    continue
                if inter / float(union) >= iou_threshold:
                    base_mask = np.logical_or(base_mask > 0, cand_mask > 0).astype(np.uint8)
                    base_sources.extend(list(candidate.get("source_indices", [])) or [candidate.get("source_idx", 0)])
                    base_conf = max(base_conf, float(candidate.get("conf", 0.0)))
                    changed = True
                else:
                    remained.append(candidate)
            pending = remained

        merged_instances.append({
            "binary": base_mask,
            "cls_id": int(base["cls_id"]),
            "source_idx": min(base_sources),
            "source_indices": sorted(set(base_sources)),
            "conf": base_conf,
        })
    return merged_instances


def apply_mask_optimizations_to_result(result, enabled: bool, steps: List[Dict[str, Any]]):
    if not enabled or not steps:
        return result
    if not hasattr(result, "masks") or result.masks is None:
        return result
    if not hasattr(result, "boxes") or result.boxes is None:
        return result

    masks = result.masks.data
    boxes_data = result.boxes.data
    if masks is None or len(masks) == 0 or boxes_data is None or len(boxes_data) == 0:
        return result

    orig_shape = result.orig_shape
    mask_shape = masks.shape[1:]
    boxes_np = boxes_data.detach().cpu().numpy()
    is_track = result.boxes.is_track
    names = getattr(result, "names", {}) or {}

    resolved_steps = [{**step, "class_filter": _resolve_class_filter(step.get("classes"), names)} for step in steps]
    instances: List[Dict[str, Any]] = []
    for i, mask_tensor in enumerate(masks):
        binary = (mask_tensor.detach().cpu().numpy() > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            continue
        cls_id = int(float(boxes_np[i][6] if is_track else boxes_np[i][5]))
        conf = float(boxes_np[i][5] if is_track else boxes_np[i][4])
        instances.append({"binary": binary, "cls_id": cls_id, "source_idx": i, "source_indices": [i], "conf": conf})

    for step in resolved_steps:
        step_name = step["name"]
        count = step["count"]
        class_filter = step.get("class_filter")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ensure_odd(step["morph_kernel"]),) * 2)
        distance_radius = max(1, _ensure_odd(step["morph_kernel"]) // 2)
        blur_size = _ensure_odd(step["blur_kernel"])
        blur_threshold = step["blur_threshold"]

        if step_name == "merge":
            for _ in range(count):
                untouched: List[Dict[str, Any]] = []
                grouped: Dict[int, List[Dict[str, Any]]] = {}
                for item in instances:
                    if class_filter is not None and item["cls_id"] not in class_filter:
                        untouched.append(item)
                    else:
                        grouped.setdefault(item["cls_id"], []).append(item)
                instances = list(untouched)
                for cls_instances in grouped.values():
                    instances.extend(_merge_instances_by_iou(cls_instances, step["merge_iou_threshold"]))
            continue

        next_instances: List[Dict[str, Any]] = []
        for item in instances:
            if class_filter is not None and item["cls_id"] not in class_filter:
                next_instances.append(item)
                continue
            binaries = [item["binary"]]
            if step_name == "erode":
                for _ in range(count): binaries = [cv2.erode(b, kernel, iterations=1) for b in binaries]
            elif step_name == "dilate":
                for _ in range(count): binaries = [cv2.dilate(b, kernel, iterations=1) for b in binaries]
            elif step_name == "distance_erode":
                for _ in range(count): binaries = [_distance_erode(b, distance_radius) for b in binaries]
            elif step_name == "distance_dilate":
                for _ in range(count): binaries = [_distance_dilate(b, distance_radius) for b in binaries]
            elif step_name == "blur":
                for _ in range(count): binaries = [
                    (cv2.GaussianBlur(b.astype(np.float32), (blur_size, blur_size), 0) >= blur_threshold).astype(np.uint8)
                    for b in binaries
                ]
            elif step_name == "remove_small":
                for _ in range(count): binaries = [_remove_small_components(b, step["min_component_area"]) for b in binaries]
            elif step_name == "fill_holes":
                for _ in range(count): binaries = [_fill_small_holes(b, step["max_hole_area"]) for b in binaries]
            elif step_name == "split":
                for _ in range(count):
                    split_bins: List[np.ndarray] = []
                    for b in binaries: split_bins.extend(_split_components(b))
                    binaries = split_bins

            for binary in binaries:
                if binary.sum() > 0:
                    next_instances.append({
                        "binary": binary,
                        "cls_id": item["cls_id"],
                        "source_idx": item["source_idx"],
                        "source_indices": list(item.get("source_indices", [item["source_idx"]])),
                        "conf": item["conf"],
                    })
        instances = next_instances

    new_masks: List[np.ndarray] = []
    new_boxes: List[List[float]] = []
    for item in instances:
        optimized = item["binary"]
        ys, xs = np.where(optimized > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        coords = np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]], dtype=np.float32)
        coords = ops.scale_coords(mask_shape, coords, orig_shape, normalize=False)
        x_min, y_min = coords[0]
        x_max, y_max = coords[1]

        source_idx = int(item["source_idx"])
        cls_float = float(item["cls_id"])
        conf = float(item["conf"])
        if is_track:
            new_boxes.append([x_min, y_min, x_max, y_max, float(boxes_np[source_idx][4]), conf, cls_float])
        else:
            new_boxes.append([x_min, y_min, x_max, y_max, conf, cls_float])
        new_masks.append(optimized.astype(np.uint8))

    if not new_boxes:
        return result

    updated = result.new()
    updated.update(
        boxes=torch.tensor(new_boxes, device=boxes_data.device, dtype=boxes_data.dtype),
        masks=torch.tensor(np.stack(new_masks, axis=0), device=masks.device, dtype=masks.dtype),
    )
    updated.names = result.names
    updated.path = result.path
    return updated


def apply_mask_optimizations(results_cache: Dict[str, Any], enabled: bool, steps_input) -> Dict[str, Any]:
    steps = parse_mask_steps(steps_input)
    if not results_cache or not enabled or not steps:
        return results_cache
    updated_cache: Dict[str, Any] = {}
    for mid, results in results_cache.items():
        updated_cache[mid] = [apply_mask_optimizations_to_result(res, enabled, steps) for res in results] if results else results
    return updated_cache


def apply_connection_contour_split(results_cache: Dict[str, Any]) -> Dict[str, Any]:
    if not results_cache:
        return results_cache
    return {
        mid: [split_connection_contours(res) for res in results] if results else results
        for mid, results in results_cache.items()
    }
