# app_utils/inference.py

import copy
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.plotting import colors as ucolors
from .polygon_utils import build_objects_from_result

# 若你之後有做 app_utils/config.py，就會用到這個；沒做就 fallback = 5
try:
    from .config import MAX_MODELS
except Exception:
    MAX_MODELS = 5


# -----------------------------
# 類別選單 / 類別解析
# -----------------------------
def names_to_choice_list(names: Dict[Any, str]) -> Tuple[List[str], List[int]]:
    """
    將 {id: name} 轉成
      choices: ["0: person", "1: bicycle", ...]
      ids    : [0, 1, ...]
    並按照 class id 排序。
    """
    pairs: List[Tuple[int, str]] = []
    for k, v in (names or {}).items():
        try:
            cid = int(k)
        except Exception:
            cid = int(str(k))
        pairs.append((cid, v))

    pairs.sort(key=lambda x: x[0])
    choices = [f"{cid}: {name}" for cid, name in pairs]
    ids = [cid for cid, _ in pairs]
    return choices, ids


def parse_selected_to_ids(selected_items: Optional[List[str]]) -> List[int]:
    """
    將 CheckboxGroup 回傳的值
      ["0: person", "2: car"] -> [0, 2]
    若為 None 或 []，回傳 []。
    """
    if not selected_items:
        return []

    out: List[int] = []
    for s in selected_items:
        try:
            cid = int(str(s).split(":")[0].strip())
            out.append(cid)
        except Exception:
            pass
    return out


# -----------------------------
# 結果過濾 / 繪圖
# -----------------------------
def filter_result_by_classes(result, allowed_class_ids: Optional[List[int]]):
    """
    回傳一個淺複製（shallow copy）的 result，只保留指定類別的 boxes/masks。

    allowed_class_ids:
      - None : 不過濾（顯示全部）
      - []   : 全部隱藏
      - [id, id, ...] : 僅顯示指定類別
    """
    if allowed_class_ids is None:
        # 不過濾
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

    # masks 與 boxes 順序一致，沿用 keep_idx
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
):
    """
    使用 Ultralytics 的 plot 畫基礎層（bbox / mask），
    再由我們自行畫：
      - polygon 邊線（可選）
      - label（class id / class name + conf）

    回傳 BGR 影像。
    """
    # 先做類別篩選
    filtered = filter_result_by_classes(result, allowed_class_ids)

    # Ultralytics 內建繪製框 / mask（labels 關掉，自己畫）
    base = filtered.plot(labels=False, boxes=show_boxes, masks=show_masks)
    
    objects = build_objects_from_result(
        result,
        allowed_class_ids=allowed_class_ids,
        simplify_mode=simplify_mode,
        simplify_eps_coeff=simplify_eps_coeff,
    )
    
    # ---- 先畫 polygon 邊界（如果有 mask） ----
    if show_polygons and getattr(filtered, "masks", None) is not None:
        for obj in objects:
            cid = obj["class_id"]
            polys = obj["polygons"]
            color = tuple(int(v) for v in ucolors(cid, bgr=True))

            for seg in polys:
                pts = np.asarray(seg, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(base, [pts], isClosed=True, color=color, thickness=2)
                if show_points:
                    for x, y in np.asarray(seg, dtype=np.int32):
                        cv2.circle(base, (int(x), int(y)), radius=3, color=(255, 255, 255), thickness=1)
    # ---- 再畫 label ----
    if label_mode == "隱藏":
        return base
    if not hasattr(filtered, "boxes") or filtered.boxes is None or len(filtered.boxes) == 0:
        return base

    names = getattr(result, "names", None) or {}
    xyxy = filtered.boxes.xyxy.cpu().numpy()
    cls_arr = filtered.boxes.cls.cpu().numpy().astype(int)

    # 可能沒有 conf，保險起見
    conf_arr = None
    try:
        conf_arr = filtered.boxes.conf.cpu().numpy()
    except Exception:
        pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    pad = 3

    for i, ((x1, y1, x2, y2), cid) in enumerate(zip(xyxy, cls_arr)):
        x1, y1 = int(x1), int(y1)

        # 基本標籤（ID 或 Name）
        base_text = f"{cid}" if label_mode == "顯示 class id" else names.get(cid, str(cid))

        # 需要的話加上 conf
        if show_confidence and conf_arr is not None and i < len(conf_arr):
            base_text = f"{base_text} {conf_arr[i]:.2f}"

        c_bgr = tuple(int(v) for v in ucolors(cid, bgr=True))
        (tw, th), baseline = cv2.getTextSize(base_text, font, font_scale, thickness)
        y_top = max(0, y1 - th - 2 * pad)

        cv2.rectangle(base, (x1, y_top), (x1 + tw + 2 * pad, y1), c_bgr, -1)

        # --- 根據背景亮度自動選字色 ---
        b, g, r = c_bgr
        luminance = 0.299*r + 0.587*g + 0.114*b
        text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
        cv2.putText(
            base,
            base_text,
            (x1 + pad, y1 - pad),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return base


def split_connection_contours(result):
    """
    將單一 result 的 masks 拆成連通元件，並以拆分後的 contour 取代原本結果。
    若 mask 沒有多段，則維持原樣。
    """
    if not hasattr(result, "masks") or result.masks is None:
        return result
    if not hasattr(result, "boxes") or result.boxes is None:
        return result

    masks = result.masks.data
    if masks is None or len(masks) == 0:
        return result

    boxes_data = result.boxes.data
    if boxes_data is None or len(boxes_data) == 0:
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

            coords = contour.reshape(-1, 2).astype(np.float32)
            coords = ops.scale_coords(mask_shape, coords, orig_shape, normalize=False)
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

    if not new_boxes:
        return result

    if not had_split:
        return result

    new_boxes_tensor = torch.tensor(new_boxes, device=boxes_data.device, dtype=boxes_data.dtype)
    new_masks_tensor = torch.tensor(np.stack(new_masks, axis=0), device=masks.device, dtype=masks.dtype)

    updated = result.new()
    updated.update(boxes=new_boxes_tensor, masks=new_masks_tensor)
    updated.names = result.names
    updated.path = result.path
    return updated


DEFAULT_MORPH_KERNEL = 3
DEFAULT_BLUR_KERNEL = 3
DEFAULT_BLUR_THRESHOLD = 0.5
DEFAULT_MIN_COMPONENT_AREA = 0
DEFAULT_MAX_HOLE_AREA = 0


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
        tokens = [t for t in raw_value if t not in (None, "")]
    else:
        text = str(raw_value).strip()
        if not text or text.lower() in {"all", "*", "any"}:
            return None
        tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    if not tokens:
        return None
    normalized: List[object] = []
    for token in tokens:
        if isinstance(token, (int, np.integer)):
            normalized.append(int(token))
            continue
        try:
            normalized.append(int(str(token)))
        except Exception:
            normalized.append(str(token))
    return normalized or None


def _resolve_class_filter(class_filter: Optional[List[object]], names: Dict[int, str]) -> Optional[Set[int]]:
    if not class_filter:
        return None
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


def _build_step(
    name: str,
    count: int,
    morph_kernel: int = DEFAULT_MORPH_KERNEL,
    blur_kernel: int = DEFAULT_BLUR_KERNEL,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
    max_hole_area: int = DEFAULT_MAX_HOLE_AREA,
    classes=None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "count": max(1, int(count)),
        "morph_kernel": _coerce_int(morph_kernel, DEFAULT_MORPH_KERNEL),
        "blur_kernel": _coerce_int(blur_kernel, DEFAULT_BLUR_KERNEL),
        "blur_threshold": _coerce_float(blur_threshold, DEFAULT_BLUR_THRESHOLD),
        "min_component_area": _coerce_int(min_component_area, DEFAULT_MIN_COMPONENT_AREA),
        "max_hole_area": _coerce_int(max_hole_area, DEFAULT_MAX_HOLE_AREA),
        "classes": _normalize_class_filter(classes),
    }


def parse_mask_steps(steps_input) -> List[Dict[str, Any]]:
    if steps_input is None:
        return []
    
    # Check for empty state safely (handles pandas DataFrames)
    if hasattr(steps_input, "empty"):
        if steps_input.empty:
            return []
    elif not steps_input:
        return []

    steps: List[Dict[str, Any]] = []
    if isinstance(steps_input, str):
        raw_parts = []
        for part in steps_input.replace("\n", ",").split(","):
            cleaned = part.strip()
            if cleaned:
                raw_parts.append(cleaned)
        for item in raw_parts:
            if ":" in item:
                name, count_text = item.split(":", 1)
            else:
                name, count_text = item, "1"
            name = name.strip().lower()
            try:
                count = int(count_text.strip())
            except Exception:
                count = 1
            if count <= 0:
                continue
            if name == "contour_split":
                name = "split"
            if name not in {
                "erode",
                "dilate",
                "distance_erode",
                "distance_dilate",
                "blur",
                "remove_small",
                "fill_holes",
                "split",
            }:
                continue
            steps.append(_build_step(name=name, count=count))
    else:
        # Handle DataFrame or list of lists
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
                count = _coerce_int(row[1] if len(row) > 1 else None, 1)
                if count <= 0:
                    continue
                if name == "contour_split":
                    name = "split"
                if name not in {
                    "erode",
                    "dilate",
                    "distance_erode",
                    "distance_dilate",
                    "blur",
                    "remove_small",
                    "fill_holes",
                    "split",
                }:
                    continue
                morph_kernel = row[2] if len(row) > 2 else DEFAULT_MORPH_KERNEL
                blur_kernel = row[3] if len(row) > 3 else DEFAULT_BLUR_KERNEL
                blur_threshold = row[4] if len(row) > 4 else DEFAULT_BLUR_THRESHOLD
                min_component_area = row[5] if len(row) > 5 else DEFAULT_MIN_COMPONENT_AREA
                max_hole_area = row[6] if len(row) > 6 else DEFAULT_MAX_HOLE_AREA
                classes = row[7] if len(row) > 7 else None
                steps.append(
                    _build_step(
                        name=name,
                        count=count,
                        morph_kernel=morph_kernel,
                        blur_kernel=blur_kernel,
                        blur_threshold=blur_threshold,
                        min_component_area=min_component_area,
                        max_hole_area=max_hole_area,
                        classes=classes,
                    )
                )
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


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    output = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = 1
    return output


def _distance_dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    inverted = (1 - binary).astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
    return (dist <= radius).astype(np.uint8)


def _distance_erode(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    return (dist > radius).astype(np.uint8)


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
        touches_border = x == 0 or y == 0 or (x + width) == w or (y + height) == h
        if touches_border:
            continue
        if area <= max_hole_area:
            output[labels == label] = 1
    return output


def apply_mask_optimizations_to_result(
    result,
    enabled: bool,
    steps: List[Dict[str, Any]],
):
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
    new_masks: List[np.ndarray] = []
    new_boxes: List[List[float]] = []

    boxes_np = boxes_data.detach().cpu().numpy()
    is_track = result.boxes.is_track
    names = getattr(result, "names", {}) or {}
    resolved_steps: List[Dict[str, Any]] = []
    for step in steps:
        resolved_steps.append(
            {
                **step,
                "class_filter": _resolve_class_filter(step.get("classes"), names),
            }
        )

    for i, mask_tensor in enumerate(masks):
        mask_np = mask_tensor.detach().cpu().numpy()
        binaries = [(mask_np > 0.5).astype(np.uint8)]
        cls_value = float(boxes_np[i][6] if is_track else boxes_np[i][5])
        cls_id = int(cls_value)

        for step in resolved_steps:
            step_name = step["name"]
            count = step["count"]
            class_filter = step.get("class_filter")
            if class_filter is not None and cls_id not in class_filter:
                continue
            kernel_size = _ensure_odd(step["morph_kernel"])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            distance_radius = max(1, kernel_size // 2)
            blur_size = _ensure_odd(step["blur_kernel"])
            blur_threshold = step["blur_threshold"]
            min_component_area = step["min_component_area"]
            max_hole_area = step["max_hole_area"]
            if not binaries:
                break
            if step_name == "erode":
                for _ in range(count):
                    binaries = [cv2.erode(b, kernel, iterations=1) for b in binaries]
            elif step_name == "dilate":
                for _ in range(count):
                    binaries = [cv2.dilate(b, kernel, iterations=1) for b in binaries]
            elif step_name == "distance_erode":
                for _ in range(count):
                    binaries = [_distance_erode(b, distance_radius) for b in binaries]
            elif step_name == "distance_dilate":
                for _ in range(count):
                    binaries = [_distance_dilate(b, distance_radius) for b in binaries]
            elif step_name == "blur":
                for _ in range(count):
                    blurred = [
                        cv2.GaussianBlur(b.astype(np.float32), (blur_size, blur_size), 0)
                        for b in binaries
                    ]
                    binaries = [(b >= blur_threshold).astype(np.uint8) for b in blurred]
            elif step_name == "remove_small":
                for _ in range(count):
                    binaries = [_remove_small_components(b, min_component_area) for b in binaries]
            elif step_name == "fill_holes":
                for _ in range(count):
                    binaries = [_fill_small_holes(b, max_hole_area) for b in binaries]
            elif step_name == "split":
                for _ in range(count):
                    split_bins: List[np.ndarray] = []
                    for b in binaries:
                        split_bins.extend(_split_components(b))
                    binaries = split_bins

        for optimized in binaries:
            if optimized.sum() == 0:
                continue

            ys, xs = np.where(optimized > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            coords = np.array([[x_min, y_min], [x_max, y_max]], dtype=np.float32)
            coords = ops.scale_coords(mask_shape, coords, orig_shape, normalize=False)
            x_min, y_min = coords[0]
            x_max, y_max = coords[1]

            if is_track:
                track_id = float(boxes_np[i][4])
                conf = float(boxes_np[i][5])
                cls = float(boxes_np[i][6])
                new_boxes.append([x_min, y_min, x_max, y_max, track_id, conf, cls])
            else:
                conf = float(boxes_np[i][4])
                cls = float(boxes_np[i][5])
                new_boxes.append([x_min, y_min, x_max, y_max, conf, cls])

            new_masks.append(optimized.astype(mask_np.dtype))

    if not new_boxes:
        return result

    new_boxes_tensor = torch.tensor(new_boxes, device=boxes_data.device, dtype=boxes_data.dtype)
    new_masks_tensor = torch.tensor(np.stack(new_masks, axis=0), device=masks.device, dtype=masks.dtype)

    updated = result.new()
    updated.update(boxes=new_boxes_tensor, masks=new_masks_tensor)
    updated.names = result.names
    updated.path = result.path
    return updated


def apply_mask_optimizations(
    results_cache: Dict[str, Any],
    enabled: bool,
    steps_input,
) -> Dict[str, Any]:
    steps = parse_mask_steps(steps_input)
    if not results_cache or not enabled or not steps:
        return results_cache

    updated_cache: Dict[str, Any] = {}
    for mid, results in results_cache.items():
        if not results:
            updated_cache[mid] = results
            continue
        updated_cache[mid] = [
            apply_mask_optimizations_to_result(
                res,
                enabled,
                steps,
            )
            for res in results
        ]

    return updated_cache


def apply_connection_contour_split(results_cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    對 results_cache 內的結果做連通元件拆分，維持結果型態供後續流程使用。
    """
    if not results_cache:
        return results_cache

    updated_cache: Dict[str, Any] = {}
    for mid, results in results_cache.items():
        if not results:
            updated_cache[mid] = results
            continue
        updated_cache[mid] = [split_connection_contours(res) for res in results]

    return updated_cache


# -----------------------------
# 單模型推論（圖 / 影） + 多模型封裝
# -----------------------------
def infer_image_single(
    model_id: str,
    image,
    image_size: int,
    conf_threshold: float,
    device: Optional[str],
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_points: bool,
    show_confidence: bool,
    simplify_mode: str,
    simplify_eps_coeff: float,
    allowed_class_ids: Optional[List[int]],
):
    model = YOLO(model_id)
    predict_kwargs = {"source": image, "imgsz": image_size, "conf": conf_threshold}
    if device:
        predict_kwargs["device"] = device
    results = model.predict(**predict_kwargs)
    annotated_bgr = annotate_from_results(
        results[0],
        label_mode,
        show_boxes,
        show_masks,
        show_polygons,
        show_points,
        show_confidence,
        simplify_mode,
        simplify_eps_coeff,
        allowed_class_ids,
    )
    # Gradio Image 用 RGB
    return annotated_bgr[:, :, ::-1], results  # (RGB, results)


def infer_video_single(
    model_id: str,
    video_path: str,
    image_size: int,
    conf_threshold: float,
    device: Optional[str],
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_points: bool,
    show_confidence: bool,
    simplify_mode: str,
    simplify_eps_coeff: float,
    allowed_class_ids: Optional[List[int]],
    mask_opt_enabled: bool,
    mask_opt_steps,
):
    model = YOLO(model_id)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".webm")
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"vp80"),
        fps,
        (frame_width, frame_height),
    )
    mask_steps = parse_mask_steps(mask_opt_steps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        predict_kwargs = {"source": frame, "imgsz": image_size, "conf": conf_threshold}
        if device:
            predict_kwargs["device"] = device
        results = model.predict(**predict_kwargs)
        if mask_opt_enabled and mask_steps:
            results[0] = apply_mask_optimizations_to_result(
                results[0],
                mask_opt_enabled,
                mask_steps,
            )
        annotated_bgr = annotate_from_results(
            results[0],
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_points,
            show_confidence,
            simplify_mode,
            simplify_eps_coeff,
            allowed_class_ids,
        )
        out.write(annotated_bgr)

    cap.release()
    out.release()
    return out_path


def yolov12_multi_inference_image(
    image,
    model_ids: List[str],
    image_size: int,
    conf_threshold: float,
    device: Optional[str],
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_points: bool,
    show_confidence: bool,
    simplify_mode: str,
    simplify_eps_coeff: float,
    allowed_class_ids: Optional[List[int]],
):
    """
    同一張 image，對多個模型推論。
    回傳：
      - gallery_items: [(RGB image, caption), ...]
      - results_cache: {model_name: results}
    """
    gallery_items: List[Tuple[np.ndarray, str]] = []
    results_cache: Dict[str, Any] = {}

    for mid in model_ids:
        img_rgb, results = infer_image_single(
            mid,
            image,
            image_size,
            conf_threshold,
            device,
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_points,
            show_confidence,
            simplify_mode,
            simplify_eps_coeff,
            allowed_class_ids,
        )
        gallery_items.append((img_rgb, mid))
        results_cache[mid] = results

    return gallery_items, results_cache


def yolov12_multi_inference_video(
    video,
    model_ids: List[str],
    image_size: int,
    conf_threshold: float,
    device: Optional[str],
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_points: bool,
    show_confidence: bool,
    simplify_mode: str,
    simplify_eps_coeff: float,
    allowed_class_ids: Optional[List[int]],
    mask_opt_enabled: bool,
    mask_opt_steps,
):
    """
    同一支影片對多個模型推論。
    回傳 [(model_name, out_video_path), ...]，最多 MAX_MODELS。
    """
    # Gradio Video 元件有時是 dict，有時是 path，保險處理一下
    if isinstance(video, dict) and "name" in video:
        src_path = video["name"]
    else:
        src_path = video

    outs: List[Tuple[str, str]] = []
    for mid in model_ids[:MAX_MODELS]:
        out_path = infer_video_single(
            mid,
            src_path,
            image_size,
            conf_threshold,
            device,
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_points,
            show_confidence,
            simplify_mode,
            simplify_eps_coeff,
            allowed_class_ids,
            mask_opt_enabled,
            mask_opt_steps,
        )
        outs.append((mid, out_path))

    return outs


def yolov12_inference_for_examples(
    image,
    model_list,
    image_size: int,
    conf_threshold: float,
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
):
    """
    給 Gradio Examples 用：
      - 支援多模型
      - 預設顯示全部類別 + 信心值 + polygon
    """
    if isinstance(model_list, str):
        model_ids = [model_list]
    else:
        model_ids = model_list or []
    if not model_ids:
        return []

    gallery, _ = yolov12_multi_inference_image(
        image,
        model_ids,
        image_size,
        conf_threshold,
        None,
        label_mode,
        show_boxes,
        show_masks,
        show_masks,
        True,   # show_polygons
        True,   # show_points
        True,   # show_confidence
        "rdp",
        1.0,
        allowed_class_ids=None,
    )
    return gallery
