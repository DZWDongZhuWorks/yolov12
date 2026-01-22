# app_utils/inference.py

import copy
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
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
    show_confidence: bool,
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
        simplify_mode="rdp",    # 跟 export_utils 用同一個 mode
        simplify_eps_ratio=0.01,
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
        cv2.putText(
            base,
            base_text,
            (x1 + pad, y1 - pad),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return base


def split_connection_contours(result):
    """
    將單一 result 的 mask 進行連通元件分割，並回傳新的 Results。
    若 mask 本來就是單一連通區塊，則保留原樣。
    """
    if (
        not hasattr(result, "masks")
        or result.masks is None
        or not hasattr(result, "boxes")
        or result.boxes is None
        or len(result.boxes) == 0
    ):
        return result

    mask_data = getattr(result.masks, "data", None)
    boxes_data = getattr(result.boxes, "data", None)
    if mask_data is None or boxes_data is None or len(mask_data) == 0:
        return result

    masks_np = mask_data.detach().cpu().numpy()
    boxes_np = boxes_data.detach().cpu().numpy()

    new_masks: List[np.ndarray] = []
    new_boxes: List[np.ndarray] = []

    for i, mask_np in enumerate(masks_np):
        binary = mask_np > 0.5
        if binary.sum() == 0:
            continue

        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
        # num_labels: 1 (全背景) / 2 (背景+1物件) / n>2 (多物件)
        if num_labels <= 2:
            new_masks.append(binary)
            new_boxes.append(boxes_np[i])
            continue

        for label_id in range(1, num_labels):
            component = labels == label_id
            if component.sum() == 0:
                continue
            ys, xs = np.where(component)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            base_box = boxes_np[i]
            conf = base_box[4] if base_box.shape[0] > 4 else 0.0
            cls = base_box[5] if base_box.shape[0] > 5 else 0.0
            new_boxes.append(np.array([x1, y1, x2, y2, conf, cls], dtype=np.float32))
            new_masks.append(component)

    if not new_boxes:
        return result

    new_masks_np = np.stack(new_masks).astype(np.float32)
    new_boxes_np = np.stack(new_boxes).astype(np.float32)

    new_masks_t = torch.from_numpy(new_masks_np).to(mask_data.device).to(mask_data.dtype)
    new_boxes_t = torch.from_numpy(new_boxes_np).to(boxes_data.device).to(boxes_data.dtype)

    new_result = result.new()
    new_result.update(boxes=new_boxes_t, masks=new_masks_t)
    return new_result


def split_connection_contours_cache(results_cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    對 results_cache 中每個模型的 result 做連通元件切分。
    """
    if not results_cache:
        return results_cache

    new_cache: Dict[str, Any] = {}
    for mid, results in results_cache.items():
        if not results:
            new_cache[mid] = results
            continue
        new_cache[mid] = [split_connection_contours(results[0])]
    return new_cache


# -----------------------------
# 單模型推論（圖 / 影） + 多模型封裝
# -----------------------------
def infer_image_single(
    model_id: str,
    image,
    image_size: int,
    conf_threshold: float,
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_confidence: bool,
    allowed_class_ids: Optional[List[int]],
    connection_contour_split: bool = False,
):
    model = YOLO(model_id)
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    if connection_contour_split:
        results = [split_connection_contours(results[0])]
    annotated_bgr = annotate_from_results(
        results[0],
        label_mode,
        show_boxes,
        show_masks,
        show_polygons,
        show_confidence,
        allowed_class_ids,
    )
    # Gradio Image 用 RGB
    return annotated_bgr[:, :, ::-1], results  # (RGB, results)


def infer_video_single(
    model_id: str,
    video_path: str,
    image_size: int,
    conf_threshold: float,
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_confidence: bool,
    allowed_class_ids: Optional[List[int]],
    connection_contour_split: bool = False,
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
        if connection_contour_split:
            results = [split_connection_contours(results[0])]
        annotated_bgr = annotate_from_results(
            results[0],
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_confidence,
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
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_confidence: bool,
    allowed_class_ids: Optional[List[int]],
    connection_contour_split: bool = False,
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
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_confidence,
            allowed_class_ids,
            connection_contour_split,
        )
        gallery_items.append((img_rgb, mid))
        results_cache[mid] = results

    return gallery_items, results_cache


def yolov12_multi_inference_video(
    video,
    model_ids: List[str],
    image_size: int,
    conf_threshold: float,
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_polygons: bool,
    show_confidence: bool,
    allowed_class_ids: Optional[List[int]],
    connection_contour_split: bool = False,
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
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_confidence,
            allowed_class_ids,
            connection_contour_split,
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
        label_mode,
        show_boxes,
        show_masks,
        True,   # show_polygons
        True,   # show_confidence
        allowed_class_ids=None,
    )
    return gallery
