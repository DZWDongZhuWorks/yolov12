import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .config import MAX_MODELS
from .inference_optimizations import apply_mask_optimizations_to_result, parse_mask_steps
from .inference_render import annotate_from_results
from .model_cache import get_model


def predict_image_single(
    model_id: str,
    image,
    image_size: int,
    conf_threshold: float,
    device: Optional[str],
):
    """只跑推論、不做標註渲染。

    結果一律移到 CPU：results 會被快取在 UI state（raw/last_results），
    且 plot() 的遮罩上色會配置 (n,h,w,3) 大張量並跟著遮罩裝置走，
    實例數多時留在 GPU 會 CUDA OOM。
    """
    model = get_model(model_id)
    predict_kwargs = {"source": image, "imgsz": image_size, "conf": conf_threshold}
    if device:
        predict_kwargs["device"] = device
    _t0 = time.perf_counter()
    results = model.predict(**predict_kwargs)
    predict_ms = (time.perf_counter() - _t0) * 1000
    _t0 = time.perf_counter()
    results = [r.cpu() for r in results]
    to_cpu_ms = (time.perf_counter() - _t0) * 1000

    speed = getattr(results[0], "speed", None) if results else None
    detail = ""
    if speed:
        detail = " (" + " | ".join(
            f"{k} {v:.1f}ms" for k, v in speed.items() if isinstance(v, (int, float))
        ) + ")"
    print(f"[Timing] predict {model_id}: total {predict_ms:.1f}ms{detail} | to_cpu {to_cpu_ms:.1f}ms")
    return results


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
    polygon_opt_steps,
    allowed_class_ids: Optional[List[int]],
):
    results = predict_image_single(model_id, image, image_size, conf_threshold, device)
    annotated_bgr = annotate_from_results(
        results[0], label_mode, show_boxes, show_masks, show_polygons, show_points,
        show_confidence, simplify_mode, simplify_eps_coeff, allowed_class_ids, polygon_opt_steps,
    )
    return annotated_bgr[:, :, ::-1], results


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
    polygon_opt_steps,
    allowed_class_ids: Optional[List[int]],
    mask_opt_enabled: bool,
    mask_opt_steps,
):
    model = get_model(model_id)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fd, out_path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"vp80"), fps, (frame_width, frame_height))
    mask_steps = parse_mask_steps(mask_opt_steps)
    pbar = tqdm(total=total_frames, desc=f"Processing {model_id}", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        predict_kwargs = {"source": frame, "imgsz": image_size, "conf": conf_threshold}
        if device:
            predict_kwargs["device"] = device
        # 移到 CPU：避免遮罩上色在 GPU 配置大張量導致 OOM（密集場景）
        results = [r.cpu() for r in model.predict(**predict_kwargs)]
        if mask_opt_enabled and mask_steps:
            results[0] = apply_mask_optimizations_to_result(results[0], mask_opt_enabled, mask_steps)
        annotated_bgr = annotate_from_results(
            results[0], label_mode, show_boxes, show_masks, show_polygons, show_points,
            show_confidence, simplify_mode, simplify_eps_coeff, allowed_class_ids, polygon_opt_steps,
        )
        out.write(annotated_bgr)
        pbar.update(1)

    pbar.close()
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
    polygon_opt_steps,
    allowed_class_ids: Optional[List[int]],
):
    gallery_items: List[Tuple[np.ndarray, str]] = []
    results_cache: Dict[str, Any] = {}

    for mid in model_ids:
        img_rgb, results = infer_image_single(
            mid, image, image_size, conf_threshold, device, label_mode, show_boxes, show_masks,
            show_polygons, show_points, show_confidence, simplify_mode, simplify_eps_coeff,
            polygon_opt_steps, allowed_class_ids,
        )
        gallery_items.append((img_rgb, mid))
        results_cache[mid] = results

    return gallery_items, results_cache


def yolov12_multi_predict_image(
    image,
    model_ids: List[str],
    image_size: int,
    conf_threshold: float,
    device: Optional[str],
) -> Dict[str, Any]:
    """多模型推論（不渲染），回傳 {model_id: results}。"""
    return {
        mid: predict_image_single(mid, image, image_size, conf_threshold, device)
        for mid in model_ids
    }


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
    polygon_opt_steps,
    allowed_class_ids: Optional[List[int]],
    mask_opt_enabled: bool,
    mask_opt_steps,
):
    src_path = video["name"] if isinstance(video, dict) and "name" in video else video
    outs: List[Tuple[str, str]] = []
    for mid in model_ids[:MAX_MODELS]:
        out_path = infer_video_single(
            mid, src_path, image_size, conf_threshold, device, label_mode, show_boxes, show_masks,
            show_polygons, show_points, show_confidence, simplify_mode, simplify_eps_coeff,
            polygon_opt_steps, allowed_class_ids, mask_opt_enabled, mask_opt_steps,
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
    """Helper used by Gradio examples to run one or more models on an image."""
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
        True,   # show_polygons
        False,  # show_points
        True,   # show_confidence
        "rdp",
        1.0,
        [],
        allowed_class_ids=None,
    )
    return gallery
