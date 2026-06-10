"""Backward-compatible facade for app inference utilities.

This module keeps app.py imports stable while delegating implementation to
smaller focused modules.
"""

from .inference_classes import get_model_names, names_to_choice_list, parse_selected_to_ids
from .inference_optimizations import (
    apply_connection_contour_split,
    apply_mask_optimizations,
    apply_mask_optimizations_to_result,
    parse_mask_steps,
    parse_polygon_steps,
    split_connection_contours,
)
from .inference_render import annotate_from_results, filter_result_by_classes
from .inference_runner import (
    infer_image_single,
    infer_video_single,
    predict_image_single,
    yolov12_inference_for_examples,
    yolov12_multi_inference_image,
    yolov12_multi_inference_video,
    yolov12_multi_predict_image,
)
from .model_cache import clear_model_cache, get_model

__all__ = [
    "annotate_from_results",
    "apply_connection_contour_split",
    "apply_mask_optimizations",
    "apply_mask_optimizations_to_result",
    "clear_model_cache",
    "filter_result_by_classes",
    "get_model",
    "get_model_names",
    "infer_image_single",
    "infer_video_single",
    "predict_image_single",
    "yolov12_inference_for_examples",
    "names_to_choice_list",
    "parse_mask_steps",
    "parse_polygon_steps",
    "parse_selected_to_ids",
    "split_connection_contours",
    "yolov12_multi_inference_image",
    "yolov12_multi_inference_video",
    "yolov12_multi_predict_image",
]
