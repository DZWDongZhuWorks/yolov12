# cli_config.py
"""CLI 推論設定載入（app.py run_cli 與 run_tif_pipeline.py 共用）。"""
import json
import os
from typing import Any, Dict, Optional

from app_utils.inference_classes import parse_selected_to_ids
from app_utils.inference_optimizations import parse_mask_steps, parse_polygon_steps


def load_inference_config(config_path: Optional[str]) -> Dict[str, Any]:
    """讀取 app.py 匯出的 config JSON，回傳顯示設定與已解析的優化步驟。

    config 不存在時回傳預設值（優化全關）。回傳鍵：
    label_mode / show_boxes / show_masks / show_polygons / show_points / show_conf /
    allowed_class_ids / mask_opt_enabled / polygon_opt_enabled /
    mask_steps_parsed（未啟用為 None）/ polygon_steps_parsed（未啟用為 []）
    """
    cfg: Dict[str, Any] = {
        "label_mode": "顯示 class name",
        "show_boxes": True,
        "show_masks": True,
        "show_polygons": True,
        "show_points": True,
        "show_conf": True,
        "allowed_class_ids": None,
        "mask_opt_enabled": False,
        "polygon_opt_enabled": False,
    }
    mask_opt_st: list = []
    polygon_opt_st: list = []

    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        display = config.get("display", {})
        cfg["label_mode"] = display.get("label_mode", cfg["label_mode"])
        cfg["show_boxes"] = display.get("show_boxes", cfg["show_boxes"])
        cfg["show_masks"] = display.get("show_masks", cfg["show_masks"])
        cfg["show_polygons"] = display.get("show_polygons", cfg["show_polygons"])
        cfg["show_points"] = display.get("show_points", cfg["show_points"])
        cfg["show_conf"] = display.get("show_confidence", cfg["show_conf"])

        allowed_class_ids = config.get("class_filter", None)
        if allowed_class_ids is not None:
            allowed_class_ids = parse_selected_to_ids(allowed_class_ids)
        cfg["allowed_class_ids"] = allowed_class_ids

        mask_opt = config.get("mask_optimizations", {})
        cfg["mask_opt_enabled"] = mask_opt.get("enabled", False)
        mask_opt_st = mask_opt.get("steps", [])

        polygon_opt = config.get("polygon_optimizations", {})
        cfg["polygon_opt_enabled"] = polygon_opt.get("enabled", False)
        polygon_opt_st = polygon_opt.get("steps", [])
    else:
        print("No config file provided or file not found. Using default settings (Optimization: OFF).")

    cfg["mask_steps_parsed"] = parse_mask_steps(mask_opt_st) if cfg["mask_opt_enabled"] else None
    cfg["polygon_steps_parsed"] = parse_polygon_steps(polygon_opt_st) if cfg["polygon_opt_enabled"] else []
    return cfg
