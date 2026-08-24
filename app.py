import datetime
import os
import re
import tempfile
import json
import argparse
import glob
from typing import List, Optional

import cv2
import gradio as gr

from app_utils.config import MAX_MODELS
from app_utils.model_registry import load_model_choices, persist_model_choices
from app_utils.inference import (
    names_to_choice_list,
    parse_selected_to_ids,
    annotate_from_results,
    apply_mask_optimizations,
    yolov12_multi_predict_image,
    yolov12_multi_inference_video,
    get_model_names,
    parse_polygon_steps,
)
from app_utils.export_utils import export_results_cache
from app_utils.inference_optimizations import (
    DEFAULT_POLYGON_SMALL_OBJECT_MAX_AREA,
    DEFAULT_POLYGON_SMALL_OBJECT_TARGET_VERTICES,
)


def _pad_polygon_rows(rows):
    """將舊版 polygon 優化 dataframe row 補齊到 10 欄。

    舊 schema：[step, enabled, count, eps_coeff, min_aspect, pca_min_cosine, pca_cross_class, classes] (8 欄)
    更舊：     [step, enabled, count, eps_coeff, min_aspect, pca_min_cosine, pca_cross_class] (7 欄)
    新 schema：[step, enabled, count, eps_coeff, min_aspect, pca_min_cosine, pca_cross_class, max_area_px, target_vertices, classes] (10 欄)
    """
    if rows is None:
        return []
    out = []
    for r in rows:
        if r is None:
            continue
        r = list(r)
        if len(r) == 8:
            r = r[:7] + [DEFAULT_POLYGON_SMALL_OBJECT_MAX_AREA,
                         DEFAULT_POLYGON_SMALL_OBJECT_TARGET_VERTICES,
                         r[7]]
        elif len(r) == 7:
            r = r + [DEFAULT_POLYGON_SMALL_OBJECT_MAX_AREA,
                     DEFAULT_POLYGON_SMALL_OBJECT_TARGET_VERTICES,
                     ""]
        while len(r) < 10:
            r.append(None)
        out.append(r[:10])
    return out


APP_CSS = """
.app-main-row {
    align-items: flex-start;
}
.control-panel {
    max-height: calc(100vh - 120px);
    overflow-y: auto;
    position: sticky;
    top: 16px;
    padding-right: 8px;
}
.result-panel {
    position: sticky;
    top: 16px;
    max-height: calc(100vh - 120px);
    overflow-y: auto;
}

.mask-steps-table textarea,
.polygon-steps-table textarea,
.mask-steps-table td:nth-child(10) > div,
.polygon-steps-table td:nth-child(10) > div {
    max-height: 80px;
    overflow: auto !important;
    white-space: pre-wrap;
}

.mask-steps-table th:nth-child(10),
.mask-steps-table td:nth-child(10),
.polygon-steps-table th:nth-child(10),
.polygon-steps-table td:nth-child(10) {
    min-width: 220px !important;
    width: 220px !important;
}
"""


def app():
    with gr.Blocks(css=APP_CSS) as demo:
        # === 初始模型清單（預設 + 已儲存自訂） ===
        initial_choices, initial_saved_custom = load_model_choices()

        with gr.Row(elem_classes=["app-main-row"]):
            # ======================= 左側：輸入與控制面板 =======================
            with gr.Column(scale=5, elem_classes=["control-panel"]):
                with gr.Tabs():
                    with gr.Tab("輸入 / 模型"):
                        # 影像 / 影片輸入
                        # type="filepath"：保留原始上傳檔名（type="pil" 會在前處理後遺失 filename，
                        # ultralytics 對記憶體影像一律命名 image0）
                        image = gr.Image(type="filepath", label="Image", visible=True)
                        video = gr.Video(label="Video", visible=False)
                        input_type = gr.Radio(
                            choices=["Image", "Video"],
                            value="Image",
                            label="Input Type",
                        )

                        # 記錄自訂模型清單（不包含 DEFAULT_MODELS）
                        saved_models_state = gr.State(value=initial_saved_custom)

                        # 多模型 Dropdown（支援自訂、可多選）
                        model_ids = gr.Dropdown(
                            label="Models (多選比較，最多 5)",
                            choices=initial_choices,
                            value=["yolov12m.pt"],
                            allow_custom_value=True,
                            multiselect=True,
                        )

                        image_size = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=2560,
                            step=32,
                            value=640,
                        )
                        conf_threshold = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.25,
                        )
                        device_select = gr.Dropdown(
                            label="Device",
                            choices=["auto", "cpu", "cuda:0", "cuda:1", "mps"],
                            value="auto",
                            allow_custom_value=True,
                        )

                    with gr.Tab("顯示 / 執行"):
                        label_mode = gr.CheckboxGroup(
                            choices=["顯示 class id", "顯示 class name"],
                            value=["顯示 class name"],
                            label="標籤顯示（可複選；全部取消 = 隱藏）",
                        )
                        show_boxes = gr.Checkbox(value=True, label="顯示 bbox 外框")
                        show_masks = gr.Checkbox(value=True, label="顯示 segmentation 遮罩")
                        show_polygons = gr.Checkbox(value=True, label="顯示 polygon 邊界")
                        show_points = gr.Checkbox(value=True, label="顯示 polygon 點")
                        show_confidence = gr.Checkbox(value=True, label="顯示信心值 (conf)")

                        yolov12_infer = gr.Button(value="Detect Objects (Run)")

                        # 匯出 JSON（polygon）
                        export_btn = gr.Button(
                            value="Export JSON (Polygons)",
                            variant="primary",
                        )
                        export_files = gr.Files(label="Exported JSON Files")

                        # 類別篩選（影響渲染與 JSON 匯出）
                        gr.Markdown("### 類別篩選（預設全選）")
                        class_filter_query = gr.Textbox(
                            label="類別查詢",
                            placeholder="輸入關鍵字或 class id",
                        )
                        with gr.Accordion("類別（ID: 名稱）", open=False):
                            class_selector = gr.CheckboxGroup(
                                label="類別（ID: 名稱）",
                                choices=[],
                                value=[],
                                interactive=True,
                            )
                        with gr.Row():
                            select_all_btn = gr.Button(value="全部選取", variant="secondary")
                            clear_all_btn = gr.Button(value="全部取消", variant="secondary")

                    with gr.Tab("Mask 優化"):
                        gr.Markdown("### Mask 優化")
                        mask_opt_enable = gr.Checkbox(
                            value=False,
                            label="啟用 Mask 優化（手動切換後立即重算）",
                        )
                        with gr.Accordion("新增步驟", open=True):
                            with gr.Row():
                                mask_opt_method = gr.Dropdown(
                                    label="步驟方法",
                                    choices=[
                                        "erode",
                                        "dilate",
                                        "distance_erode",
                                        "distance_dilate",
                                        "split",
                                        "merge",
                                        "blur",
                                        "remove_small",
                                        "fill_holes",
                                    ],
                                    value="erode",
                                )
                                mask_opt_count = gr.Slider(
                                    label="次數",
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=1,
                                )
                            # 參數依所選方法動態顯示（預設 erode → 只顯示 Morph Kernel）
                            with gr.Row():
                                step_morph_kernel = gr.Slider(
                                    label="Morph Kernel (odd)",
                                    minimum=1,
                                    maximum=15,
                                    step=2,
                                    value=3,
                                    visible=True,
                                )
                                step_blur_kernel = gr.Slider(
                                    label="Blur Kernel (odd)",
                                    minimum=1,
                                    maximum=15,
                                    step=2,
                                    value=3,
                                    visible=False,
                                )
                                step_blur_threshold = gr.Slider(
                                    label="Blur Threshold",
                                    minimum=0.1,
                                    maximum=0.9,
                                    step=0.05,
                                    value=0.5,
                                    visible=False,
                                )
                                step_min_component_area = gr.Slider(
                                    label="Min Component Area",
                                    minimum=0,
                                    maximum=5000,
                                    step=10,
                                    value=0,
                                    visible=False,
                                )
                                step_max_hole_area = gr.Slider(
                                    label="Max Hole Area",
                                    minimum=0,
                                    maximum=5000,
                                    step=10,
                                    value=0,
                                    visible=False,
                                )
                                step_merge_iou_threshold = gr.Slider(
                                    label="Merge IoU Threshold",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=0.1,
                                    visible=False,
                                )
                            step_class_filter_query = gr.Textbox(
                                label="類別查詢",
                                placeholder="輸入關鍵字或 class id",
                            )
                            with gr.Accordion("套用類別（未選或全選 = 全部類別）", open=False):
                                step_class_filter = gr.CheckboxGroup(
                                    label="Classes",
                                    choices=[],
                                    value=[],
                                )
                                with gr.Row():
                                    step_select_all_btn = gr.Button(value="全部選取", variant="secondary")
                                    step_clear_all_btn = gr.Button(value="全部取消", variant="secondary")
                            mask_opt_add = gr.Button(value="加入步驟", variant="secondary")
                        gr.Markdown(
                            "步驟由上而下依序執行，可直接編輯表格。`enabled` 開關單一步驟；空白欄位使用預設值；`classes` 留空 = 全部類別。表格編輯後按「套用」更新結果。"
                        )
                        mask_opt_steps = gr.Dataframe(
                            headers=[
                                "step",
                                "enabled",
                                "count",
                                "morph_kernel",
                                "blur_kernel",
                                "blur_threshold",
                                "min_component_area",
                                "max_hole_area",
                                "merge_iou_threshold",
                                "classes",
                            ],
                            datatype=[
                                "str",
                                "bool",
                                "number",
                                "number",
                                "number",
                                "number",
                                "number",
                                "number",
                                "number",
                                "str",
                            ],
                            row_count=0,
                            col_count=(10, "fixed"),
                            wrap=True,
                            label="Mask 優化流程",
                            type="array",
                            column_widths=["120px", "90px", "90px", "120px", "120px", "130px", "150px", "130px", "160px", "220px"],
                            elem_classes=["mask-steps-table"],
                        )
                        with gr.Row():
                            mask_opt_apply = gr.Button(value="套用 Mask 優化並重繪", variant="primary")
                            mask_opt_clear = gr.Button(value="清空步驟", variant="secondary")

                    with gr.Tab("Polygon 優化"):
                        gr.Markdown("### Polygon 優化")
                        polygon_opt_enable = gr.Checkbox(
                            value=False,
                            label="啟用 Polygon 優化（手動切換後立即重繪）",
                        )
                        with gr.Accordion("新增步驟", open=True):
                            with gr.Row():
                                polygon_opt_method = gr.Dropdown(
                                    label="步驟方法",
                                    choices=["convex_hull", "rdp", "visvalingam_whyatt", "min_area_rect", "export_line", "polygon_to_lane_line", "pca", "small_object_fit", "smooth_spline", "fit_lines_arcs", "fit_spline_pca"],
                                    value="rdp",
                                )
                                polygon_opt_count = gr.Slider(
                                    label="次數",
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=1,
                                )
                            # 參數依所選方法動態顯示（預設 rdp → 只顯示 Epsilon 係數）
                            with gr.Row():
                                polygon_step_eps_coeff = gr.Slider(
                                    label="Epsilon 係數（簡化 / 對齊強度）",
                                    minimum=0.1,
                                    maximum=5.0,
                                    step=0.1,
                                    value=1.0,
                                    visible=True,
                                )
                                polygon_step_min_aspect = gr.Slider(
                                    label="Min Aspect（長寬比門檻）",
                                    minimum=0.0,
                                    maximum=20.0,
                                    step=0.1,
                                    value=0.0,
                                    visible=False,
                                )
                                polygon_step_pca_min_cosine = gr.Slider(
                                    label="PCA Min Cosine（方向分群門檻）",
                                    minimum=0.7,
                                    maximum=0.999,
                                    step=0.001,
                                    value=0.94,
                                    visible=False,
                                )
                                polygon_step_pca_cross_class = gr.Checkbox(
                                    label="PCA 跨類別共同計算",
                                    value=False,
                                    visible=False,
                                )
                                polygon_step_max_area_px = gr.Slider(
                                    label="Max Area px²（0 = 不限制）",
                                    minimum=0.0,
                                    maximum=50000.0,
                                    step=100.0,
                                    value=5000.0,
                                    visible=False,
                                )
                                polygon_step_target_vertices = gr.Slider(
                                    label="Target Vertices",
                                    minimum=3,
                                    maximum=32,
                                    step=1,
                                    value=8,
                                    visible=False,
                                )
                            polygon_step_class_filter_query = gr.Textbox(
                                label="類別查詢",
                                placeholder="輸入關鍵字或 class id",
                            )
                            with gr.Accordion("套用類別（未選或全選 = 全部類別）", open=False):
                                polygon_step_class_filter = gr.CheckboxGroup(
                                    label="Classes",
                                    choices=[],
                                    value=[],
                                )
                                with gr.Row():
                                    polygon_step_select_all_btn = gr.Button(value="全部選取", variant="secondary")
                                    polygon_step_clear_all_btn = gr.Button(value="全部取消", variant="secondary")
                            polygon_opt_add = gr.Button(value="加入步驟", variant="secondary")
                        gr.Markdown(
                            "步驟由上而下依序執行，可直接編輯表格。`enabled` 開關單一步驟；空白欄位使用預設值；`classes` 留空 = 全部類別。表格編輯後按「套用」更新結果。"
                        )
                        polygon_opt_steps = gr.Dataframe(
                            headers=["step", "enabled", "count", "eps_coeff", "min_aspect", "pca_min_cosine", "pca_cross_class", "max_area_px", "target_vertices", "classes"],
                            datatype=["str", "bool", "number", "number", "number", "number", "bool", "number", "number", "str"],
                            row_count=0,
                            col_count=(10, "fixed"),
                            wrap=True,
                            label="Polygon 優化流程",
                            type="array",
                            column_widths=["150px", "80px", "80px", "100px", "100px", "110px", "100px", "110px", "120px", "220px"],
                            elem_classes=["polygon-steps-table"],
                        )
                        with gr.Row():
                            polygon_opt_apply = gr.Button(value="套用 Polygon 優化並重繪", variant="primary")
                            polygon_opt_clear = gr.Button(value="清空步驟", variant="secondary")

                    with gr.Tab("配置管理 (Config)"):
                        gr.Markdown("### 儲存 / 載入配置\n將目前的「顯示 / 執行」、「Mask 優化」與「Polygon 優化」設定匯出為 JSON，或從檔案還原。")
                        with gr.Row():
                            export_config_btn = gr.Button("匯出目前配置", variant="primary")
                            import_config_file = gr.File(label="匯入配置檔 (.json)", type="filepath")
                        export_config_file = gr.File(label="下載配置檔", interactive=False)

                # 保留目前 choices 狀態（避免僅從元件讀不到 choices）
                class_choices_state = gr.State(value=[])
                # 保存當前影像中繼資訊（檔名、寬高）
                image_meta_state = gr.State(value=None)
                # 當前配置名稱（匯入/匯出 config 時更新，用於下載圖檔命名）
                config_name_state = gr.State(value=None)

                # Global Selection State (To persist selection even when filtered)
                selected_classes_global = gr.State(value=[])
                step_selected_classes_global = gr.State(value=[])
                polygon_step_selected_classes_global = gr.State(value=[])

            # ======================= 右側：輸出 =======================
            with gr.Column(scale=7, elem_classes=["result-panel"]):
                # 影像輸出：Gallery 並排
                output_gallery = gr.Gallery(
                    label="Annotated Images（多模型比較）",
                    columns=2,
                    preview=True,
                    visible=True,
                )
                # 影片輸出：最多 5 路
                with gr.Group(visible=False) as video_group:
                    with gr.Row():
                        v1 = gr.Video(label="Model #1")
                        v2 = gr.Video(label="Model #2")
                    with gr.Row():
                        v3 = gr.Video(label="Model #3")
                        v4 = gr.Video(label="Model #4")
                    v5 = gr.Video(label="Model #5")

        # 快取最後一次「影像」結果（每個模型一份）
        # 型別: Dict[str, results]
        last_results = gr.State(value=None)
        raw_results = gr.State(value=None)

        def resolve_allowed_ids(class_selected_items_in):
            selected_ids = parse_selected_to_ids(class_selected_items_in)
            if class_selected_items_in is None:
                return None
            if class_selected_items_in == []:
                return []
            return selected_ids

        def _fs_safe(name: str) -> str:
            return re.sub(r"[^\w\-]", "_", str(name), flags=re.UNICODE)

        def render_gallery_from_results(
            results_dict,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_points_in,
            show_conf_in,
            polygon_steps,
            allowed_ids,
            image_meta=None,
            config_name=None,
        ):
            if not results_dict:
                return []

            file_name = (image_meta or {}).get("file_name") if isinstance(image_meta, dict) else None
            image_stem = _fs_safe(os.path.splitext(os.path.basename(file_name))[0]) if file_name else None
            config_stem = _fs_safe(config_name) if config_name else None

            # 存成具名圖檔讓 Gallery 下載時保留檔名：{原始檔名}__{config}__{模型}.jpg
            out_dir = tempfile.mkdtemp(prefix="yolo_gallery_")
            gallery = []
            for mid, results in results_dict.items():
                annotated_bgr = annotate_from_results(
                    results[0],
                    label_mode_in,
                    show_boxes_in,
                    show_masks_in,
                    show_polygons_in,
                    show_points_in,
                    show_conf_in,
                    "none",
                    1.0,
                    allowed_ids,
                    polygon_steps,
                )
                caption = f"{file_name} | {mid}" if file_name else str(mid)
                model_stem = _fs_safe(os.path.splitext(os.path.basename(str(mid)))[0])
                name_parts = [p for p in (image_stem, config_stem, model_stem) if p]
                out_path = os.path.join(out_dir, "__".join(name_parts) + ".jpg")
                ok, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    buf.tofile(out_path)  # tofile 可處理非 ASCII 路徑
                    gallery.append((out_path, caption))
                else:
                    gallery.append((annotated_bgr[:, :, ::-1], caption))  # 後備：直接給陣列

            return gallery

        # ======== Input Type 切換：控制元件可視性 ========
        def update_visibility(input_type_val: str):
            image_v = gr.update(visible=(input_type_val == "Image"))
            video_v = gr.update(visible=(input_type_val == "Video"))
            gallery_v = gr.update(visible=(input_type_val == "Image"))
            video_group_v = gr.update(visible=(input_type_val == "Video"))
            return image_v, video_v, gallery_v, video_group_v

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_gallery, video_group],
        )

        # ======== 主推論函式 ========
        def run_inference(
            image_in,
            video_in,
            model_ids_in,
            image_size_in,
            conf_th_in,
            device_in,
            input_type_in,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in, 
            show_points_in,
            show_conf_in,
            mask_opt_enable_in,
            mask_opt_steps_in,
            polygon_opt_enable_in,
            polygon_opt_steps_in,
            saved_models_in,
            class_selected_items_in,
            class_choices_in,
            config_name_in,
        ):
            polygon_steps = parse_polygon_steps(polygon_opt_steps_in) if polygon_opt_enable_in else []
            # 1) 正規化模型清單（最多 MAX_MODELS 個）
            if isinstance(model_ids_in, str):
                mids: List[str] = [model_ids_in]
            else:
                mids = [m for m in (model_ids_in or []) if m]
            mids = mids[:MAX_MODELS]

            # 沒有選到任何模型：清空結果快取並維持現有狀態（未列出的元件不更新）
            if not mids:
                # 由目前已儲存的自訂模型重建 choices，避免重置回 app 啟動時的清單
                empty_choices, _ = persist_model_choices(saved_models_in, [])
                return {
                    last_results: None,
                    raw_results: None,
                    model_ids: gr.update(choices=empty_choices, value=[]),
                    saved_models_state: saved_models_in,
                    class_choices_state: class_choices_in or [],
                    image_meta_state: None,
                    step_class_filter: gr.update(choices=class_choices_in or [], value=[]),
                    polygon_step_class_filter: gr.update(choices=class_choices_in or [], value=[]),
                    class_filter_query: "",
                    step_class_filter_query: "",
                    selected_classes_global: [],
                    step_selected_classes_global: [],
                    polygon_step_selected_classes_global: [],
                }

            # 2) 持久化自訂模型選項
            new_choices, new_saved = persist_model_choices(saved_models_in, mids)

            # 3) 類別篩選條件計算
            allowed_ids: Optional[List[int]] = resolve_allowed_ids(class_selected_items_in)

            # 3-1) 自動取得模型 class list（等同按下「讀取模型資訊」）
            class_choices_new = class_choices_in or []
            try:
                auto_names = get_model_names(mids[0])
                if auto_names:
                    class_choices_new, _ = names_to_choice_list(auto_names)
            except Exception:
                pass

            if not class_selected_items_in and class_choices_new:
                class_selected_items_out = class_choices_new
            else:
                valid_set = set(class_choices_new)
                class_selected_items_out = [
                    c for c in (class_selected_items_in or []) if c in valid_set
                ]

            # 共通更新：模型清單、類別選單與全域選取狀態（各分支皆相同）
            common_updates = {
                model_ids: gr.update(choices=new_choices, value=mids),
                saved_models_state: new_saved,
                class_selector: gr.update(choices=class_choices_new, value=class_selected_items_out),
                class_choices_state: class_choices_new,
                step_class_filter: gr.update(choices=class_choices_new, value=class_choices_new),
                polygon_step_class_filter: gr.update(choices=class_choices_new, value=class_choices_new),
                class_filter_query: "",
                step_class_filter_query: "",
                selected_classes_global: class_selected_items_out,
                step_selected_classes_global: class_choices_new,
                polygon_step_selected_classes_global: class_choices_new,
            }

            # 4) Image 模式
            if input_type_in == "Image":
                if image_in is None:
                    # 沒有圖可跑：清空結果快取並回傳共通更新
                    return {
                        **common_updates,
                        last_results: None,
                        raw_results: None,
                        image_meta_state: None,
                    }

                # 4-1) 多模型推論（不渲染，渲染統一交給 render_gallery_from_results）
                results_cache = yolov12_multi_predict_image(
                    image_in,
                    mids,
                    image_size_in,
                    conf_th_in,
                    None if device_in == "auto" else device_in,
                )

                raw_results_cache = results_cache
                results_cache = apply_mask_optimizations(
                    results_cache,
                    mask_opt_enable_in,
                    mask_opt_steps_in,
                )

                # 4-2) 構建 image meta（檔名、寬高）——先建 meta，gallery 命名與標題會用到
                first_result = next(iter(results_cache.values()), [None])[0]
                width = height = 0
                try:
                    if first_result is not None:
                        h, w = map(int, getattr(first_result, "orig_shape", (0, 0))[:2])
                        width, height = w, h
                except Exception:
                    width = height = 0

                fname = None
                # filepath 輸入：直接取檔名
                if isinstance(image_in, str):
                    fname = os.path.basename(image_in)
                # PIL 後備：從物件上抓 filename/name/path
                if not fname:
                    for attr in ("filename", "name", "path"):
                        candidate = getattr(image_in, attr, None)
                        if candidate:
                            fname = os.path.basename(str(candidate))
                            break
                # 再退而求其次從 result.path
                if not fname and first_result is not None:
                    rp = getattr(first_result, "path", None)
                    if rp:
                        fname = os.path.basename(str(rp))
                if not fname:
                    fname = "uploaded_image.png"

                image_meta = {
                    "file_name": fname,
                    "width": int(width),
                    "height": int(height),
                }

                gallery = render_gallery_from_results(
                    results_cache,
                    label_mode_in,
                    show_boxes_in,
                    show_masks_in,
                    show_polygons_in,
                    show_points_in,
                    show_conf_in,
                    polygon_steps,
                    allowed_ids,
                    image_meta=image_meta,
                    config_name=config_name_in,
                )

                return {
                    **common_updates,
                    output_gallery: gallery,
                    v1: gr.update(value=None, label="Model #1"),
                    v2: gr.update(value=None, label="Model #2"),
                    v3: gr.update(value=None, label="Model #3"),
                    v4: gr.update(value=None, label="Model #4"),
                    v5: gr.update(value=None, label="Model #5"),
                    last_results: results_cache,
                    raw_results: raw_results_cache,
                    image_meta_state: image_meta,
                }

            # 5) Video 模式
            else:
                if video_in is None:
                    # 沒有影片可跑：清空結果快取並回傳共通更新
                    return {
                        **common_updates,
                        last_results: None,
                        raw_results: None,
                        image_meta_state: None,
                    }

                outs = yolov12_multi_inference_video(
                    video_in,
                    mids,
                    image_size_in,
                    conf_th_in,
                    None if device_in == "auto" else device_in,
                    label_mode_in,
                    show_boxes_in,
                    show_masks_in,
                    show_polygons_in, 
                    show_points_in,
                    show_conf_in,
                    "none",
                    1.0,
                    polygon_steps,
                    allowed_class_ids=allowed_ids,
                    mask_opt_enabled=mask_opt_enable_in,
                    mask_opt_steps=mask_opt_steps_in,
                )

                video_updates = [gr.update(value=None, visible=False)] * 5
                for idx, (mid, out_path) in enumerate(outs[:5]):
                    video_updates[idx] = gr.update(value=out_path, label=str(mid), visible=True)

                return {
                    **common_updates,
                    v1: video_updates[0],
                    v2: video_updates[1],
                    v3: video_updates[2],
                    v4: video_updates[3],
                    v5: video_updates[4],
                    last_results: None,  # 影片不快取
                    raw_results: None,
                    image_meta_state: None,  # 影片無需
                }

        yolov12_infer.click(
            fn=run_inference,
            inputs=[
                image,
                video,
                model_ids,
                image_size,
                conf_threshold,
                device_select,
                input_type,
                label_mode,
                show_boxes,
                show_masks,
                show_polygons,
                show_points,
                show_confidence,
                mask_opt_enable,
                mask_opt_steps,
                polygon_opt_enable,
                polygon_opt_steps,
                saved_models_state,
                class_selector,
                class_choices_state,
                config_name_state,
            ],
            outputs=[
                output_gallery,
                v1,
                v2,
                v3,
                v4,
                v5,
                last_results,
                raw_results,
                model_ids,
                saved_models_state,
                class_selector,
                class_choices_state,
                image_meta_state,
                step_class_filter,
                polygon_step_class_filter,
                class_filter_query,
                step_class_filter_query,
                selected_classes_global,      # NEW output
                step_selected_classes_global, # NEW output
                polygon_step_selected_classes_global,
            ],
        )

        # ======== 即時重繪（只針對 Image 模式） ========
        def _filter_class_choices(query_text: str, choices: List[str]) -> List[str]:
            if not choices:
                return []
            query = (query_text or "").strip().lower()
            if not query:
                return choices
            filtered: List[str] = []
            for choice in choices:
                choice_text = str(choice).lower()
                if query in choice_text:
                    filtered.append(choice)
            return filtered

        def update_global_selection(query_text, all_choices, current_visible_selection, old_global_selection):
            # 1. Determine visible choices
            visible_choices = _filter_class_choices(query_text, all_choices)
            visible_set = set(visible_choices)
            
            # 2. Keep items from old global that are NOT visible (hidden ones)
            hidden_selected = [c for c in (old_global_selection or []) if c not in visible_set]
            
            # 3. Add items currently selected in the visible UI
            # Note: current_visible_selection might contain items not in visible_set if logic is loose, 
            # but usually it comes from CheckboxGroup value which is restricted to choices.
            new_global_set = set(hidden_selected + (current_visible_selection or []))
            
            # 4. Sort to maintain consistency (optional but good for UX)
            # Try to sort by ID if possible
            def sort_key(s):
                try:
                    return int(str(s).split(":")[0])
                except:
                    return str(s)
            
            return sorted(list(new_global_set), key=sort_key)

        def update_class_selector_filter_and_sync(query_text, all_choices, global_selected):
            # Calculate visible choices
            filtered = _filter_class_choices(query_text, all_choices)
            valid_global = set(global_selected or [])
            # Value for UI is intersection of Visible & Global
            ui_value = [c for c in filtered if c in valid_global]
            return gr.update(choices=filtered, value=ui_value)

        def on_class_selector_change(current_visible_selection, query_text, all_choices, old_global):
            new_global = update_global_selection(query_text, all_choices, current_visible_selection, old_global)
            return new_global

        def on_step_class_selector_change(current_visible_selection, query_text, all_choices, old_global):
            new_global = update_global_selection(query_text, all_choices, current_visible_selection, old_global)
            return new_global

        def sync_class_choices_from_model(model_ids_in, selected_items_in, old_choices_in):
            if isinstance(model_ids_in, str):
                mids = [model_ids_in]
            else:
                mids = [m for m in (model_ids_in or []) if m]

            if not mids:
                return (
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    [],
                    [],
                    [],
                    [],
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                )

            names = get_model_names(mids[0])
            if not names:
                choices = old_choices_in or []
            else:
                choices, _ = names_to_choice_list(names)

            if not selected_items_in and choices:
                selected_items_out = choices
            else:
                valid_set = set(choices)
                selected_items_out = [c for c in (selected_items_in or []) if c in valid_set]

            return (
                gr.update(choices=choices, value=selected_items_out),
                gr.update(choices=choices, value=choices),
                gr.update(choices=choices, value=choices),
                choices,
                selected_items_out,
                choices,
                choices,
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
            )

        model_ids.change(
            fn=sync_class_choices_from_model,
            inputs=[model_ids, selected_classes_global, class_choices_state],
            outputs=[
                class_selector,
                step_class_filter,
                polygon_step_class_filter,
                class_choices_state,
                selected_classes_global,
                step_selected_classes_global,
                polygon_step_selected_classes_global,
                class_filter_query,
                step_class_filter_query,
                polygon_step_class_filter_query,
            ],
        )
        # ======== 步驟參數對應表（動態顯示 + 加入步驟時只記錄相關參數） ========
        MASK_STEP_PARAMS = {
            "erode": ("morph_kernel",),
            "dilate": ("morph_kernel",),
            "distance_erode": ("morph_kernel",),
            "distance_dilate": ("morph_kernel",),
            "split": (),
            "merge": ("merge_iou_threshold",),
            "blur": ("blur_kernel", "blur_threshold"),
            "remove_small": ("min_component_area",),
            "fill_holes": ("max_hole_area",),
        }
        MASK_PARAM_ORDER = (
            "morph_kernel", "blur_kernel", "blur_threshold",
            "min_component_area", "max_hole_area", "merge_iou_threshold",
        )
        POLYGON_STEP_PARAMS = {
            "convex_hull": (),
            "rdp": ("eps_coeff",),
            "visvalingam_whyatt": ("eps_coeff",),
            "min_area_rect": ("min_aspect",),
            "export_line": ("min_aspect",),
            "polygon_to_lane_line": ("eps_coeff",),
            "pca": ("eps_coeff", "pca_min_cosine", "pca_cross_class"),
            "small_object_fit": ("max_area_px", "target_vertices"),
            "smooth_spline": ("eps_coeff",),
            "fit_lines_arcs": ("eps_coeff",),
            "fit_spline_pca": ("eps_coeff",),
        }
        POLYGON_PARAM_ORDER = (
            "eps_coeff", "min_aspect", "pca_min_cosine",
            "pca_cross_class", "max_area_px", "target_vertices",
        )

        def on_mask_method_change(method):
            visible = set(MASK_STEP_PARAMS.get(method, ()))
            return [gr.update(visible=(p in visible)) for p in MASK_PARAM_ORDER]

        mask_opt_method.change(
            fn=on_mask_method_change,
            inputs=[mask_opt_method],
            outputs=[
                step_morph_kernel, step_blur_kernel, step_blur_threshold,
                step_min_component_area, step_max_hole_area, step_merge_iou_threshold,
            ],
        )

        def on_polygon_method_change(method):
            visible = set(POLYGON_STEP_PARAMS.get(method, ()))
            return [gr.update(visible=(p in visible)) for p in POLYGON_PARAM_ORDER]

        polygon_opt_method.change(
            fn=on_polygon_method_change,
            inputs=[polygon_opt_method],
            outputs=[
                polygon_step_eps_coeff, polygon_step_min_aspect, polygon_step_pca_min_cosine,
                polygon_step_pca_cross_class, polygon_step_max_area_px, polygon_step_target_vertices,
            ],
        )

        # ======== 加入步驟 ========
        def _snapshot_class_filter(class_filter_in, all_choices_in):
            """未選或全選 → 存空字串（= 全部類別），否則存逗號分隔清單。"""
            if class_filter_in is None:
                return ""
            if isinstance(class_filter_in, (list, tuple, set)):
                valid_items = [str(x) for x in class_filter_in if x is not None]
                all_items = [str(x) for x in (all_choices_in or [])]
                if not valid_items or (all_items and set(valid_items) >= set(all_items)):
                    return ""
                return ", ".join(valid_items)
            return str(class_filter_in)

        def _steps_to_rows(steps):
            if steps is None:
                return []
            if hasattr(steps, "tolist"):
                return steps.tolist()
            if isinstance(steps, list):
                return list(steps)
            return []

        def add_mask_step(
            steps,
            method,
            count,
            morph_kernel_in,
            blur_kernel_in,
            blur_threshold_in,
            min_component_area_in,
            max_hole_area_in,
            merge_iou_threshold_in,
            class_filter_in,
            all_choices_in,
        ):
            rows = _steps_to_rows(steps)
            params = {
                "morph_kernel": morph_kernel_in,
                "blur_kernel": blur_kernel_in,
                "blur_threshold": blur_threshold_in,
                "min_component_area": min_component_area_in,
                "max_hole_area": max_hole_area_in,
                "merge_iou_threshold": merge_iou_threshold_in,
            }
            relevant = set(MASK_STEP_PARAMS.get(method, ()))
            rows.append(
                [method, True, int(count)]
                + [params[p] if p in relevant else None for p in MASK_PARAM_ORDER]
                + [_snapshot_class_filter(class_filter_in, all_choices_in)]
            )
            # 加入步驟即代表要使用優化 → 自動勾選啟用（不會觸發重算，需按「套用」）
            return rows, gr.update(value=True)

        mask_opt_add.click(
            fn=add_mask_step,
            inputs=[
                mask_opt_steps,
                mask_opt_method,
                mask_opt_count,
                step_morph_kernel,
                step_blur_kernel,
                step_blur_threshold,
                step_min_component_area,
                step_max_hole_area,
                step_merge_iou_threshold,
                step_class_filter,
                class_choices_state,
            ],
            outputs=[mask_opt_steps, mask_opt_enable],
        )

        def add_polygon_step(
            steps,
            method,
            count,
            eps_coeff_in,
            min_aspect_in,
            pca_min_cosine_in,
            pca_cross_class_in,
            max_area_px_in,
            target_vertices_in,
            class_filter_in,
            all_choices_in,
        ):
            rows = _steps_to_rows(steps)
            params = {
                "eps_coeff": float(eps_coeff_in),
                "min_aspect": float(min_aspect_in),
                "pca_min_cosine": float(pca_min_cosine_in),
                "pca_cross_class": bool(pca_cross_class_in),
                "max_area_px": float(max_area_px_in),
                "target_vertices": int(target_vertices_in),
            }
            relevant = set(POLYGON_STEP_PARAMS.get(method, ()))
            row_vals = [
                params[p] if p in relevant else (False if p == "pca_cross_class" else None)
                for p in POLYGON_PARAM_ORDER
            ]
            rows.append(
                [method, True, int(count)]
                + row_vals
                + [_snapshot_class_filter(class_filter_in, all_choices_in)]
            )
            return rows, gr.update(value=True)

        polygon_opt_add.click(
            fn=add_polygon_step,
            inputs=[
                polygon_opt_steps,
                polygon_opt_method,
                polygon_opt_count,
                polygon_step_eps_coeff,
                polygon_step_min_aspect,
                polygon_step_pca_min_cosine,
                polygon_step_pca_cross_class,
                polygon_step_max_area_px,
                polygon_step_target_vertices,
                polygon_step_class_filter,
                class_choices_state,
            ],
            outputs=[polygon_opt_steps, polygon_opt_enable],
        )

        # ======== 清空步驟 ========
        mask_opt_clear.click(fn=lambda: [], inputs=[], outputs=[mask_opt_steps])
        polygon_opt_clear.click(fn=lambda: [], inputs=[], outputs=[polygon_opt_steps])

        # --- Class Filter Logic Wiring ---
        # 1. When Query Changes -> Update UI Choices & Value (Read from Global)
        class_filter_query.change(
            fn=update_class_selector_filter_and_sync,
            inputs=[class_filter_query, class_choices_state, selected_classes_global],
            outputs=[class_selector],
        )
        
        # 2. When Checkbox Selection Changes -> Update Global State
        class_selector.change(
            fn=on_class_selector_change,
            inputs=[class_selector, class_filter_query, class_choices_state, selected_classes_global],
            outputs=[selected_classes_global],
        )
        
        # --- Step Class Filter Logic Wiring ---
        step_class_filter_query.change(
            fn=update_class_selector_filter_and_sync,
            inputs=[step_class_filter_query, class_choices_state, step_selected_classes_global],
            outputs=[step_class_filter],
        )
        step_class_filter.change(
             fn=on_step_class_selector_change,
             inputs=[step_class_filter, step_class_filter_query, class_choices_state, step_selected_classes_global],
             outputs=[step_selected_classes_global],
        )
        def step_select_all_classes(choices):
            return gr.update(value=choices or []), gr.update(value="")

        def step_clear_all_classes():
            return gr.update(value=[]), gr.update(value="")

        step_select_all_btn.click(
            fn=step_select_all_classes,
            inputs=[class_choices_state],
            outputs=[step_class_filter, step_class_filter_query],
        )
        step_clear_all_btn.click(
            fn=step_clear_all_classes,
            inputs=[],
            outputs=[step_class_filter, step_class_filter_query],
        )

        # --- Polygon Step Class Filter Logic Wiring ---
        polygon_step_class_filter_query.change(
            fn=update_class_selector_filter_and_sync,
            inputs=[polygon_step_class_filter_query, class_choices_state, polygon_step_selected_classes_global],
            outputs=[polygon_step_class_filter],
        )
        polygon_step_class_filter.change(
             fn=on_step_class_selector_change,
             inputs=[polygon_step_class_filter, polygon_step_class_filter_query, class_choices_state, polygon_step_selected_classes_global],
             outputs=[polygon_step_selected_classes_global],
        )
        polygon_step_select_all_btn.click(
            fn=step_select_all_classes,
            inputs=[class_choices_state],
            outputs=[polygon_step_class_filter, polygon_step_class_filter_query],
        )
        polygon_step_clear_all_btn.click(
            fn=step_clear_all_classes,
            inputs=[],
            outputs=[polygon_step_class_filter, polygon_step_class_filter_query],
        )

        def replot_all_filtered(
            last_results_dict,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_points_in,
            show_conf_in,
            polygon_opt_enable_in,
            polygon_opt_steps_in,
            input_type_in,
            class_selected_items_in,
            image_meta_in=None,
            config_name_in=None,
        ):
            if input_type_in != "Image" or not last_results_dict:
                return gr.update()

            allowed_ids = resolve_allowed_ids(class_selected_items_in)
            polygon_steps = parse_polygon_steps(polygon_opt_steps_in) if polygon_opt_enable_in else []

            return render_gallery_from_results(
                last_results_dict,
                label_mode_in,
                show_boxes_in,
                show_masks_in,
                show_polygons_in,
                show_points_in,
                show_conf_in,
                polygon_steps,
                allowed_ids,
                image_meta=image_meta_in,
                config_name=config_name_in,
            )

        def apply_polygon_replot_clicked(
            last_results_dict,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_points_in,
            show_conf_in,
            polygon_opt_enable_in,
            polygon_opt_steps_in,
            input_type_in,
            class_selected_items_in,
            image_meta_in=None,
            config_name_in=None,
        ):
            if not polygon_opt_enable_in:
                gr.Warning("尚未啟用 Polygon 優化：請先勾選「啟用 Polygon 優化」。目前以原始 polygon 重繪。")
            return replot_all_filtered(
                last_results_dict,
                label_mode_in,
                show_boxes_in,
                show_masks_in,
                show_polygons_in,
                show_points_in,
                show_conf_in,
                polygon_opt_enable_in,
                polygon_opt_steps_in,
                input_type_in,
                class_selected_items_in,
                image_meta_in,
                config_name_in,
            )

        # 標籤模式/框/遮罩/polygon/信心值 改變時即時重繪；
        # polygon 啟用開關只回應手動切換（.input），步驟表格編輯改由「套用」按鈕觸發
        _replot_inputs = [
            last_results,
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_points,
            show_confidence,
            polygon_opt_enable,
            polygon_opt_steps,
            input_type,
            selected_classes_global,
            image_meta_state,
            config_name_state,
        ]
        for register in (
            label_mode.change,
            show_boxes.change,
            show_masks.change,
            show_polygons.change,
            show_points.change,
            show_confidence.change,
            polygon_opt_enable.input,
        ):
            register(
                fn=replot_all_filtered,
                inputs=_replot_inputs,
                outputs=[output_gallery],
            )
        polygon_opt_apply.click(
            fn=apply_polygon_replot_clicked,
            inputs=_replot_inputs,
            outputs=[output_gallery],
        )

        # Global Selection Change -> Replot
        selected_classes_global.change(
             fn=replot_all_filtered,
             inputs=_replot_inputs,
             outputs=[output_gallery],
        )

        def update_mask_processing(
            raw_results_dict,
            mask_opt_enable_in,
            mask_opt_steps_in,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_points_in,
            show_conf_in,
            polygon_opt_enable_in,
            polygon_opt_steps_in,
            input_type_in,
            class_selected_items_in,
            image_meta_in=None,
            config_name_in=None,
        ):
            if not raw_results_dict:
                return None, gr.update()

            updated_results = apply_mask_optimizations(
                raw_results_dict,
                mask_opt_enable_in,
                mask_opt_steps_in,
            )

            gallery = replot_all_filtered(
                updated_results,
                label_mode_in,
                show_boxes_in,
                show_masks_in,
                show_polygons_in,
                show_points_in,
                show_conf_in,
                polygon_opt_enable_in,
                polygon_opt_steps_in,
                input_type_in,
                class_selected_items_in,
                image_meta_in,
                config_name_in,
            )
            return updated_results, gallery

        def apply_mask_processing_clicked(
            raw_results_dict,
            mask_opt_enable_in,
            mask_opt_steps_in,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_points_in,
            show_conf_in,
            polygon_opt_enable_in,
            polygon_opt_steps_in,
            input_type_in,
            class_selected_items_in,
            image_meta_in=None,
            config_name_in=None,
        ):
            if not mask_opt_enable_in:
                gr.Warning("尚未啟用 Mask 優化：請先勾選「啟用 Mask 優化」。目前顯示未優化結果。")
            return update_mask_processing(
                raw_results_dict,
                mask_opt_enable_in,
                mask_opt_steps_in,
                label_mode_in,
                show_boxes_in,
                show_masks_in,
                show_polygons_in,
                show_points_in,
                show_conf_in,
                polygon_opt_enable_in,
                polygon_opt_steps_in,
                input_type_in,
                class_selected_items_in,
                image_meta_in,
                config_name_in,
            )

        # 啟用開關只回應手動切換（.input，即時重算）；
        # 步驟表格編輯改由「套用」按鈕觸發，避免每格編輯都重跑完整 mask 管線
        _mask_processing_inputs = [
            raw_results,
            mask_opt_enable,
            mask_opt_steps,
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_points,
            show_confidence,
            polygon_opt_enable,
            polygon_opt_steps,
            input_type,
            selected_classes_global,
            image_meta_state,
            config_name_state,
        ]
        mask_opt_enable.input(
            fn=update_mask_processing,
            inputs=_mask_processing_inputs,
            outputs=[last_results, output_gallery],
        )
        mask_opt_apply.click(
            fn=apply_mask_processing_clicked,
            inputs=_mask_processing_inputs,
            outputs=[last_results, output_gallery],
        )


        # ======== 全選 / 取消全選 ========
        # 「選擇全選」按鈕：同步更新 Checkbox 與即時重繪
        def select_all_action(choices):
            # Global=All, UI=All (if filter is empty, works out), Query=""
            return (choices or []), gr.update(value=choices or []), gr.update(value="")

        def clear_all_action():
            return [], gr.update(value=[]), gr.update(value="")

        select_all_btn.click(
            fn=select_all_action,
            inputs=[class_choices_state],
            outputs=[selected_classes_global, class_selector, class_filter_query],
        )

        clear_all_btn.click(
            fn=clear_all_action,
            inputs=[],
            outputs=[selected_classes_global, class_selector, class_filter_query],
        )




        # ======== 匯出 JSON ========
        def export_json_click(
            last_results_dict,
            class_selected_items_in,
            image_meta,
            polygon_opt_enable_in,
            polygon_opt_steps_in,
        ):
            # 僅支援影像模式（因影片逐幀 polygon 通常會很大）
            if not last_results_dict or not image_meta:
                return []

            allowed_ids = resolve_allowed_ids(class_selected_items_in)

            files = export_results_cache(
                last_results_dict,
                image_info=image_meta,
                out_dir=None,
                allowed_class_ids=allowed_ids,
                simplify_mode="none",
                simplify_eps_coeff=1.0,
                polygon_opt_steps=parse_polygon_steps(polygon_opt_steps_in) if polygon_opt_enable_in else [],
            )
            return files

        export_btn.click(
            fn=export_json_click,
            inputs=[
                last_results,
                selected_classes_global,
                image_meta_state,
                polygon_opt_enable,
                polygon_opt_steps,
            ],
            outputs=[export_files],
        )

        # ======== 匯出 / 匯入 Config ========
        def export_ui_config(
            label_mode_in, show_boxes_in, show_masks_in, show_polygons_in, show_points_in, show_conf_in,
            selected_classes_in,
            mask_opt_enable_in, mask_opt_steps_in,
            polygon_opt_enable_in, polygon_opt_steps_in,
        ):
            config = {
                "display": {
                    "label_mode": label_mode_in,
                    "show_boxes": show_boxes_in,
                    "show_masks": show_masks_in,
                    "show_polygons": show_polygons_in,
                    "show_points": show_points_in,
                    "show_confidence": show_conf_in,
                },
                "class_filter": selected_classes_in,
                "mask_optimizations": {
                    "enabled": mask_opt_enable_in,
                    "steps": mask_opt_steps_in.values.tolist() if hasattr(mask_opt_steps_in, "values") else list(mask_opt_steps_in) if mask_opt_steps_in is not None else [],
                },
                "polygon_optimizations": {
                    "enabled": polygon_opt_enable_in,
                    "steps": polygon_opt_steps_in.values.tolist() if hasattr(polygon_opt_steps_in, "values") else list(polygon_opt_steps_in) if polygon_opt_steps_in is not None else [],
                }
            }
            # 以日期時間命名，避免 NamedTemporaryFile 的亂碼檔名
            config_stem = f"config_{datetime.datetime.now():%Y%m%d_%H%M%S}"
            out_dir = tempfile.mkdtemp(prefix="yolo_config_")
            out_path = os.path.join(out_dir, f"{config_stem}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return out_path, config_stem

        export_config_btn.click(
            fn=export_ui_config,
            inputs=[
                label_mode, show_boxes, show_masks, show_polygons, show_points, show_confidence,
                selected_classes_global,
                mask_opt_enable, mask_opt_steps,
                polygon_opt_enable, polygon_opt_steps
            ],
            outputs=[export_config_file, config_name_state]
        )

        def import_ui_config(file_path, current_class_choices):
            if not file_path:
                return (gr.update(),) * 15

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                return (gr.update(),) * 15
                
            display = config.get("display", {})
            label_mode_v = display.get("label_mode", ["顯示 class name"])
            # 舊版 config 為單選字串 → 轉為複選清單
            if isinstance(label_mode_v, str):
                label_mode_v = [] if label_mode_v == "隱藏" else [label_mode_v]
            show_boxes_v = display.get("show_boxes", True)
            show_masks_v = display.get("show_masks", True)
            show_polygons_v = display.get("show_polygons", True)
            show_points_v = display.get("show_points", True)
            show_conf_v = display.get("show_confidence", True)
            
            classes_v = config.get("class_filter", [])
            mask_opt = config.get("mask_optimizations", {})
            mask_opt_en = mask_opt.get("enabled", False)
            mask_opt_st = mask_opt.get("steps", [])
            
            polygon_opt = config.get("polygon_optimizations", {})
            polygon_opt_en = polygon_opt.get("enabled", False)
            polygon_opt_st = polygon_opt.get("steps", [])
            polygon_opt_st = _pad_polygon_rows(polygon_opt_st)

            valid_set = set(current_class_choices or [])
            filtered_ui_classes = [c for c in classes_v if c in valid_set]

            # 記錄配置名稱（檔名去副檔名），供下載推論圖檔命名
            config_stem = os.path.splitext(os.path.basename(str(file_path)))[0]

            return (
                gr.update(value=label_mode_v),
                gr.update(value=show_boxes_v),
                gr.update(value=show_masks_v),
                gr.update(value=show_polygons_v),
                gr.update(value=show_points_v),
                gr.update(value=show_conf_v),
                classes_v,
                gr.update(value=filtered_ui_classes),
                gr.update(value=mask_opt_en),
                gr.update(value=mask_opt_st),
                gr.update(value=polygon_opt_en),
                gr.update(value=polygon_opt_st),
                classes_v,
                classes_v,
                config_stem,
            )

        import_config_file.upload(
            fn=import_ui_config,
            inputs=[import_config_file, class_choices_state],
            outputs=[
                label_mode, show_boxes, show_masks, show_polygons, show_points, show_confidence,
                selected_classes_global, class_selector,
                mask_opt_enable, mask_opt_steps,
                polygon_opt_enable, polygon_opt_steps,
                step_selected_classes_global, polygon_step_selected_classes_global,
                config_name_state,
            ]
        )

    return demo


def run_cli(args):
    from app_utils.inference_optimizations import apply_mask_optimizations_to_result
    from app_utils.cli_config import load_inference_config

    # === 1+2. 載入 config（不存在時使用預設值，優化全關） ===
    cfg = load_inference_config(args.config)
    label_mode = cfg["label_mode"]
    show_boxes = cfg["show_boxes"]
    show_masks = cfg["show_masks"]
    show_polygons = cfg["show_polygons"]
    show_points = cfg["show_points"]
    show_conf = cfg["show_conf"]
    allowed_class_ids = cfg["allowed_class_ids"]
    mask_opt_en = cfg["mask_opt_enabled"]
    mask_steps_parsed = cfg["mask_steps_parsed"]
    polygon_steps_parsed = cfg["polygon_steps_parsed"]

    models = [m.strip() for m in args.models.split(",")]
    
    # 透過共用模型快取預先載入，避免在迴圈中重複 instantiate 導致 CUDA Out Of Memory
    from app_utils.model_cache import get_model
    loaded_models = {mid: get_model(mid) for mid in models}
    
    input_path = args.input
    if os.path.isdir(input_path):
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'):
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        # Windows case-insensitive deduplication
        image_files = sorted({os.path.abspath(f) for f in image_files})
    else:
        image_files = [input_path]
        
    os.makedirs(args.output, exist_ok=True)
    
    try:
        from tqdm import tqdm
        progress_bar = tqdm(image_files, desc="Batch Processing", unit="img")
    except ImportError:
        progress_bar = image_files

    device_val = None if args.device == "auto" else args.device
    sanitized_model_names = {
        mid: re.sub(r'[^a-zA-Z0-9_\-]', '_', os.path.splitext(os.path.basename(mid))[0])
        for mid in models
    }


    for img_path in progress_bar:
        fname = os.path.basename(img_path)
        if not hasattr(progress_bar, 'update'):
            print(f"Processing {fname}...")
        else:
            progress_bar.set_postfix({"file": fname})
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        image_info = {"file_name": fname, "width": img_w, "height": img_h}
        
        results_cache = {}
        for mid in models:
            model = loaded_models[mid]
            # 直接餵入已讀取的影像，避免 predict 再從磁碟讀一次
            predict_kwargs = {"source": img, "imgsz": args.imgsz, "conf": args.conf}
            if device_val:
                predict_kwargs["device"] = device_val

            # 移到 CPU：避免遮罩上色在 GPU 配置大張量導致 OOM（密集場景）
            results = [r.cpu() for r in model.predict(**predict_kwargs)]

            if mask_opt_en and mask_steps_parsed and results:
                results[0] = apply_mask_optimizations_to_result(results[0], mask_opt_en, mask_steps_parsed)

            final_bgr = annotate_from_results(
                results[0], label_mode, show_boxes, show_masks, show_polygons, show_points, show_conf,
                "none", 1.0, allowed_class_ids, polygon_steps_parsed
            )

            results_cache[mid] = results

            if args.save_img:
                base_name = os.path.splitext(fname)[0]
                out_img_path = os.path.join(args.output, f"{base_name}__{sanitized_model_names[mid]}.jpg")
                cv2.imwrite(out_img_path, final_bgr)
        
        export_results_cache(
            results_cache,
            image_info,
            out_dir=args.output,
            allowed_class_ids=allowed_class_ids,
            simplify_mode="none",
            simplify_eps_coeff=1.0,
            polygon_opt_steps=polygon_steps_parsed
        )
    print("Batch processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 Inference App & CLI")
    parser.add_argument("-i", "--input", type=str, help="Input image or directory for CLI mode")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory for JSON/images")
    parser.add_argument("-m", "--models", type=str, default="yolov12m.pt", help="Comma-separated model names/paths")
    parser.add_argument("-c", "--config", type=str, required=False, help="Path to exported config JSON")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--save-img", action="store_true", default=True, help="Save annotated images in CLI mode")
    parser.add_argument("--no-save-img", dest="save_img", action="store_false", help="Do not save annotated images")

    args, unknown = parser.parse_known_args()
    if args.input:
        run_cli(args)
    else:
        app().launch()
