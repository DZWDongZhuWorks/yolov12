import os
from typing import List, Optional

import gradio as gr

from app_utils.config import MAX_MODELS
from app_utils.model_registry import load_model_choices, persist_model_choices
from app_utils.inference import (
    names_to_choice_list,
    parse_selected_to_ids,
    annotate_from_results,
    apply_mask_optimizations,
    yolov12_multi_inference_image,
    yolov12_multi_inference_video,
    get_model_names,
    parse_polygon_steps,
)
from app_utils.export_utils import export_results_cache

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

.step-table-wrap {
    max-height: 260px;
    overflow-y: auto;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 4px;
}

.step-table-wrap table td:nth-child(1),
.step-table-wrap table th:nth-child(1) {
    width: 80px;
    min-width: 80px;
}

.step-table-wrap table td:last-child,
.step-table-wrap table th:last-child {
    min-width: 220px;
    max-width: 320px;
    white-space: nowrap;
    overflow-x: auto;
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
                        image = gr.Image(type="pil", label="Image", visible=True)
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
                        label_mode = gr.Radio(
                            choices=["隱藏", "顯示 class id", "顯示 class name"],
                            value="顯示 class name",
                            label="標籤模式",
                        )
                        show_boxes = gr.Checkbox(value=True, label="顯示 bbox 外框")
                        show_masks = gr.Checkbox(value=True, label="顯示 segmentation 遮罩")
                        show_polygons = gr.Checkbox(value=True, label="顯示 polygon 邊界")
                        show_points = gr.Checkbox(value=False, label="顯示 polygon 點")
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
                            select_all_btn = gr.Button(value="選擇全選", variant="secondary")
                            clear_all_btn = gr.Button(value="取消全選", variant="secondary")

                    with gr.Tab("Mask 優化"):
                        gr.Markdown("### Mask 優化（先處理 mask 再生成 polygon）")
                        mask_opt_enable = gr.Checkbox(value=False, label="啟用 Mask 優化")
                        with gr.Row():
                            mask_opt_method = gr.Dropdown(
                                label="新增步驟",
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
                            mask_opt_add = gr.Button(value="加入步驟", variant="secondary")
                        with gr.Row():
                            step_morph_kernel = gr.Slider(
                                label="Morph Kernel (odd)",
                                minimum=1,
                                maximum=15,
                                step=2,
                                value=3,
                            )
                            step_blur_kernel = gr.Slider(
                                label="Blur Kernel (odd)",
                                minimum=1,
                                maximum=15,
                                step=2,
                                value=3,
                            )
                            step_blur_threshold = gr.Slider(
                                label="Blur Threshold",
                                minimum=0.1,
                                maximum=0.9,
                                step=0.05,
                                value=0.5,
                            )
                        with gr.Row():
                            step_min_component_area = gr.Slider(
                                label="Min Component Area",
                                minimum=0,
                                maximum=5000,
                                step=10,
                                value=0,
                            )
                            step_max_hole_area = gr.Slider(
                                label="Max Hole Area",
                                minimum=0,
                                maximum=5000,
                                step=10,
                                value=0,
                            )
                            step_merge_iou_threshold = gr.Slider(
                                label="Merge IoU Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.1,
                            )
                        step_class_filter_query = gr.Textbox(
                            label="類別查詢",
                            placeholder="輸入關鍵字或 class id",
                        )
                        with gr.Accordion("套用 Classes（全選=全部）", open=False):
                            step_class_filter = gr.CheckboxGroup(
                                label="套用 Classes（全選=全部）",
                                choices=[],
                                value=[],
                            )
                            with gr.Row():
                                step_select_all_btn = gr.Button(value="全部選取", variant="secondary")
                                step_clear_all_btn = gr.Button(value="全部取消", variant="secondary")
                        gr.Markdown(
                            "可直接編輯下表調整順序、次數與參數（classes 留空 = 全部類別）。"
                        )
                        with gr.Group(elem_classes=["step-table-wrap"]):
                            mask_opt_steps = gr.Dataframe(
                                headers=[
                                    "enabled",
                                    "step",
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
                                    "bool",
                                    "str",
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
                            )

                    with gr.Tab("Polygon 優化"):
                        gr.Markdown("### Polygon 優化")
                        polygon_opt_enable = gr.Checkbox(value=False, label="啟用 Polygon 優化")
                        with gr.Row():
                            polygon_opt_method = gr.Dropdown(
                                label="新增步驟",
                                choices=["convex_hull", "rdp", "visvalingam_whyatt"],
                                value="rdp",
                            )
                            polygon_opt_count = gr.Slider(
                                label="次數",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=1,
                            )
                            polygon_opt_add = gr.Button(value="加入步驟", variant="secondary")
                        polygon_step_eps_coeff = gr.Slider(
                            label="Step Epsilon Coefficient",
                            minimum=0.1,
                            maximum=5.0,
                            step=0.1,
                            value=1.0,
                        )
                        polygon_step_class_filter_query = gr.Textbox(
                            label="類別查詢",
                            placeholder="輸入關鍵字或 class id",
                        )
                        with gr.Accordion("Polygon 套用 Classes（全選=全部）", open=False):
                            polygon_step_class_filter = gr.CheckboxGroup(
                                label="套用 Classes（全選=全部）",
                                choices=[],
                                value=[],
                            )
                            with gr.Row():
                                polygon_step_select_all_btn = gr.Button(value="全部選取", variant="secondary")
                                polygon_step_clear_all_btn = gr.Button(value="全部取消", variant="secondary")
                        gr.Markdown(
                            "可直接編輯下表調整 Polygon 優化順序、次數與參數（classes 留空 = 全部類別）。"
                        )
                        with gr.Group(elem_classes=["step-table-wrap"]):
                            polygon_opt_steps = gr.Dataframe(
                                headers=["enabled", "step", "count", "eps_coeff", "classes"],
                                datatype=["bool", "str", "number", "number", "str"],
                                row_count=0,
                                col_count=(5, "fixed"),
                                wrap=True,
                                label="Polygon 優化流程",
                                type="array",
                            )

                # 保留目前 choices 狀態（避免僅從元件讀不到 choices）
                class_choices_state = gr.State(value=[])
                # 保存當前影像中繼資訊（檔名、寬高）
                image_meta_state = gr.State(value=None)

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
        ):
            if not results_dict:
                return []

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
                gallery.append((annotated_bgr[:, :, ::-1], mid))  # BGR -> RGB

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
        ):
            polygon_steps = parse_polygon_steps(polygon_opt_steps_in) if polygon_opt_enable_in else []
            # 1) 正規化模型清單（最多 MAX_MODELS 個）
            if isinstance(model_ids_in, str):
                mids: List[str] = [model_ids_in]
            else:
                mids = [m for m in (model_ids_in or []) if m]
            mids = mids[:MAX_MODELS]

            # 沒有選到任何模型：清空輸出並維持現有狀態
            if not mids:
                return (
                    gr.update(),  # output_gallery
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),  # v1~v5
                    None,  # last_results
                    None,  # raw_results
                    gr.update(choices=initial_choices, value=[]),  # model_ids
                    saved_models_in,  # saved_models_state
                    gr.update(),  # class_selector
                    class_choices_in or [],  # class_choices_state
                    None,  # image_meta_state
                    gr.update(choices=class_choices_in or [], value=[]),  # step_class_filter
                    gr.update(choices=class_choices_in or [], value=[]),  # polygon_step_class_filter
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=[]),
                    gr.update(value=[]),
                    gr.update(value=[]),
                )

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

            class_selector_update = gr.update(
                choices=class_choices_new,
                value=class_selected_items_out,
            )
            step_class_filter_update = gr.update(
                choices=class_choices_new,
                value=class_choices_new,
            )
            polygon_step_class_filter_update = gr.update(
                choices=class_choices_new,
                value=class_choices_new,
            )
            class_filter_query_update = gr.update(value="")
            step_filter_query_update = gr.update(value="")

            # 4) Image 模式
            if input_type_in == "Image":
                if image_in is None:
                    # 沒有圖可跑，直接回傳目前狀態
                    return (
                        gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        None,
                        None,
                        gr.update(choices=new_choices, value=mids),
                        new_saved,
                        class_selector_update,
                        class_choices_new,
                        None,
                        step_class_filter_update,
                        polygon_step_class_filter_update,
                        class_filter_query_update,
                        step_filter_query_update,
                        gr.update(value=class_selected_items_out),
                        gr.update(value=class_choices_new),
                        gr.update(value=class_choices_new),
                    )

                # 4-1) 多模型推論
                _, results_cache = yolov12_multi_inference_image(
                    image_in,
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
                )

                raw_results_cache = results_cache
                results_cache = apply_mask_optimizations(
                    results_cache,
                    mask_opt_enable_in,
                    mask_opt_steps_in,
                )

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
                )

                # 4-2) 構建 image meta（檔名、寬高）
                width = height = 0
                try:
                    if "first_result" not in locals():
                        first_result = next(iter(results_cache.values()))[0]
                    h, w = map(int, getattr(first_result, "orig_shape", (0, 0))[:2])
                    width, height = w, h
                except Exception:
                    width = height = 0

                fname = None
                # 優先從 PIL 物件上抓 filename/name/path
                for attr in ("filename", "name", "path"):
                    candidate = getattr(image_in, attr, None)
                    if candidate:
                        fname = os.path.basename(str(candidate))
                        break
                # 再退而求其次從 result.path
                if not fname:
                    try:
                        if "first_result" not in locals():
                            first_result = next(iter(results_cache.values()))[0]
                        rp = getattr(first_result, "path", None)
                        if rp:
                            fname = os.path.basename(str(rp))
                    except Exception:
                        fname = None
                if not fname:
                    fname = "uploaded_image.png"

                image_meta = {
                    "file_name": fname,
                    "width": int(width),
                    "height": int(height),
                }

                return (
                    gallery,
                    gr.update(value=None, label="Model #1"),
                    gr.update(value=None, label="Model #2"),
                    gr.update(value=None, label="Model #3"),
                    gr.update(value=None, label="Model #4"),
                    gr.update(value=None, label="Model #5"),
                    results_cache,  # last_results
                    raw_results_cache,  # raw_results
                    gr.update(choices=new_choices, value=mids),  # model_ids
                    new_saved,  # saved_models_state
                    class_selector_update,
                    class_choices_new,
                    image_meta,  # image_meta_state
                    step_class_filter_update,
                    polygon_step_class_filter_update,
                    class_filter_query_update,
                    step_filter_query_update,
                    gr.update(value=class_selected_items_out),  # Update Global Selection
                    gr.update(value=class_choices_new), # Update Step Global Selection
                    gr.update(value=class_choices_new), # Update Polygon Step Global Selection
                )

            # 5) Video 模式
            else:
                if video_in is None:
                    return (
                        gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        None,
                        None,
                        gr.update(choices=new_choices, value=mids),
                        new_saved,
                        class_selector_update,
                        class_choices_new,
                        None,
                        step_class_filter_update,
                        polygon_step_class_filter_update,
                        class_filter_query_update,
                        step_filter_query_update,
                        gr.update(value=class_selected_items_out),
                        gr.update(value=class_choices_new),
                        gr.update(value=class_choices_new),
                    )

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

                return (
                    gr.update(),  # output_gallery
                    video_updates[0],
                    video_updates[1],
                    video_updates[2],
                    video_updates[3],
                    video_updates[4],
                    None,  # last_results（影片不快取）
                    None,  # raw_results
                    gr.update(choices=new_choices, value=mids),
                    new_saved,
                    class_selector_update,
                    class_choices_new,
                    None,  # image_meta_state（影片無需）
                    step_class_filter_update,
                    polygon_step_class_filter_update,
                    class_filter_query_update,
                    step_filter_query_update,
                    gr.update(value=class_selected_items_out),
                    gr.update(value=class_choices_new),
                    gr.update(value=class_choices_new),
                )

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
        ):
            if isinstance(class_filter_in, (list, tuple, set)):
                # [Modified] Convert list to comma-joined string to avoid "['a', 'b']" stringification issue
                valid_items = [str(x) for x in class_filter_in if x is not None]
                class_filter_snapshot = ", ".join(valid_items)
            elif class_filter_in is None:
                class_filter_snapshot = None
            else:
                class_filter_snapshot = str(class_filter_in)
            if steps is None:
                rows = []
            elif hasattr(steps, "tolist"):
                rows = steps.tolist()
            elif isinstance(steps, list):
                rows = list(steps)
            else:
                rows = []
            rows.append(
                [
                    True,
                    method,
                    int(count),
                    morph_kernel_in,
                    blur_kernel_in,
                    blur_threshold_in,
                    min_component_area_in,
                    max_hole_area_in,
                    merge_iou_threshold_in,
                    class_filter_snapshot,
                ]
            )
            return rows

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
            ],
            outputs=[mask_opt_steps],
        )

        def add_polygon_step(
            steps,
            method,
            count,
            eps_coeff_in,
            class_filter_in,
        ):
            if isinstance(class_filter_in, (list, tuple, set)):
                valid_items = [str(x) for x in class_filter_in if x is not None]
                class_filter_snapshot = ", ".join(valid_items)
            elif class_filter_in is None:
                class_filter_snapshot = None
            else:
                class_filter_snapshot = str(class_filter_in)

            if steps is None:
                rows = []
            elif hasattr(steps, "tolist"):
                rows = steps.tolist()
            elif isinstance(steps, list):
                rows = list(steps)
            else:
                rows = []

            rows.append([True, method, int(count), eps_coeff_in, class_filter_snapshot])
            return rows

        polygon_opt_add.click(
            fn=add_polygon_step,
            inputs=[
                polygon_opt_steps,
                polygon_opt_method,
                polygon_opt_count,
                polygon_step_eps_coeff,
                polygon_step_class_filter,
            ],
            outputs=[polygon_opt_steps],
        )

        
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
            )

        # 標籤模式/框/遮罩/polygon/信心值 改變時即時重繪
        for ctrl in (
            label_mode,
            show_boxes,
            show_masks,
            show_polygons,
            show_points,
            show_confidence,
            polygon_opt_enable,
            polygon_opt_steps,
        ):
            ctrl.change(
                fn=replot_all_filtered,
                inputs=[
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
                ],
                outputs=[output_gallery],
            )
 
        # Global Selection Change -> Replot
        selected_classes_global.change(
             fn=replot_all_filtered,
             inputs=[
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
                selected_classes_global, # Use Global State
             ],
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
            )
            return updated_results, gallery

        for ctrl in (mask_opt_enable, mask_opt_steps):
            ctrl.change(
                fn=update_mask_processing,
                inputs=[
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
                ],
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

    return demo


if __name__ == "__main__":
    app().launch()
