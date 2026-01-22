import os
from typing import List, Optional

import gradio as gr

from app_utils.config import MAX_MODELS
from app_utils.model_registry import load_model_choices, persist_model_choices
from app_utils.inference import (
    names_to_choice_list,
    parse_selected_to_ids,
    annotate_from_results,
    yolov12_multi_inference_image,
    yolov12_multi_inference_video,
    yolov12_inference_for_examples,
)
from app_utils.export_utils import export_results_cache


def app():
    with gr.Blocks() as demo:
        # === 初始模型清單（預設 + 已儲存自訂） ===
        initial_choices, initial_saved_custom = load_model_choices()

        with gr.Row():
            # ======================= 左側：輸入與控制面板 =======================
            with gr.Column():
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
                device_in = gr.Radio(
                    choices=["0", "1"],
                    value="0",
                    label="Device",
                    interactive=True,
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

                label_mode = gr.Radio(
                    choices=["隱藏", "顯示 class id", "顯示 class name"],
                    value="顯示 class name",
                    label="標籤模式",
                )
                show_boxes = gr.Checkbox(value=True, label="顯示 bbox 外框")
                show_masks = gr.Checkbox(value=True, label="顯示 segmentation 遮罩")
                show_polygons = gr.Checkbox(value=True, label="顯示 polygon 邊界")  # ★ 新增
                show_confidence = gr.Checkbox(value=True, label="顯示信心值 (conf)")
                split_components = gr.Checkbox(value=False, label="分離連通域")  # ★ 新增

                yolov12_infer = gr.Button(value="Detect Objects (Run)")

                # 匯出 JSON（polygon）
                export_btn = gr.Button(
                    value="Export JSON (Polygons)",
                    variant="primary",
                )
                export_files = gr.Files(label="Exported JSON Files")

                # 類別篩選
                gr.Markdown("### 類別篩選（預設全選）")
                with gr.Row():
                    class_selector = gr.CheckboxGroup(
                        label="類別（ID: 名稱）",
                        choices=[],
                        value=[],
                        interactive=True,
                    )
                with gr.Row():
                    select_all_btn = gr.Button(value="選擇全選", variant="secondary")
                    clear_all_btn = gr.Button(value="取消全選", variant="secondary")

                # 保留目前 choices 狀態（避免僅從元件讀不到 choices）
                class_choices_state = gr.State(value=[])
                # 保存當前影像中繼資訊（檔名、寬高）
                image_meta_state = gr.State(value=None)

            # ======================= 右側：輸出 =======================
            with gr.Column():
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
            input_type_in,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in, 
            show_conf_in,
            saved_models_in,
            class_selected_items_in,
            class_choices_in,
            device_in,
            split_components_in,
        ):
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
                    gr.update(choices=initial_choices, value=[]),  # model_ids
                    saved_models_in,  # saved_models_state
                    gr.update(),  # class_selector
                    class_choices_in or [],  # class_choices_state
                    None,  # image_meta_state
                )

            # 2) 持久化自訂模型選項
            new_choices, new_saved = persist_model_choices(saved_models_in, mids)

            # 3) 類別篩選條件計算
            parsed_selected_ids = parse_selected_to_ids(class_selected_items_in)
            if class_selected_items_in is None:
                allowed_ids: Optional[List[int]] = None  # 不過濾
            elif len(class_selected_items_in) == 0:
                allowed_ids = []  # 全部隱藏
            else:
                allowed_ids = parsed_selected_ids

            # 4) Image 模式
            if input_type_in == "Image":
                if image_in is None:
                    # 沒有圖可跑，直接回傳目前狀態
                    return (
                        gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        None,
                        gr.update(choices=new_choices, value=mids),
                        new_saved,
                        gr.update(),  # class_selector
                        class_choices_in or [],
                        None,
                    )

                # 4-1) 多模型推論
                gallery, results_cache = yolov12_multi_inference_image(
                    image_in,
                    mids,
                    image_size_in,
                    conf_th_in,
                    label_mode_in,
                    show_boxes_in,
                    show_masks_in,
                    show_polygons_in, 
                    show_conf_in,
                    allowed_class_ids=allowed_ids,
                    device=device_in,
                    split_components=split_components_in,
                )

                # 4-2) 從第一個結果建立類別選單
                try:
                    first_result = next(iter(results_cache.values()))[0]
                    names = getattr(first_result, "names", {}) or {}
                except Exception:
                    names = {}

                if names:
                    class_choices_new, _all_ids = names_to_choice_list(names)
                else:
                    class_choices_new = []

                # 若是第一次推論（或尚未選擇任何類別），預設「全選」
                if not class_selected_items_in and class_choices_new:
                    class_selected_items_out = class_choices_new
                else:
                    # 使用既有勾選（但要過濾掉已不存在的項目）
                    valid_set = set(class_choices_new)
                    class_selected_items_out = [
                        c for c in (class_selected_items_in or []) if c in valid_set
                    ]

                class_selector_update = gr.update(
                    choices=class_choices_new,
                    value=class_selected_items_out,
                )

                # 4-3) 構建 image meta（檔名、寬高）
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
                    gr.update(choices=new_choices, value=mids),  # model_ids
                    new_saved,  # saved_models_state
                    class_selector_update,
                    class_choices_new,
                    image_meta,  # image_meta_state
                )

            # 5) Video 模式
            else:
                if video_in is None:
                    return (
                        gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        None,
                        gr.update(choices=new_choices, value=mids),
                        new_saved,
                        gr.update(),  # class_selector
                        class_choices_in or [],
                        None,
                    )

                outs = yolov12_multi_inference_video(
                    video_in,
                    mids,
                    image_size_in,
                    conf_th_in,
                    label_mode_in,
                    show_boxes_in,
                    show_masks_in,
                    show_polygons_in, 
                    show_conf_in,
                    allowed_class_ids=allowed_ids,
                    device=device_in,
                    split_components=split_components_in,
                )

                video_updates = [gr.update(value=None)] * 5
                for idx, (mid, out_path) in enumerate(outs[:5]):
                    video_updates[idx] = gr.update(value=out_path, label=str(mid))

                return (
                    gr.update(),  # output_gallery
                    video_updates[0],
                    video_updates[1],
                    video_updates[2],
                    video_updates[3],
                    video_updates[4],
                    None,  # last_results（影片不快取）
                    gr.update(choices=new_choices, value=mids),
                    new_saved,
                    gr.update(),  # class_selector：維持原樣
                    class_choices_in or [],
                    None,  # image_meta_state（影片無需）
                )

        yolov12_infer.click(
            fn=run_inference,
            inputs=[
                image,
                video,
                model_ids,
                image_size,
                conf_threshold,
                input_type,
                label_mode,
                show_boxes,
                show_masks,
                show_polygons,
                show_confidence,
                saved_models_state,
                class_selector,
                class_choices_state,
                device_in,
                split_components,
            ],
            outputs=[
                output_gallery,
                v1,
                v2,
                v3,
                v4,
                v5,
                last_results,
                model_ids,
                saved_models_state,
                class_selector,
                class_choices_state,
                image_meta_state,
            ],
        )

        # ======== 即時重繪（只針對 Image 模式） ========
        def replot_all_filtered(
            last_results_dict,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_conf_in,
            input_type_in,
            class_selected_items_in,
            split_components_in,
        ):
            if input_type_in != "Image" or not last_results_dict:
                return gr.update()

            # 解析類別
            selected_ids = parse_selected_to_ids(class_selected_items_in)
            if class_selected_items_in is None:
                allowed_ids = None
            elif class_selected_items_in == []:
                allowed_ids = []
            else:
                allowed_ids = selected_ids

            gallery = []
            for mid, results in last_results_dict.items():
                annotated_bgr = annotate_from_results(
                    results[0],
                    label_mode_in,
                    show_boxes_in,
                    show_masks_in,
                    show_polygons_in,
                    show_conf_in,
                    allowed_ids,
                    split_components_in,
                )
                gallery.append((annotated_bgr[:, :, ::-1], mid))  # BGR -> RGB

            return gallery

        # 標籤模式/框/遮罩/polygon/信心值/連通域 改變時即時重繪
        for ctrl in (label_mode, show_boxes, show_masks, show_polygons, show_confidence, split_components):
            ctrl.change(
                fn=replot_all_filtered,
                inputs=[
                    last_results,
                    label_mode,
                    show_boxes,
                    show_masks,
                    show_polygons,
                    show_confidence,
                    input_type,
                    class_selector,
                    split_components,
                ],
                outputs=[output_gallery],
            )

        # 類別選取改變時即時重繪
        class_selector.change(
            fn=replot_all_filtered,
            inputs=[
                last_results,
                label_mode,
                show_boxes,
                show_masks,
                show_polygons,
                show_confidence,
                input_type,
                class_selector,
                split_components,
            ],
            outputs=[output_gallery],
        )


        # ======== 全選 / 取消全選 ========
        # 「選擇全選」按鈕：同步更新 Checkbox 與即時重繪
        def select_all_and_replot(
            class_choices_in,
            last_results_dict,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_conf_in,
            input_type_in,
            split_components_in,
        ):
            # 將值設為目前 choices（全選）
            update_component = gr.update(value=class_choices_in or [])
            # 重繪
            gallery = replot_all_filtered(
                last_results_dict,
                label_mode_in,
                show_boxes_in,
                show_masks_in,
                show_polygons_in,
                show_conf_in,
                input_type_in,
                class_choices_in or [],
                split_components_in,
            )
            return update_component, gallery

        select_all_btn.click(
            fn=select_all_and_replot,
            inputs=[
                class_choices_state,
                last_results,
                label_mode,
                show_boxes,
                show_masks,
                show_polygons,
                show_confidence,
                input_type,
                split_components,
            ],
            outputs=[class_selector, output_gallery],
        )

        # 「取消全選」按鈕：清空並即時重繪（不顯示任何類別）
        def clear_all_and_replot(
            last_results_dict,
            label_mode_in,
            show_boxes_in,
            show_masks_in,
            show_polygons_in,
            show_conf_in,
            input_type_in,
            split_components_in,
        ):
            update_component = gr.update(value=[])
            gallery = replot_all_filtered(
                last_results_dict,
                label_mode_in,
                show_boxes_in,
                show_masks_in,
                show_polygons_in,
                show_conf_in,
                input_type_in,
                [],
                split_components_in,
            )
            return update_component, gallery

        clear_all_btn.click(
            fn=clear_all_and_replot,
            inputs=[
                last_results,
                label_mode,
                show_boxes,
                show_masks,
                show_polygons,
                show_confidence,
                input_type,
                split_components,
            ],
            outputs=[class_selector, output_gallery],
        )




        # ======== 匯出 JSON ========
        def export_json_click(last_results_dict, class_selected_items_in, image_meta):
            # 僅支援影像模式（因影片逐幀 polygon 通常會很大）
            if not last_results_dict or not image_meta:
                return []

            selected_ids = parse_selected_to_ids(class_selected_items_in)
            if class_selected_items_in is None:
                allowed_ids = None   # 不過濾
            elif class_selected_items_in == []:
                allowed_ids = []     # 全部隱藏
            else:
                allowed_ids = selected_ids

            files = export_results_cache(
                last_results_dict,
                image_info=image_meta,
                out_dir=None,
                allowed_class_ids=allowed_ids,
            )
            return files

        export_btn.click(
            fn=export_json_click,
            inputs=[last_results, class_selector, image_meta_state],
            outputs=[export_files],
        )

        # ======== 範例（Examples） ========
        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    ["yolov12s.pt", "yolov12m.pt"],
                    640,
                    0.25,
                    "顯示 class name",
                    True,
                    True,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    ["yolov12x.pt", "yolov12l.pt"],
                    640,
                    0.25,
                    "顯示 class id",
                    True,
                    True,
                ],
            ],
            fn=yolov12_inference_for_examples,
            inputs=[
                image,
                model_ids,
                image_size,
                conf_threshold,
                label_mode,
                show_boxes,
                show_masks,
            ],
            outputs=[output_gallery],
        )

    return demo


if __name__ == "__main__":
    app().launch()
