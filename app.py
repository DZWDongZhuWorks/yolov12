# --------------------------------------------------------
# Multi-model comparison（維持原流程）+ 類別篩選 + 顯示信心值
# Based on your YOLOv12 app
# --------------------------------------------------------

import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as ucolors
import os, json, copy
from typing import List, Dict, Tuple, Any, Optional

DEFAULT_MODELS = [
    "yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt", "yolov12x.pt"
]
PERSIST_FILE = os.environ.get("YOLOv12_MODEL_MEMO_FILE", "model_choices.json")

# 多模型比較：最多同時比較幾個
MAX_MODELS = 5


def _load_model_choices():
    try:
        with open(PERSIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            saved = data.get("models", [])
    except Exception:
        saved = []
    seen = set()
    choices = []
    for x in DEFAULT_MODELS + saved:
        if x not in seen:
            seen.add(x)
            choices.append(x)
    return choices, saved


def _persist_model_choice(saved_list, model_id):
    if model_id and model_id not in DEFAULT_MODELS and model_id not in saved_list:
        saved_list.append(model_id)
        try:
            with open(PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump({"models": saved_list}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    seen = set()
    choices = []
    for x in DEFAULT_MODELS + saved_list:
        if x not in seen:
            seen.add(x)
            choices.append(x)
    return choices, saved_list


def _persist_model_choices(saved_list, model_ids: List[str]):
    """一次處理多個自訂模型的持久化與下拉清單更新"""
    choices, saved = None, saved_list
    for mid in (model_ids or []):
        choices, saved = _persist_model_choice(saved, mid)
    if choices is None:
        # 沒有新模型，仍回傳現況
        seen = set()
        choices = []
        for x in DEFAULT_MODELS + saved:
            if x not in seen:
                seen.add(x)
                choices.append(x)
    return choices, saved


# ======= 類別篩選相關工具 =======

def _names_to_choice_list(names: Dict[Any, str]) -> Tuple[List[str], List[int]]:
    """
    將 {id: name} 轉成 ["0: person", "1: bicycle", ...] 與對應的 id 清單（已排序）
    """
    pairs = []
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


def _parse_selected_to_ids(selected_items: Optional[List[str]]) -> Optional[List[int]]:
    """
    由 CheckboxGroup 的值（如 ["0: person", "2: car"]）解析出 [0, 2]
    若為 None 或空清單，回傳空清單（代表不顯示任何類別）
    """
    if not selected_items:
        return []
    out = []
    for s in selected_items:
        try:
            cid = int(str(s).split(":")[0].strip())
            out.append(cid)
        except Exception:
            pass
    return out


def _filter_result_by_classes(result, allowed_class_ids: Optional[List[int]]):
    """
    回傳一個淺複製（shallow copy）的 result，僅保留指定類別的 boxes/masks。
    allowed_class_ids 為 None 代表「不過濾（全顯示）」；為空清單 [] 代表「全部隱藏」。
    """
    if allowed_class_ids is None:
        # 不過濾
        return result

    r = copy.copy(result)
    keep_idx = []

    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        cls_arr = result.boxes.cls.cpu().numpy().astype(int)
        keep_idx = [i for i, cid in enumerate(cls_arr) if cid in set(allowed_class_ids)]
        r.boxes = result.boxes[keep_idx] if len(keep_idx) > 0 else None
    else:
        r.boxes = None

    # masks 與 boxes 順序一致，沿用 keep_idx
    if hasattr(result, "masks") and result.masks is not None:
        r.masks = result.masks[keep_idx] if len(keep_idx) > 0 else None
    else:
        r.masks = None

    return r


def _annotate_from_results(
    result,
    label_mode: str,
    show_boxes: bool,
    show_masks: bool,
    show_confidence: bool,
    allowed_class_ids: Optional[List[int]] = None
):
    """
    使用 Ultralytics 的 plot 畫基礎層（boxes/masks），labels 交由我們自行繪製。
    回傳 BGR 影像。若 show_confidence=True，於標籤文字後附上 conf（小數點兩位）。
    """
    filtered = _filter_result_by_classes(result, allowed_class_ids)
    base = filtered.plot(labels=False, boxes=show_boxes, masks=show_masks)

    if label_mode == "隱藏":
        return base
    if not hasattr(filtered, "boxes") or filtered.boxes is None or len(filtered.boxes) == 0:
        return base

    names = getattr(result, "names", None) or {}  # 使用原 result 的 names
    xyxy = filtered.boxes.xyxy.cpu().numpy()
    cls_arr = filtered.boxes.cls.cpu().numpy().astype(int)

    # 可能沒有 conf（理論上 Ultralytics 皆有），保險起見處理 None
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
        cv2.putText(base, base_text, (x1 + pad, y1 - pad), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return base


def _infer_image_single(model_id: str, image, image_size: int, conf_threshold: float,
                        label_mode: str, show_boxes: bool, show_masks: bool,
                        show_confidence: bool,
                        allowed_class_ids: Optional[List[int]]):
    model = YOLO(model_id)
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    annotated_bgr = _annotate_from_results(results[0], label_mode, show_boxes, show_masks, show_confidence, allowed_class_ids)
    return annotated_bgr[:, :, ::-1], results  # RGB, results


def _infer_video_single(model_id: str, video_path: str, image_size: int, conf_threshold: float,
                        label_mode: str, show_boxes: bool, show_masks: bool,
                        show_confidence: bool,
                        allowed_class_ids: Optional[List[int]]):
    model = YOLO(model_id)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".webm")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'vp80'),
                          fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
        annotated_bgr = _annotate_from_results(results[0], label_mode, show_boxes, show_masks, show_confidence, allowed_class_ids)
        out.write(annotated_bgr)

    cap.release()
    out.release()
    return out_path


def yolov12_multi_inference_image(image, model_ids: List[str], image_size, conf_threshold,
                                  label_mode, show_boxes, show_masks, show_confidence,
                                  allowed_class_ids: Optional[List[int]]):
    """對同一張影像以多個模型推理，回傳 [(img, caption), ...] 與 results 快取 dict"""
    gallery_items = []
    results_cache: Dict[str, Any] = {}
    for mid in model_ids:
        img_rgb, results = _infer_image_single(mid, image, image_size, conf_threshold,
                                               label_mode, show_boxes, show_masks, show_confidence,
                                               allowed_class_ids)
        gallery_items.append((img_rgb, mid))
        results_cache[mid] = results
    return gallery_items, results_cache


def yolov12_multi_inference_video(video, model_ids: List[str], image_size, conf_threshold,
                                  label_mode, show_boxes, show_masks, show_confidence,
                                  allowed_class_ids: Optional[List[int]]):
    """對同一支影片以多個模型推理，回傳每個模型對應的輸出影片路徑（最多 MAX_MODELS 個）"""
    # 先把上傳/路徑存到臨時檔（Gradio 有時是 file-like 或 path，統一處理）
    src_path = tempfile.mktemp(suffix=".webm")
    with open(src_path, "wb") as f:
        with open(video, "rb") as g:
            f.write(g.read())

    outs = []
    for mid in model_ids[:MAX_MODELS]:
        out_path = _infer_video_single(mid, src_path, image_size, conf_threshold,
                                       label_mode, show_boxes, show_masks, show_confidence,
                                       allowed_class_ids)
        outs.append((mid, out_path))
    return outs


def yolov12_inference_for_examples(image, model_list, image_size, conf_threshold,
                                   label_mode, show_boxes, show_masks):
    # Examples 也支援多模型（可給 1 或多）
    if isinstance(model_list, str):
        model_ids = [model_list]
    else:
        model_ids = model_list or []
    if not model_ids:
        return []
    # 範例預設顯示全部類別 + 顯示信心值
    gallery, _ = yolov12_multi_inference_image(
        image, model_ids, image_size, conf_threshold,
        label_mode, show_boxes, show_masks, True,
        allowed_class_ids=None
    )
    return gallery


def app():
    with gr.Blocks():
        initial_choices, initial_saved_custom = _load_model_choices()

        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(choices=["Image", "Video"], value="Image", label="Input Type")

                saved_models_state = gr.State(value=initial_saved_custom)

                # 改為多選 Dropdown（支援自訂）
                model_ids = gr.Dropdown(
                    label="Models (多選比較，最多 5)",
                    choices=initial_choices,
                    value=["yolov12m.pt"],
                    allow_custom_value=True,
                    multiselect=True,
                )

                image_size = gr.Slider(label="Image Size", minimum=320, maximum=2560, step=32, value=640)
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.25)

                label_mode = gr.Radio(
                    choices=["隱藏", "顯示 class id", "顯示 class name"],
                    value="顯示 class name",
                    label="標籤模式"
                )
                show_boxes = gr.Checkbox(value=True, label="顯示 bbox 外框")
                show_masks = gr.Checkbox(value=True, label="顯示 mask 封遮")
                show_confidence = gr.Checkbox(value=True, label="顯示信心值 (conf)")
                yolov12_infer = gr.Button(value="Detect Objects (Run)")
                # ===== 類別篩選（ID:Name），預設全選 + 快捷按鈕 =====
                gr.Markdown("### 類別篩選（預設全選）")
                with gr.Row():
                    class_selector = gr.CheckboxGroup(
                        label="類別（ID: 名稱）",
                        choices=[],   # 由第一次推論後依模型自動填入
                        value=[],     # 預設會設為全選
                        interactive=True
                    )
                with gr.Row():
                    select_all_btn = gr.Button(value="選擇全選", variant="secondary")
                    clear_all_btn = gr.Button(value="取消全選", variant="secondary")

                # 保留目前 choices 狀態（避免從元件讀不到 choices）
                class_choices_state = gr.State(value=[])

                

            with gr.Column():
                # 影像：用 Gallery 並排
                output_gallery = gr.Gallery(
                    label="Annotated Images（多模型比較）",
                    columns=2,
                    preview=True,
                    visible=True
                )
                # 影片：預先放 5 路輸出
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

        def update_visibility(input_type_val):
            image_v = gr.update(visible=True) if input_type_val == "Image" else gr.update(visible=False)
            video_v = gr.update(visible=False) if input_type_val == "Image" else gr.update(visible=True)
            gallery_v = gr.update(visible=True) if input_type_val == "Image" else gr.update(visible=False)
            video_group_v = gr.update(visible=False) if input_type_val == "Image" else gr.update(visible=True)
            return image_v, video_v, gallery_v, video_group_v

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_gallery, video_group],
        )

        def run_inference(image_in, video_in, model_ids_in, image_size_in, conf_th_in, input_type_in,
                          label_mode_in, show_boxes_in, show_masks_in, show_conf_in,
                          saved_models_in,
                          class_selected_items_in, class_choices_in):
            # 正規化模型清單
            if isinstance(model_ids_in, str):
                mids = [model_ids_in]
            else:
                mids = (model_ids_in or [])[:MAX_MODELS]

            # 若空清單，直接不動
            if not mids:
                return (
                    gr.update(),  # gallery
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),  # videos v1~v5
                    None,  # last_results
                    gr.update(choices=initial_choices, value=[]),  # model_ids
                    saved_models_in,
                    gr.update(choices=[], value=[]),  # class_selector
                    []  # class_choices_state
                )

            # 更新自訂模型持久化
            new_choices, new_saved = _persist_model_choices(saved_models_in, mids)

            # 解析目前的類別選取（若尚未建立，代表預設全選 => 傳 None 給底層表示「不過濾」）
            parsed_selected_ids = _parse_selected_to_ids(class_selected_items_in)
            allowed_ids = None if class_selected_items_in in (None, []) else parsed_selected_ids

            if input_type_in == "Image":
                gallery, results_cache = yolov12_multi_inference_image(
                    image_in, mids, image_size_in, conf_th_in, label_mode_in, show_boxes_in, show_masks_in, show_conf_in,
                    allowed_class_ids=allowed_ids
                )

                # 從第一個模型的 names 建立類別清單；若與舊的 choices 不同，更新並預設全選
                try:
                    first_result = next(iter(results_cache.values()))[0]
                    names = getattr(first_result, "names", {}) or {}
                except Exception:
                    names = {}

                new_class_choices, _all_ids = _names_to_choice_list(names)
                # 若之前尚未有 choices，或模型更換導致 choices 改變，預設為全選
                if not class_choices_in or set(class_choices_in) != set(new_class_choices):
                    class_selector_update = gr.update(choices=new_class_choices, value=new_class_choices)
                    class_choices_state_new = new_class_choices
                else:
                    # 維持原本選取與 choices
                    keep_vals = [v for v in (class_selected_items_in or []) if v in new_class_choices]
                    if not keep_vals:  # 若原本的選取在新 choices 中皆不存在，改為全選
                        keep_vals = new_class_choices
                    class_selector_update = gr.update(choices=new_class_choices, value=keep_vals)
                    class_choices_state_new = new_class_choices

                return (
                    gallery,
                    gr.update(value=None, label="Model #1"),  # v1~v5 清空
                    gr.update(value=None, label="Model #2"),
                    gr.update(value=None, label="Model #3"),
                    gr.update(value=None, label="Model #4"),
                    gr.update(value=None, label="Model #5"),
                    results_cache,
                    gr.update(choices=new_choices, value=mids),
                    new_saved,
                    class_selector_update,
                    class_choices_state_new
                )
            else:
                outs = yolov12_multi_inference_video(
                    video_in, mids, image_size_in, conf_th_in, label_mode_in, show_boxes_in, show_masks_in, show_conf_in,
                    allowed_class_ids=allowed_ids
                )
                # 影片模式：類別清單仍以第一個模型動態刷新（如可讀）
                try:
                    # 粗略載一張影像（第一幀）取 names，不強制
                    tmp_cap = cv2.VideoCapture(video_in)
                    ret, frame = tmp_cap.read()
                    tmp_cap.release()
                    names = {}
                    if ret:
                        tmp_model = YOLO(mids[0])
                        tmp_res = tmp_model.predict(source=frame, imgsz=image_size_in, conf=conf_th_in)
                        names = getattr(tmp_res[0], "names", {}) or {}
                except Exception:
                    names = {}

                new_class_choices, _ = _names_to_choice_list(names)
                if not class_choices_in or set(class_choices_in) != set(new_class_choices):
                    class_selector_update = gr.update(choices=new_class_choices, value=new_class_choices)
                    class_choices_state_new = new_class_choices
                else:
                    keep_vals = [v for v in (class_selected_items_in or []) if v in new_class_choices]
                    if not keep_vals:
                        keep_vals = new_class_choices
                    class_selector_update = gr.update(choices=new_class_choices, value=keep_vals)
                    class_choices_state_new = new_class_choices

                # 依序填滿 v1~v5
                video_updates = []
                for idx in range(MAX_MODELS):
                    if idx < len(outs):
                        lbl, path = outs[idx][0], outs[idx][1]
                        video_updates.append(gr.update(value=path, label=lbl))
                    else:
                        video_updates.append(gr.update(value=None, label=f"Model #{idx+1}"))
                return (
                    gr.update(),  # gallery 不動
                    video_updates[0],
                    video_updates[1],
                    video_updates[2],
                    video_updates[3],
                    video_updates[4],
                    None,  # 視訊不快取 results
                    gr.update(choices=new_choices, value=mids),
                    new_saved,
                    class_selector_update,
                    class_choices_state_new
                )

        # 即時重繪（Image 模式用）：標籤/框/遮罩/類別/信心值改變就以 last_results 重畫每一張
        def replot_all_filtered(last_results_dict, label_mode_in, show_boxes_in, show_masks_in, show_conf_in, input_type_in, class_selected_items_in):
            if input_type_in != "Image" or not last_results_dict:
                return gr.update()
            # 解析類別
            selected_ids = _parse_selected_to_ids(class_selected_items_in)
            allowed_ids = None if class_selected_items_in in (None, []) else selected_ids
            gallery = []
            for mid, results in last_results_dict.items():
                annotated_bgr = _annotate_from_results(results[0], label_mode_in, show_boxes_in, show_masks_in, show_conf_in, allowed_ids)
                gallery.append((annotated_bgr[:, :, ::-1], mid))
            return gallery

        # 標籤模式/框/遮罩/信心值改變時即時重繪
        for ctrl in (label_mode, show_boxes, show_masks, show_confidence):
            ctrl.change(
                fn=replot_all_filtered,
                inputs=[last_results, label_mode, show_boxes, show_masks, show_confidence, input_type, class_selector],
                outputs=[output_gallery],
            )

        # 類別選取改變時即時重繪
        class_selector.change(
            fn=replot_all_filtered,
            inputs=[last_results, label_mode, show_boxes, show_masks, show_confidence, input_type, class_selector],
            outputs=[output_gallery],
        )

        # 「選擇全選」按鈕：同步更新 Checkbox 與即時重繪
        def select_all_and_replot(class_choices_in, last_results_dict, label_mode_in, show_boxes_in, show_masks_in, show_conf_in, input_type_in):
            # 將值設為目前 choices（全選）
            update_component = gr.update(value=class_choices_in)
            # 重繪
            gallery = replot_all_filtered(last_results_dict, label_mode_in, show_boxes_in, show_masks_in, show_conf_in, input_type_in, class_choices_in)
            return update_component, gallery

        select_all_btn.click(
            fn=select_all_and_replot,
            inputs=[class_choices_state, last_results, label_mode, show_boxes, show_masks, show_confidence, input_type],
            outputs=[class_selector, output_gallery],
        )

        # 「取消全選」按鈕：清空並即時重繪（不顯示任何類別）
        def clear_all_and_replot(last_results_dict, label_mode_in, show_boxes_in, show_masks_in, show_conf_in, input_type_in):
            update_component = gr.update(value=[])
            gallery = replot_all_filtered(last_results_dict, label_mode_in, show_boxes_in, show_masks_in, show_conf_in, input_type_in, [])
            return update_component, gallery

        clear_all_btn.click(
            fn=clear_all_and_replot,
            inputs=[last_results, label_mode, show_boxes, show_masks, show_confidence, input_type],
            outputs=[class_selector, output_gallery],
        )

        yolov12_infer.click(
            fn=run_inference,
            inputs=[
                image, video, model_ids, image_size, conf_threshold, input_type,
                label_mode, show_boxes, show_masks, show_confidence,
                saved_models_state,
                class_selector, class_choices_state
            ],
            outputs=[
                output_gallery,
                v1, v2, v3, v4, v5,
                last_results,
                model_ids,
                saved_models_state,
                class_selector,
                class_choices_state
            ],
        )

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
            inputs=[image, model_ids, image_size, conf_threshold, label_mode, show_boxes, show_masks],
            outputs=[output_gallery],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv12: Attention-Centric Real-Time Object Detectors — Multi-Model Comparison
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2502.12524' target='_blank'>arXiv</a> | <a href='https://github.com/sunsmarterjie/yolov12' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch()
