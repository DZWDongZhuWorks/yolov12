# export_utils.py
import os, json, tempfile
from typing import Any, Dict, List, Optional
from .polygon_utils import build_objects_from_result

def _extract_objects(result, allowed_class_ids: Optional[List[int]], simplify_mode, simplify_eps_ratio):
    # 這裡就不自己拆 masks.xy / bbox 了
    # 想要使用凸包，就把 simplify_mode 設成 "convex_hull"
    objs = build_objects_from_result(
        result,
        allowed_class_ids=allowed_class_ids,
        simplify_mode=simplify_mode,
        simplify_eps_ratio=simplify_eps_ratio,
    )
    return objs

def _sanitize_filename(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return keep or "model"

def build_payload(result, simplify_mode, simplify_eps_ratio, model_name: str, image_info: Dict[str, Any],
                  allowed_class_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    將單一模型的一張影像推論結果整理成 JSON 結構
    """
    h = int(image_info.get("height"))
    w = int(image_info.get("width"))
    fname = str(image_info.get("file_name") or "uploaded_image")

    return {
        "image": {
            "file_name": fname,
            "width": w,
            "height": h
        },
        "model": {
            "name": model_name
        },
        "objects": _extract_objects(result, simplify_mode, simplify_eps_ratio, allowed_class_ids=allowed_class_ids)
    }

def export_results_cache(
    results_cache,
    image_info,
    out_dir=None,
    allowed_class_ids=None,
    simplify_mode: str = "convex_hull",
    simplify_eps_ratio: float = 0.01,
):
    """
    逐模型輸出成多個 JSON 檔。檔名格式：{image_base}__{model_name}.json
    回傳檔案路徑清單（可直接餵給 gr.Files）
    """
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="yolo_json_")

    base = os.path.splitext(os.path.basename(image_info.get("file_name") or "image"))[0]
    base = _sanitize_filename(base)

    filepaths = []
    for model_name, results in results_cache.items():
        model_stem = _sanitize_filename(os.path.splitext(os.path.basename(model_name))[0])
        payload = build_payload(results[0], simplify_mode, simplify_eps_ratio, model_name=model_name, image_info=image_info,
                                allowed_class_ids=allowed_class_ids)
        out_path = os.path.join(out_dir, f"{base}__{model_stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        filepaths.append(out_path)
    return filepaths
