# app_utils/polygon_comparison.py
"""對比不同 polygon 優化 step 的效果（點數、面積保留、與稠密輪廓的偏差）。

所有方法統一走 build_objects_from_result 的 polygon_opt_steps 路徑（production 路徑），
因此 rdp / convex_hull / smooth_spline / fit_lines_arcs 等新舊 step 都能同框比較。

CLI（需以套件模組執行）：
    python -m app_utils.polygon_comparison --weights best.pt --image a.jpg --classes 12,13 --out cmp_out
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from .polygon_utils import build_objects_from_result


def calculate_polygon_area(polygon: np.ndarray) -> float:
    """
    使用 Shoelace formula 計算 polygon 面積。

    Args:
        polygon: (N, 2) array

    Returns:
        面積（絕對值）
    """
    if len(polygon) < 3:
        return 0.0

    x = polygon[:, 0]
    y = polygon[:, 1]

    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area)


def calculate_polygon_stats(polygons: List[List[List[float]]]) -> Dict[str, Any]:
    """
    計算 polygon 統計數據。

    Args:
        polygons: polygon 列表（來自 objects 字典）

    Returns:
        統計數據字典
    """
    total_points = 0
    total_area = 0.0

    for poly in polygons:
        arr = np.array(poly, dtype=np.float32)
        total_points += len(arr)
        total_area += calculate_polygon_area(arr)

    return {
        "total_points": total_points,
        "total_area": total_area,
        "polygon_count": len(polygons),
    }


def compare_polygon_sets(
    original_stats: Dict[str, Any],
    simplified_stats: Dict[str, Any],
) -> Dict[str, float]:
    """
    比較兩組 polygon 的差異。

    Returns:
        包含簡化率、面積保留率等指標
    """
    orig_points = original_stats["total_points"]
    simp_points = simplified_stats["total_points"]
    orig_area = original_stats["total_area"]
    simp_area = simplified_stats["total_area"]

    # 避免除以零
    point_reduction = 0.0
    if orig_points > 0:
        point_reduction = ((orig_points - simp_points) / orig_points) * 100.0

    area_preservation = 100.0
    if orig_area > 0:
        area_preservation = (simp_area / orig_area) * 100.0

    return {
        "point_reduction_percent": point_reduction,
        "area_preservation_percent": area_preservation,
    }


# --------------------------------------------------------------------------- #
# 偏差指標：簡化/平滑後的幾何「離原始稠密輪廓多遠」
# 對每個原始稠密點，量到簡化折線最近線段的距離（directed Hausdorff: 原始→簡化）。
# 這正是「鋸齒被抹掉多少 / 形狀被改動多少」的直接量。
# --------------------------------------------------------------------------- #
def _point_segment_dists(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """每個點到線段 ab 的距離（向量化）。pts:(N,2), a/b:(2,)。"""
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - proj, axis=1)


def _min_dist_to_polylines(pts: np.ndarray, polylines: List[np.ndarray]) -> np.ndarray:
    """每個點到一組折線的最小距離。"""
    if pts.shape[0] == 0 or not polylines:
        return np.zeros(pts.shape[0], dtype=np.float64)
    best = np.full(pts.shape[0], np.inf, dtype=np.float64)
    for line in polylines:
        if line.ndim != 2 or line.shape[0] == 0:
            continue
        if line.shape[0] == 1:
            best = np.minimum(best, np.linalg.norm(pts - line[0], axis=1))
            continue
        for i in range(line.shape[0] - 1):
            best = np.minimum(best, _point_segment_dists(pts, line[i], line[i + 1]))
    return best


def _object_polylines(obj: Dict[str, Any]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for poly in obj.get("polygons", []):
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            out.append(arr[:, :2])
    return out


def deviation_values(
    original_objects: List[Dict[str, Any]],
    simplified_objects: List[Dict[str, Any]],
) -> np.ndarray:
    """回傳「每個原始稠密點到簡化幾何的最近距離」的彙整陣列（px）。

    物件依索引對齊（同一 result、同一類別篩選 → build_objects_from_result 順序一致）。
    供批次模式跨圖彙整（保留逐點值才能算出正確的全域 mean）。
    """
    all_dev: List[np.ndarray] = []
    for orig, simp in zip(original_objects, simplified_objects):
        orig_lines = _object_polylines(orig)
        if not orig_lines:
            continue
        simp_lines = _object_polylines(simp)
        if not simp_lines:
            continue
        dev = _min_dist_to_polylines(np.vstack(orig_lines), simp_lines)
        dev = dev[np.isfinite(dev)]
        if dev.size:
            all_dev.append(dev)
    return np.concatenate(all_dev) if all_dev else np.zeros(0, dtype=np.float64)


def calculate_deviation(
    original_objects: List[Dict[str, Any]],
    simplified_objects: List[Dict[str, Any]],
) -> Dict[str, float]:
    """原始稠密輪廓點 → 簡化幾何 的偏差（max / mean，單位 px）。"""
    cat = deviation_values(original_objects, simplified_objects)
    if cat.size == 0:
        return {"max_deviation_px": 0.0, "mean_deviation_px": 0.0}
    return {
        "max_deviation_px": float(cat.max()),
        "mean_deviation_px": float(cat.mean()),
    }


# --------------------------------------------------------------------------- #
# 方法定義：每個方法 = (顯示名稱, polygon_opt step 名稱, eps_coeff)
# step 名稱 "none" 代表不套任何 step（稠密輪廓，作為偏差基準）。
# 尚未實作的 step（如 smooth_spline / fit_lines_arcs）在 dispatch 無對應分支時為 no-op，
# 不會報錯，落地後自動生效。
# --------------------------------------------------------------------------- #
DEFAULT_METHODS: List[Tuple[str, str, float]] = [
    ("Original (dense)", "none", 0.0),
    ("RDP (eps_coeff=1)", "rdp", 1.0),
    ("RDP (eps_coeff=2)", "rdp", 2.0),
    ("Visvalingam (eps_coeff=1)", "visvalingam_whyatt", 1.0),
    ("smooth_spline (eps_coeff=1)", "smooth_spline", 1.0),
    ("fit_lines_arcs (eps_coeff=1)", "fit_lines_arcs", 1.0),
]


def _steps_for_method(step_name: str, eps_coeff: float) -> List[Dict[str, Any]]:
    if not step_name or step_name == "none":
        return []
    return [{"name": step_name, "count": 1, "eps_coeff": float(eps_coeff)}]


def _aggregate_stats(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = {"total_points": 0, "total_area": 0.0, "polygon_count": 0}
    for obj in objects:
        stats = calculate_polygon_stats(obj["polygons"])
        agg["total_points"] += stats["total_points"]
        agg["total_area"] += stats["total_area"]
        agg["polygon_count"] += stats["polygon_count"]
    return agg


def compare_polygon_optimizations(
    result,  # YOLO result object
    methods: Optional[List[Tuple[str, str, float]]] = None,
    allowed_class_ids: Optional[List[int]] = None,
    subpixel_contour: bool = False,
    subpixel_scale: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    比較不同 polygon 優化方法的效果。

    Args:
        result: YOLO 推論結果（單張圖）
        methods: 優化方法列表，每個是 (顯示名稱, step 名稱, eps_coeff) tuple；
                 step 名稱為 polygon_opt step（rdp / smooth_spline / fit_lines_arcs ...）或 "none"。
                 若為 None 使用 DEFAULT_METHODS。
        allowed_class_ids: 類別篩選
        subpixel_contour: 是否啟用層次 0 的 sub-pixel 輪廓抽取（基準與各方法一致套用）
        subpixel_scale: sub-pixel 上採樣倍率

    Returns:
        (metrics_list, objects_list)
        - metrics_list: 各方法的統計指標（含與稠密輪廓的偏差）
        - objects_list: 各方法產生的 objects
    """
    if methods is None:
        methods = DEFAULT_METHODS

    # 稠密（無 step）版本作為點數與偏差的基準（與各方法套用相同的輪廓抽取模式）
    original_objects = build_objects_from_result(
        result,
        allowed_class_ids=allowed_class_ids,
        simplify_mode="none",
        simplify_eps_coeff=1.0,
        polygon_opt_steps=[],
        subpixel_contour=subpixel_contour,
        subpixel_scale=subpixel_scale,
    )
    original_stats = _aggregate_stats(original_objects)

    metrics_list: List[Dict[str, Any]] = []
    objects_list: List[Dict[str, Any]] = []

    for method_name, step_name, eps in methods:
        steps = _steps_for_method(step_name, eps)

        start_time = time.perf_counter()
        objects = build_objects_from_result(
            result,
            allowed_class_ids=allowed_class_ids,
            simplify_mode="none",
            simplify_eps_coeff=1.0,
            polygon_opt_steps=steps,
            subpixel_contour=subpixel_contour,
            subpixel_scale=subpixel_scale,
        )
        process_time = (time.perf_counter() - start_time) * 1000

        current_stats = _aggregate_stats(objects)
        comparison = compare_polygon_sets(original_stats, current_stats)
        deviation = calculate_deviation(original_objects, objects)

        metrics_list.append({
            "method": method_name,
            "mode": step_name,
            "epsilon": eps,
            "total_points": current_stats["total_points"],
            "polygon_count": current_stats["polygon_count"],
            "point_reduction_percent": comparison["point_reduction_percent"],
            "area_preservation_percent": comparison["area_preservation_percent"],
            "max_deviation_px": deviation["max_deviation_px"],
            "mean_deviation_px": deviation["mean_deviation_px"],
            "processing_time_ms": process_time,
        })
        objects_list.append(objects)

    return metrics_list, objects_list


def generate_comparison_report(metrics_list: List[Dict[str, Any]]) -> str:
    """
    生成文字格式的比較報告。
    """
    if not metrics_list:
        return "無可用數據"

    report = "=" * 64 + "\n"
    report += "Polygon Optimization Comparison Report\n"
    report += "=" * 64 + "\n\n"

    for i, m in enumerate(metrics_list, 1):
        report += f"{i}. {m['method']}\n"
        report += f"   - Total Points: {m['total_points']}\n"
        report += f"   - Point Reduction: {m['point_reduction_percent']:.2f}%\n"
        report += f"   - Area Preservation: {m['area_preservation_percent']:.2f}%\n"
        report += f"   - Deviation (px): max {m['max_deviation_px']:.2f} | mean {m['mean_deviation_px']:.2f}\n"
        report += f"   - Processing Time: {m['processing_time_ms']:.2f} ms\n"
        report += f"   - Step: {m['mode']}, eps_coeff: {m['epsilon']}\n\n"

    report += "=" * 64 + "\n"
    return report


def create_comparison_images(
    image: np.ndarray,
    result,  # YOLO result
    methods: Optional[List[Tuple[str, str, float]]] = None,
    allowed_class_ids: Optional[List[int]] = None,
    show_vertices: bool = True,
    subpixel_contour: bool = False,
    subpixel_scale: int = 3,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    創建比較視覺化影像（整合函數）。

    Args:
        image: 原始影像 (BGR)
        result: YOLO 結果
        methods: 優化方法列表
        allowed_class_ids: 類別篩選
        show_vertices: 是否顯示頂點

    Returns:
        (images, metrics_list)
        - images: 視覺化影像列表 (BGR)，每張疊原始(稠密)與該方法結果
        - metrics_list: 指標列表
    """
    from .visualization_utils import create_comparison_visualization

    img_bgr = image

    # 稠密基準（與 compare_polygon_optimizations 內部一致）
    original_objects = build_objects_from_result(
        result,
        allowed_class_ids=allowed_class_ids,
        simplify_mode="none",
        simplify_eps_coeff=1.0,
        polygon_opt_steps=[],
        subpixel_contour=subpixel_contour,
        subpixel_scale=subpixel_scale,
    )

    metrics_list, objects_list = compare_polygon_optimizations(
        result,
        methods=methods,
        allowed_class_ids=allowed_class_ids,
        subpixel_contour=subpixel_contour,
        subpixel_scale=subpixel_scale,
    )

    images: List[np.ndarray] = []
    for metrics, objects in zip(metrics_list, objects_list):
        vis_img = create_comparison_visualization(
            img_bgr,
            original_objects,
            objects,
            metrics["method"],
            show_vertices,
        )
        images.append(vis_img)

    return images, metrics_list


def run_batch_comparison(
    model,
    image_paths: List[str],
    methods: Optional[List[Tuple[str, str, float]]] = None,
    allowed_class_ids: Optional[List[int]] = None,
    imgsz: int = 1024,
    conf: float = 0.25,
    device: Optional[str] = None,
    subpixel_contour: bool = False,
    subpixel_scale: int = 3,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """跨多張影像彙整 polygon 優化方法的指標（取得較平均的結論）。

    逐點偏差跨圖彙整（保留 sum/count/max）→ 全域 mean 與 max 正確；點數/面積以總和計算。
    回傳 (aggregated_metrics_list, n_images, n_objects)。
    """
    if methods is None:
        methods = DEFAULT_METHODS

    acc = {
        m[0]: {
            "step": m[1], "eps": m[2],
            "method_pts": 0, "dense_pts": 0,
            "area_orig": 0.0, "area_simp": 0.0,
            "dev_sum": 0.0, "dev_count": 0, "dev_max": 0.0, "time": 0.0,
        }
        for m in methods
    }
    n_images = 0
    n_objects = 0

    for path in image_paths:
        predict_kwargs = {"source": path, "imgsz": imgsz, "conf": conf, "verbose": False}
        if device:
            predict_kwargs["device"] = device
        preds = model.predict(**predict_kwargs)
        if not preds:
            continue
        result = preds[0].cpu()

        dense = build_objects_from_result(
            result, allowed_class_ids=allowed_class_ids,
            simplify_mode="none", simplify_eps_coeff=1.0, polygon_opt_steps=[],
            subpixel_contour=subpixel_contour, subpixel_scale=subpixel_scale,
        )
        dense_stats = _aggregate_stats(dense)
        n_images += 1
        n_objects += len(dense)

        for method_name, step_name, eps in methods:
            steps = _steps_for_method(step_name, eps)
            t0 = time.perf_counter()
            objects = build_objects_from_result(
                result, allowed_class_ids=allowed_class_ids,
                simplify_mode="none", simplify_eps_coeff=1.0, polygon_opt_steps=steps,
                subpixel_contour=subpixel_contour, subpixel_scale=subpixel_scale,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            st = _aggregate_stats(objects)
            dev = deviation_values(dense, objects)

            a = acc[method_name]
            a["method_pts"] += st["total_points"]
            a["dense_pts"] += dense_stats["total_points"]
            a["area_orig"] += dense_stats["total_area"]
            a["area_simp"] += st["total_area"]
            a["time"] += dt
            if dev.size:
                a["dev_sum"] += float(dev.sum())
                a["dev_count"] += int(dev.size)
                a["dev_max"] = max(a["dev_max"], float(dev.max()))

    metrics_list: List[Dict[str, Any]] = []
    for method_name, step_name, eps in methods:
        a = acc[method_name]
        reduction = ((a["dense_pts"] - a["method_pts"]) / a["dense_pts"] * 100.0) if a["dense_pts"] > 0 else 0.0
        area_pres = (a["area_simp"] / a["area_orig"] * 100.0) if a["area_orig"] > 0 else 100.0
        mean_dev = (a["dev_sum"] / a["dev_count"]) if a["dev_count"] > 0 else 0.0
        metrics_list.append({
            "method": method_name, "mode": step_name, "epsilon": eps,
            "total_points": a["method_pts"], "polygon_count": 0,
            "point_reduction_percent": reduction, "area_preservation_percent": area_pres,
            "max_deviation_px": a["dev_max"], "mean_deviation_px": mean_dev,
            "processing_time_ms": a["time"],
        })
    return metrics_list, n_images, n_objects


def _gather_images(image_dir: str) -> List[str]:
    import glob
    import os
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(image_dir, e)))
    return sorted(paths)


def _run_cli() -> None:
    import argparse
    import os

    from ultralytics import YOLO

    from .visualization_utils import create_metrics_table, create_side_by_side_view

    parser = argparse.ArgumentParser(description="Compare polygon optimization steps on one image or a folder.")
    parser.add_argument("--weights", required=True, help="YOLO segmentation weights (.pt)")
    parser.add_argument("--image", default=None, help="Input image path (single-image mode)")
    parser.add_argument("--image-dir", default=None, help="Folder of images (batch / averaged mode)")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu")
    parser.add_argument("--classes", default=None, help="comma-separated class ids to keep")
    parser.add_argument("--out", default="comparison_out", help="output directory")
    parser.add_argument("--no-vertices", action="store_true", help="hide vertex markers")
    parser.add_argument("--subpixel", action="store_true", help="enable layer-0 sub-pixel contour extraction")
    parser.add_argument("--subpixel-scale", type=int, default=3, help="sub-pixel upsample factor")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise SystemExit("需指定 --image 或 --image-dir")

    allowed = None
    if args.classes:
        allowed = [int(c.strip()) for c in args.classes.split(",") if c.strip()]

    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)

    # ---- 批次 / 平均模式 ----
    if args.image_dir:
        paths = _gather_images(args.image_dir)
        if not paths:
            raise SystemExit(f"資料夾內找不到影像: {args.image_dir}")
        print(f"批次比較：{len(paths)} 張影像 @ imgsz={args.imgsz}"
              f"{'（sub-pixel）' if args.subpixel else ''} ...")
        metrics, n_images, n_objects = run_batch_comparison(
            model, paths, methods=None, allowed_class_ids=allowed,
            imgsz=args.imgsz, conf=args.conf, device=args.device,
            subpixel_contour=args.subpixel, subpixel_scale=args.subpixel_scale,
        )
        header = f"批次平均結論：{n_images} 張影像、{n_objects} 個物件\n"
        report = header + generate_comparison_report(metrics) + "\n" + create_metrics_table(metrics)
        with open(os.path.join(args.out, "batch_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        print("\n" + report)
        print(f"\n報告已寫入: {os.path.abspath(os.path.join(args.out, 'batch_report.md'))}")
        return

    # ---- 單張模式 ----
    predict_kwargs = {"source": args.image, "imgsz": args.imgsz, "conf": args.conf}
    if args.device:
        predict_kwargs["device"] = args.device
    result = model.predict(**predict_kwargs)[0].cpu()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"無法讀取影像: {args.image}")

    images, metrics = create_comparison_images(
        image, result, methods=None, allowed_class_ids=allowed,
        show_vertices=not args.no_vertices,
        subpixel_contour=args.subpixel, subpixel_scale=args.subpixel_scale,
    )

    for img, m in zip(images, metrics):
        safe = "".join(ch if ch.isalnum() else "_" for ch in m["method"]).strip("_")
        cv2.imwrite(os.path.join(args.out, f"{safe}.jpg"), img)

    titles = [m["method"] for m in metrics]
    side = create_side_by_side_view(images, titles)
    cv2.imwrite(os.path.join(args.out, "side_by_side.jpg"), side)

    print(generate_comparison_report(metrics))
    print(create_metrics_table(metrics))
    print(f"\n輸出已寫入: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    _run_cli()
