# app_utils/polygon_comparison.py

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


def compare_polygon_optimizations(
    result,  # YOLO result object
    methods: Optional[List[Tuple[str, str, float]]] = None,
    allowed_class_ids: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    比較不同 polygon 優化方法的效果。
    
    Args:
        result: YOLO 推論結果（單張圖）
        methods: 優化方法列表，每個是 (名稱, 模式, epsilon) tuple
                 如果為 None，使用預設方法
        allowed_class_ids: 類別篩選
        
    Returns:
        (metrics_list, objects_list)
        - metrics_list: 各方法的統計指標
        - objects_list: 各方法產生的 objects
    """
    if methods is None:
        methods = [
            ("Original (No Simplification)", "none", 0.0),
            ("RDP (ε=0.01)", "rdp", 0.01),
            ("RDP (ε=0.02)", "rdp", 0.02),
            ("Convex Hull", "convex_hull", 0.0),
        ]
    
    metrics_list = []
    objects_list = []
    
    # 先產生原始（無簡化）版本作為基準
    start_time = time.perf_counter()
    original_objects = build_objects_from_result(
        result,
        allowed_class_ids=allowed_class_ids,
        simplify_mode="none",
        simplify_eps_ratio=0.0,
    )
    original_time = (time.perf_counter() - start_time) * 1000
    
    original_stats = {"total_points": 0, "total_area": 0.0, "polygon_count": 0}
    for obj in original_objects:
        stats = calculate_polygon_stats(obj["polygons"])
        original_stats["total_points"] += stats["total_points"]
        original_stats["total_area"] += stats["total_area"]
        original_stats["polygon_count"] += stats["polygon_count"]
    
    # 對每種方法進行處理
    for method_name, mode, eps in methods:
        start_time = time.perf_counter()
        
        objects = build_objects_from_result(
            result,
            allowed_class_ids=allowed_class_ids,
            simplify_mode=mode,
            simplify_eps_ratio=eps,
        )
        
        process_time = (time.perf_counter() - start_time) * 1000
        
        # 計算統計
        current_stats = {"total_points": 0, "total_area": 0.0, "polygon_count": 0}
        for obj in objects:
            stats = calculate_polygon_stats(obj["polygons"])
            current_stats["total_points"] += stats["total_points"]
            current_stats["total_area"] += stats["total_area"]
            current_stats["polygon_count"] += stats["polygon_count"]
        
        # 比較
        comparison = compare_polygon_sets(original_stats, current_stats)
        
        metrics = {
            "method": method_name,
            "mode": mode,
            "epsilon": eps,
            "total_points": current_stats["total_points"],
            "polygon_count": current_stats["polygon_count"],
            "point_reduction_percent": comparison["point_reduction_percent"],
            "area_preservation_percent": comparison["area_preservation_percent"],
            "processing_time_ms": process_time,
        }
        
        metrics_list.append(metrics)
        objects_list.append(objects)
    
    return metrics_list, objects_list


def generate_comparison_report(metrics_list: List[Dict[str, Any]]) -> str:
    """
    生成文字格式的比較報告。
    
    Args:
        metrics_list: 指標列表
        
    Returns:
        格式化的報告字符串
    """
    if not metrics_list:
        return "無可用數據"
    
    report = "=" * 60 + "\n"
    report += "Polygon Optimization Comparison Report\n"
    report += "=" * 60 + "\n\n"
    
    for i, m in enumerate(metrics_list, 1):
        report += f"{i}. {m['method']}\n"
        report += f"   - Total Points: {m['total_points']}\n"
        report += f"   - Point Reduction: {m['point_reduction_percent']:.2f}%\n"
        report += f"   - Area Preservation: {m['area_preservation_percent']:.2f}%\n"
        report += f"   - Processing Time: {m['processing_time_ms']:.2f} ms\n"
        report += f"   - Mode: {m['mode']}, Epsilon: {m['epsilon']}\n"
        report += "\n"
    
    report += "=" * 60 + "\n"
    
    return report


def create_comparison_images(
    image: np.ndarray,
    result,  # YOLO result
    methods: Optional[List[Tuple[str, str, float]]] = None,
    allowed_class_ids: Optional[List[int]] = None,
    show_vertices: bool = True,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    創建比較視覺化影像（整合函數）。
    
    Args:
        image: 原始影像 (RGB or BGR)
        result: YOLO 結果
        methods: 優化方法列表
        allowed_class_ids: 類別篩選
        show_vertices: 是否顯示頂點
        
    Returns:
        (images, metrics_list)
        - images: 視覺化影像列表 (BGR)
        - metrics_list: 指標列表
    """
    from .visualization_utils import create_comparison_visualization
    
    # 確保是 BGR 格式
    if image.ndim == 3 and image.shape[2] == 3:
        # 假設輸入可能是 RGB，轉為 BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]) else image
    else:
        img_bgr = image
    
    # 獲取原始物件（作為比較基準）
    original_objects = build_objects_from_result(
        result,
        allowed_class_ids=allowed_class_ids,
        simplify_mode="none",
        simplify_eps_ratio=0.0,
    )
    
    # 進行比較
    metrics_list, objects_list = compare_polygon_optimizations(
        result,
        methods=methods,
        allowed_class_ids=allowed_class_ids,
    )
    
    # 創建視覺化
    images = []
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
