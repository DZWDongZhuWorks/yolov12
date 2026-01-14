# app_utils/visualization_utils.py

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2


def draw_polygon_overlay(
    image: np.ndarray,
    original_polys: List[np.ndarray],
    simplified_polys: List[np.ndarray],
    class_id: int,
    class_name: str,
    confidence: float,
    show_vertices: bool = True,
) -> np.ndarray:
    """
    繪製原始與簡化後的 polygon 疊加比較圖。
    
    Args:
        image: 底圖 (BGR)
        original_polys: 原始 polygon 列表，每個是 (N, 2) ndarray
        simplified_polys: 簡化後 polygon 列表
        class_id: 類別 ID
        class_name: 類別名稱
        confidence: 信心值
        show_vertices: 是否顯示頂點
    
    Returns:
        繪製後的影像 (BGR)
    """
    from ultralytics.utils.plotting import colors as ucolors
    
    overlay = image.copy()
    
    # 顏色設定
    color_original = (100, 255, 100)  # 淺綠色 - 原始
    color_simplified = tuple(int(v) for v in ucolors(class_id, bgr=True))  # 類別顏色 - 簡化後
    
    # 1. 繪製原始 polygon（半透明綠色邊框）
    for poly in original_polys:
        if len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=color_original, thickness=1, lineType=cv2.LINE_AA)
        
        # 顯示原始頂點
        if show_vertices:
            for pt in poly:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, color_original, -1)
    
    # 2. 繪製簡化後 polygon（實線 + 半透明填充）
    for poly in simplified_polys:
        if len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        
        # 填充半透明
        temp = overlay.copy()
        cv2.fillPoly(temp, [pts], color_simplified)
        cv2.addWeighted(temp, 0.3, overlay, 0.7, 0, overlay)
        
        # 邊框
        cv2.polylines(overlay, [pts], isClosed=True, color=color_simplified, thickness=2, lineType=cv2.LINE_AA)
        
        # 顯示簡化後頂點（更大的圓點）
        if show_vertices:
            for pt in poly:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, color_simplified, -1)
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), 1)
    
    # 3. 添加標籤
    if simplified_polys and len(simplified_polys[0]) >= 3:
        # 使用簡化後 polygon 的中心
        pts = np.asarray(simplified_polys[0])
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        
        label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 背景
        cv2.rectangle(overlay, (cx - 2, cy - th - 5), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
        # 文字
        cv2.putText(overlay, label, (cx, cy - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return overlay


def create_comparison_visualization(
    image: np.ndarray,
    objects_original: List[Dict[str, Any]],
    objects_simplified: List[Dict[str, Any]],
    method_name: str,
    show_vertices: bool = True,
) -> np.ndarray:
    """
    創建完整的比較視覺化圖像。
    
    Args:
        image: 原始影像
        objects_original: 原始 polygon 物件列表（來自 build_objects_from_result）
        objects_simplified: 簡化後 polygon 物件列表
        method_name: 優化方法名稱（顯示在圖上）
        show_vertices: 是否顯示頂點
    
    Returns:
        比較視覺化影像
    """
    result = image.copy()
    
    # 繪製每個物件的比較
    for orig, simp in zip(objects_original, objects_simplified):
        # 轉換 polygon 格式
        orig_polys = [np.array(p, dtype=np.float32) for p in orig["polygons"]]
        simp_polys = [np.array(p, dtype=np.float32) for p in simp["polygons"]]
        
        result = draw_polygon_overlay(
            result,
            orig_polys,
            simp_polys,
            simp["class_id"],
            simp["class_name"],
            simp["confidence"] or 0.0,
            show_vertices,
        )
    
    # 添加方法名稱水印
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text = f"Method: {method_name}"
    
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 右上角
    x = result.shape[1] - tw - 10
    y = th + 10
    
    cv2.rectangle(result, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(result, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return result


def create_metrics_table(metrics_list: List[Dict[str, Any]]) -> str:
    """
    創建 Markdown 格式的指標比較表格。
    
    Args:
        metrics_list: 各種優化方法的指標列表
        
    Returns:
        Markdown 表格字符串
    """
    if not metrics_list:
        return "無可用指標"
    
    # 表頭
    table = "| 優化方法 | 總點數 | 簡化率 (%) | 面積保留率 (%) | 處理時間 (ms) |\n"
    table += "|---------|--------|-----------|---------------|-------------|\n"
    
    # 數據行
    for m in metrics_list:
        method = m.get("method", "Unknown")
        total_points = m.get("total_points", 0)
        reduction = m.get("point_reduction_percent", 0.0)
        area_preservation = m.get("area_preservation_percent", 0.0)
        time_ms = m.get("processing_time_ms", 0.0)
        
        table += f"| {method} | {total_points} | {reduction:.1f}% | {area_preservation:.1f}% | {time_ms:.2f} |\n"
    
    return table


def create_side_by_side_view(
    images: List[np.ndarray],
    titles: List[str],
    max_width: int = 1920,
) -> np.ndarray:
    """
    創建並排視圖（最多 4 張圖）。
    
    Args:
        images: 影像列表
        titles: 標題列表
        max_width: 最大寬度（自動縮放）
        
    Returns:
        合併後的影像
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    n = len(images)
    
    # 確保所有圖片高度一致
    target_height = images[0].shape[0]
    resized = []
    for img in images:
        if img.shape[0] != target_height:
            scale = target_height / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, target_height))
        resized.append(img)
    
    # 水平拼接
    combined = np.hstack(resized)
    
    # 如果太寬，縮小
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        new_h = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (max_width, new_h))
    
    return combined


def create_legend_image(width: int = 400, height: int = 150) -> np.ndarray:
    """
    創建圖例說明。
    
    Returns:
        圖例影像 (BGR)
    """
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    y_offset = 30
    
    # 標題
    cv2.putText(legend, "Legend:", (10, y_offset), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    y_offset += 30
    
    # 原始 polygon
    cv2.line(legend, (10, y_offset), (40, y_offset), (100, 255, 100), 2, cv2.LINE_AA)
    cv2.circle(legend, (25, y_offset), 2, (100, 255, 100), -1)
    cv2.putText(legend, "Original Polygon", (50, y_offset + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    y_offset += 30
    
    # 簡化後 polygon
    cv2.line(legend, (10, y_offset), (40, y_offset), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(legend, (25, y_offset), 4, (0, 0, 255), -1)
    cv2.circle(legend, (25, y_offset), 4, (255, 255, 255), 1)
    cv2.putText(legend, "Simplified Polygon", (50, y_offset + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    y_offset += 30
    
    # 填充區域
    pts = np.array([[10, y_offset - 10], [40, y_offset - 10], [40, y_offset + 10], [10, y_offset + 10]], dtype=np.int32)
    temp = legend.copy()
    cv2.fillPoly(temp, [pts.reshape(-1, 1, 2)], (200, 200, 255))
    cv2.addWeighted(temp, 0.3, legend, 0.7, 0, legend)
    cv2.putText(legend, "Filled Area (30% opacity)", (50, y_offset + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    return legend
