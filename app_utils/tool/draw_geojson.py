from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from pyproj import Transformer

# ==========================================
# 顏色配置與工具
# ==========================================

# 固定顏色調色盤 (BGR 格式)
COLOR_PALETTE = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (128, 0, 128),    # Purple
    (0, 128, 0),      # Dark Green
    (128, 128, 128),  # Gray
    (0, 165, 255),    # Orange
]
# 用於儲存已分配顏色的字典，確保同一類別每次運行顏色一致
CLASS_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {}

def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """根據類別名稱，返回一個固定的顏色"""
    if class_name not in CLASS_COLOR_MAP:
        # 使用類別名稱的雜湊值，以確保同一類別每次都選到同一個顏色
        color_index = hash(class_name) % len(COLOR_PALETTE)
        CLASS_COLOR_MAP[class_name] = COLOR_PALETTE[color_index]
    return CLASS_COLOR_MAP[class_name]


# ==========================================
# 1. WorldFile 與 逆向座標轉換
# (此段程式碼與原程式碼相同，略過註解)
# ==========================================
@dataclass
class WorldFile:
    A: float
    D: float
    B: float
    E: float
    C: float
    F: float

    @staticmethod
    def from_file(path: Path) -> "WorldFile":
        values = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    values.append(float(line.strip()))
        if len(values) != 6:
            raise ValueError("TFW 格式錯誤")
        return WorldFile(*values)

    def map_to_pixel(self, x_map: float, y_map: float) -> Tuple[float, float]:
        """
        逆向轉換：Map Coordinate (X, Y) -> Pixel (px, py)
        """
        coeffs = np.array([[self.A, self.B], [self.D, self.E]])
        constants = np.array([x_map - self.C, y_map - self.F])
        
        try:
            px, py = np.linalg.solve(coeffs, constants)
            return px, py
        except np.linalg.LinAlgError:
            raise ValueError("World File 矩陣無法求解 (可能是參數全為0)")


def latlon_to_local_pixel(
    lon: float,
    lat: float,
    wf: WorldFile,
    transformer: Transformer,
    crop_x: float,
    crop_y: float
) -> Tuple[int, int]:
    """
    流程：WGS84 (Lon, Lat) -> 投影座標 (Map X, Y) -> 全域 Pixel -> 局部 Crop Pixel
    """
    map_x, map_y = transformer.transform(lon, lat)
    global_px, global_py = wf.map_to_pixel(map_x, map_y)
    local_px = global_px - crop_x
    local_py = global_py - crop_y
    
    return int(round(local_px)), int(round(local_py))


# ==========================================
# 2. 繪圖邏輯 (已修改)
# ==========================================

def draw_on_image(
    image: np.ndarray,
    geojson: Dict[str, Any],
    wf: WorldFile,
    transformer: Transformer,
    crop_x: float,
    crop_y: float,
    draw_labels: bool # 新增參數
):
    """
    解析 GeoJSON FeatureCollection 並繪製在影像上
    """
    COLOR_TEXT = (255, 255, 255)  # 白色 (文字)
    height, width, _ = image.shape # 取得影像尺寸
    
    features = geojson.get("features", [])
    print(f"正在繪製 {len(features)} 個物件...")

    for idx, feat in enumerate(features):
        print(f"繪製物件 {idx + 1}/{len(features)}", end='\r')
        props = feat.get("properties", {})
        geom = feat.get("geometry")
        
        if not geom:
            continue

        label = props.get("class_name", "Unknown")
        conf = props.get("confidence", 0.0)
        label_text = f"{label} {conf:.2f}"
        
        # 根據類別取得顏色
        current_color = get_class_color(label) 

        # --- 處理幾何 ---
        polys_to_draw = []
        lines_to_draw = []

        bbox_min_x, bbox_min_y = float('inf'), float('inf')
        bbox_max_x, bbox_max_y = float('-inf'), float('-inf')

        geom_type = geom["type"]
        
        # 輔助函式：遍歷座標點並計算 BBox
        def process_coordinates(coordinates_list, is_closed=False):
            nonlocal bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y
            
            pts = []
            # 座標點清單在 GeoJSON 結構中可能巢狀不同層次，這裡假設最裡層是 [lon, lat]
            # 對於 Polygon/LineString，外層是 Ring/Points 陣列
            if not isinstance(coordinates_list[0][0], (int, float)):
                # 如果是多層巢狀，只處理最外層的 Ring/Line (例如 Polygon 的外環)
                coords_to_process = coordinates_list[0]
            else:
                coords_to_process = coordinates_list

            for coord in coords_to_process:
                lon, lat = coord[0], coord[1]
                px, py = latlon_to_local_pixel(lon, lat, wf, transformer, crop_x, crop_y)
                pts.append([px, py])
                
                # 計算繪圖用的 BBox
                if px < bbox_min_x: bbox_min_x = px
                if py < bbox_min_y: bbox_min_y = py
                if px > bbox_max_x: bbox_max_x = px
                if py > bbox_max_y: bbox_max_y = py
                
            return np.array(pts, dtype=np.int32)
        
        
        if geom_type == "Polygon":
            raw_polys = geom["coordinates"]
            # 處理 Polygon / MultiPolygon
            for poly in raw_polys:
                polys_to_draw.append(process_coordinates(poly))
                
        elif geom_type == "MultiPolygon":
            for multi_poly in geom["coordinates"]:
                for poly in multi_poly:
                     polys_to_draw.append(process_coordinates(poly))

        elif geom_type == "LineString":
            # LineString 座標結構是 [ [lon, lat], [lon, lat], ... ]
            lines_to_draw.append(process_coordinates(geom["coordinates"]))

        elif geom_type == "MultiLineString":
            for line_coords in geom["coordinates"]:
                lines_to_draw.append(process_coordinates(line_coords))


        # --- 0. 邊界檢查 (除錯關鍵) ---
        has_geometry = polys_to_draw or lines_to_draw
        if has_geometry and (bbox_max_x < 0 or bbox_min_x > width or
                             bbox_max_y < 0 or bbox_min_y > height):
            
            print(f"\n[Warning] 物件 {idx} ({label}) 座標超出影像範圍！")
            print(f"  影像尺寸: ({width}, {height})")
            print(f"  計算 BBox: ({bbox_min_x:.1f}, {bbox_min_y:.1f}) to ({bbox_max_x:.1f}, {bbox_max_y:.1f})")
            continue # 跳過繪製這個不可見的物件


        # 1. 繪製多邊形 (使用 current_color)
        if polys_to_draw:
            # 畫外框
            cv2.polylines(image, polys_to_draw, isClosed=True, color=current_color, thickness=2)
            # 填滿半透明 (使用 current_color)
            overlay = image.copy()
            cv2.fillPoly(overlay, polys_to_draw, color=current_color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
        # 1.5 繪製線段 (新增)
        if lines_to_draw:
             # LineString 不閉合 (isClosed=False)
            cv2.polylines(image, lines_to_draw, isClosed=False, color=current_color, thickness=2)


        # 2. 繪製標籤 (使用 draw_labels 檢查和 current_color)
        if draw_labels and bbox_min_x != float('inf'):
            # 確保 BBox 座標在影像範圍內，防止繪圖越界，這裡只取在圖上的部分 BBox 進行標籤繪製
            x_start = max(0, int(bbox_min_x))
            y_start = max(0, int(bbox_min_y))

            # 簡單的文字背景
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 標籤背景 (使用 current_color)
            cv2.rectangle(image, 
                          (x_start, y_start - 20), 
                          (x_start + w, y_start), 
                          current_color, -1)
            
            # 標籤文字
            cv2.putText(image, label_text, 
                        (x_start, y_start - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    return image


# ==========================================
# 3. 主程式 (已修改)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="驗證工具：將 GeoJSON 逆向繪製回影像上")
    parser.add_argument("--tfw", required=True, help="原始大圖的 World File (.tfw)")
    parser.add_argument("--geojson", required=True, help="包含經緯度的 GeoJSON 檔案")
    parser.add_argument("--image", required=True, help="要繪製的影像檔案 (例如 crop 的 jpg)")
    parser.add_argument("--output", required=True, help="輸出結果影像路徑")
    
    parser.add_argument("--src-epsg", type=int, default=3826, help="TFW 對應的投影座標 EPSG (預設 3826)")
    parser.add_argument("--crop-x", type=float, default=0.0, help="裁切時的 X 偏移量")
    parser.add_argument("--crop-y", type=float, default=0.0, help="裁切時的 Y 偏移量")
    # 修正需求 1：增加 --no-label
    parser.add_argument("--no-label", action='store_true', help="關閉繪製類別標籤與信心分數。")
    

    args = parser.parse_args()

    # 處理參數
    tfw_path = Path(args.tfw)
    geojson_path = Path(args.geojson)
    img_path = Path(args.image)
    out_path = Path(args.output)
    draw_labels_flag = not args.no_label # 如果指定了 --no-label，則 draw_labels_flag 為 False

    # 1. 讀取影像
    print(f"[Info] 讀取影像: {img_path}")
    # 使用 numpy.fromfile 和 cv2.imdecode 來支援包含中文的路徑
    image_data = np.fromfile(str(img_path), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取影像: {img_path}")

    # 2. 讀取 GeoJSON
    print(f"[Info] 讀取 GeoJSON: {geojson_path}")
    with geojson_path.open("r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # 3. 初始化轉換器
    wf = WorldFile.from_file(tfw_path)
    print(f"[Info] 建立投影轉換: EPSG:4326 -> EPSG:{args.src_epsg}")
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{args.src_epsg}", always_xy=True)

    # 4. 繪製
    print(f"[Info] 開始繪製 (Crop Offset: {args.crop_x}, {args.crop_y})...")
    result_img = draw_on_image(
        image, 
        geojson_data, 
        wf, 
        transformer, 
        args.crop_x, 
        args.crop_y,
        draw_labels_flag # 傳入是否繪製標籤的旗標
    )

    # 5. 存檔
    # 使用 cv2.imencode 和 numpy.tofile 來支援包含中文的路徑存檔
    ext = out_path.suffix if out_path.suffix else ".jpg"
    is_success, im_buf = cv2.imencode(ext, result_img)
    if is_success:
        im_buf.tofile(str(out_path))
        print(f"[Success] 繪製完成！結果已儲存至: {out_path}")
    else:
        print(f"[Error] 無法編碼並儲存影像至: {out_path}")

if __name__ == "__main__":
    main()