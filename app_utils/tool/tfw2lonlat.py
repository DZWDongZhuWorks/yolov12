from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, List, Dict, Optional

from pyproj import Transformer


# ==========================================
# 1. WorldFile 與 基礎座標轉換
# ==========================================
@dataclass
class WorldFile:
    """
    處理 TFW (World File) 的六參數轉換。
    公式：
    X_map = A * px + B * py + C
    Y_map = D * px + E * py + F
    """
    A: float  # X 方向像素大小
    D: float  # 旋轉參數 (通常為 0)
    B: float  # 旋轉參數 (通常為 0)
    E: float  # Y 方向像素大小 (通常為負)
    C: float  # 左上角 X 座標
    F: float  # 左上角 Y 座標

    @staticmethod
    def from_file(path: Path) -> "WorldFile":
        """讀取 .tfw 或 .jgw 檔案"""
        values: List[float] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    values.append(float(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到 World File: {path}")
        
        if len(values) != 6:
            raise ValueError(f"World file {path} 格式錯誤，應包含 6 行數值。")
        
        return WorldFile(*values)

    def pixel_to_map(self, px: float, py: float) -> Tuple[float, float]:
        """將『絕對像素座標』轉為『原始地圖平面座標 (Map Grid)』"""
        X = self.A * px + self.B * py + self.C
        Y = self.D * px + self.E * py + self.F
        return X, Y


def pixel_to_lonlat(
    px: float,
    py: float,
    wf: WorldFile,
    transformer: Transformer,
) -> Tuple[float, float]:
    """
    單點轉換：Pixel (絕對座標) -> Map Grid -> Lon/Lat
    """
    map_x, map_y = wf.pixel_to_map(px, py)
    # transform(y, x) 或 (x, y) 取決於 proj 版本與定義，always_xy=True 確保輸入輸出為 (lon, lat)
    lon, lat = transformer.transform(map_x, map_y)
    return lon, lat


# ==========================================
# 2. 遞迴座標列表轉換 (核心運算)
# ==========================================
def _convert_coord_list(
    coords: Any,
    wf: WorldFile,
    transformer: Transformer,
    crop_x: float,
    crop_y: float,
) -> Any:
    """
    遞迴遍歷座標陣列，將最底層的 [x, y] 轉為 [lon, lat]。
    在此處加入 crop_x / crop_y 的偏移量運算。
    """
    # 判斷是否為座標點 [x, y] (長度>=2, 且元素可轉為 float)
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            # 嘗試解析數值
            local_px = float(coords[0])
            local_py = float(coords[1])
            
            # --- 關鍵修正：加上 Crop 偏移量 ---
            # 原始 TIF 的絕對像素 = 局部裁切像素 + 裁切原點偏移
            global_px = local_px + crop_x
            global_py = local_py + crop_y

            # 轉換為經緯度
            lon, lat = pixel_to_lonlat(global_px, global_py, wf, transformer)

            # 保留可能的 Z 值或其他維度
            extra_dims = list(coords[2:])
            return [lon, lat, *extra_dims]

        except (TypeError, ValueError):
            # 若無法轉為 float (例如是巢狀 list [[x,y], [x,y]])，則繼續遞迴
            pass

    # 若是 list 但不是點，繼續往下遞迴
    if isinstance(coords, (list, tuple)):
        return [
            _convert_coord_list(c, wf, transformer, crop_x, crop_y)
            for c in coords
        ]

    # 其他情況直接回傳 (如 None 或字串)
    return coords


# ==========================================
# 3. GeoJSON 處理邏輯
# ==========================================
def convert_geojson_featurecollection(
    obj: Dict[str, Any],
    wf: WorldFile,
    transformer: Transformer,
    crop_x: float,
    crop_y: float,
) -> Dict[str, Any]:
    """處理標準 GeoJSON 輸入"""
    
    def _process_geometry(geom: Dict[str, Any]) -> Dict[str, Any]:
        if not geom:
            return None
        new_geom = dict(geom)
        # 直接遞迴轉換 coordinates
        new_geom["coordinates"] = _convert_coord_list(
            geom.get("coordinates"), wf, transformer, crop_x, crop_y
        )
        return new_geom

    new_features = []
    for feat in obj.get("features", []):
        new_feat = dict(feat)
        if "geometry" in feat:
            new_feat["geometry"] = _process_geometry(feat["geometry"])
        new_features.append(new_feat)

    new_obj = dict(obj)
    new_obj["features"] = new_features
    return new_obj


# ==========================================
# 4. YOLO JSON 處理邏輯 (修正版 - 支援 LineString)
# ==========================================
def convert_yolo_json_to_geojson(
    obj: Dict[str, Any],
    wf: WorldFile,
    transformer: Transformer,
    crop_x: float,
    crop_y: float,
) -> Dict[str, Any]:
    """
    將 YOLO 偵測結果轉為 GeoJSON。
    包含 Polygon 閉合修正、BBox 轉換，並透過點序列閉合狀態判斷 Polygon/LineString。
    """
    features: List[Dict[str, Any]] = []
    objects = obj.get("objects", [])

    for idx, o in enumerate(objects):
        if not isinstance(o, dict):
            continue

        # --- 資料讀取 ---
        class_id = o.get("class_id")
        class_name = o.get("class_name")
        confidence = o.get("confidence")
        bbox_pixel = o.get("bbox_xyxy")  # [x1, y1, x2, y2]
        polygons = o.get("polygons", []) # List[List[[x,y]...]]
        holes_aligned = o.get("holes") or []        # 與 polygons 索引對齊的洞清單（新版欄位，可缺漏）
        line_strings = o.get("line_strings") or []  # lane line 步驟輸出（閉合座標 = 環形線，非面）

        # --- 1. 處理幾何 (Polygon/LineString) ---
        geometry = None

        # 1-a. line_strings 優先：物件經 polygon_to_lane_line 轉換後，
        #      閉合座標（首尾相同）代表「環形 lane line」，須輸出 LineString 而非 Polygon
        if line_strings:
            converted_lines: List[List[List[float]]] = []
            for ls in line_strings:
                coords = ls.get("coordinates") if isinstance(ls, dict) else ls
                if not coords or len(coords) < 2:
                    continue
                converted_lines.append(_convert_coord_list(coords, wf, transformer, crop_x, crop_y))
            if len(converted_lines) == 1:
                geometry = {"type": "LineString", "coordinates": converted_lines[0]}
            elif converted_lines:
                geometry = {"type": "MultiLineString", "coordinates": converted_lines}

        # 1-b. 一般路徑：遍歷每個點序列（多邊形外環或線）
        geometry_type: Optional[str] = None
        converted_shapes: List[Any] = []

        if geometry is None:
            for ring_idx, poly_ring in enumerate(polygons):

                # 檢查原始像素座標是否閉合 (首尾點是否相同)
                is_input_closed = (len(poly_ring) > 2 and poly_ring[0] == poly_ring[-1])

                # 轉換座標 (先轉換，不論是否閉合)
                transformed_ring = _convert_coord_list(
                    poly_ring, wf, transformer, crop_x, crop_y
                )

                # --- 決定類型與 GeoJSON 閉合修正 ---
                current_type = None
                shape: Any = None
                if is_input_closed:
                    # 判斷為 Polygon。強制 GeoJSON 閉合規範。
                    current_type = "Polygon"
                    if transformed_ring and transformed_ring[0] != transformed_ring[-1]:
                        transformed_ring.append(transformed_ring[0])
                    # 帶洞 Polygon：GeoJSON ring 順序 = [外環, 洞1, 洞2, ...]
                    ring_group = [transformed_ring]
                    obj_holes = holes_aligned[ring_idx] if ring_idx < len(holes_aligned) else []
                    for hole in (obj_holes or []):
                        if not hole or len(hole) <= 2:
                            continue
                        transformed_hole = _convert_coord_list(hole, wf, transformer, crop_x, crop_y)
                        if transformed_hole and transformed_hole[0] != transformed_hole[-1]:
                            transformed_hole.append(transformed_hole[0])
                        ring_group.append(transformed_hole)
                    shape = ring_group

                elif len(transformed_ring) >= 2:
                    # 判斷為 LineString (開放狀態，且至少兩個點)
                    current_type = "LineString"
                    shape = transformed_ring
                else:
                    continue # 點數不足，跳過

                # --- 類型一致性檢查 ---
                if geometry_type is None:
                    geometry_type = current_type # 設定第一個檢測到的類型
                elif geometry_type != current_type:
                    # 如果一個 Feature 內包含混合類型 (LineString 和 Polygon)，這是無效的 GeoJSON Feature，跳過此幾何
                    print(f"Warning: Object {idx} contains mixed Polygon/LineString geometries. Skipping geometry conversion.")
                    geometry_type = None
                    break

                converted_shapes.append(shape)

            # --- 最終組裝 Geometry ---
            if converted_shapes:
                if geometry_type == "Polygon":
                    if len(converted_shapes) == 1:
                        # 單一 Polygon: GeoJSON 座標結構: [[外環], [洞1], ...]
                        geometry = {"type": "Polygon", "coordinates": converted_shapes[0]}
                    else:
                        # MultiPolygon: GeoJSON 座標結構: [[[外環, 洞...]], ...]（每組已是 ring list）
                        geometry = {"type": "MultiPolygon", "coordinates": converted_shapes}

                elif geometry_type == "LineString":
                    if len(converted_shapes) == 1:
                        # 單一 LineString: GeoJSON 座標結構: [點1, 點2, ...]
                        geometry = {"type": "LineString", "coordinates": converted_shapes[0]}
                    else:
                        # MultiLineString: GeoJSON 座標結構: [[線1點], [線2點], ...]
                        geometry = {"type": "MultiLineString", "coordinates": converted_shapes}

        # --- 2. 處理 BBox (轉為經緯度) ---
        bbox_lonlat = None
        if bbox_pixel and len(bbox_pixel) == 4:
            x1, y1, x2, y2 = bbox_pixel
            # 轉換對角點 (加上 crop 偏移)
            l1, la1 = pixel_to_lonlat(x1 + crop_x, y1 + crop_y, wf, transformer)
            l2, la2 = pixel_to_lonlat(x2 + crop_x, y2 + crop_y, wf, transformer)
            
            # 重新排序 min/max，因為投影轉換可能改變方向
            bbox_lonlat = [
                min(l1, l2), min(la1, la2),
                max(l1, l2), max(la1, la2)
            ]

        # --- 3. 組裝 Properties ---
        properties = {
            "object_id": idx,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox_pixel": bbox_pixel,    # 保留原始 pixel 數據方便除錯
            "bbox_lonlat": bbox_lonlat,  # 新增經緯度 bbox
        }

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": properties,
        }
        features.append(feature)

    # 輸出結果
    geojson: Dict[str, Any] = {
        "type": "FeatureCollection",
        "name": "YOLO_Inference_Results",
        "features": features,
        "metadata": {
            "source_image": obj.get("image", {}),
            "model_info": obj.get("model", {}),
            "crs": "EPSG:4326"
        }
    }
    return geojson

# ==========================================
# 5. 主程式入口
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Pixel JSON to GeoJSON Converter (Crop Supported)")
    
    parser.add_argument("--tfw", required=True, help="原始大圖的 World File (.tfw/.jgw) 路徑")
    parser.add_argument("--json", required=True, help="輸入 JSON 檔案路徑")
    parser.add_argument("--output", required=True, help="輸出 GeoJSON 檔案路徑")
    
    # 座標系統設定
    parser.add_argument("--src-epsg", type=int, default=3826, help="原始 TIF 的 EPSG 代碼 (預設 TWD97/3826)")
    
    # Crop 設定：這對於解決你的問題至關重要
    parser.add_argument("--crop-x", type=float, default=0.0, help="裁切圖左上角在原始大圖中的 X 像素座標")
    parser.add_argument("--crop-y", type=float, default=0.0, help="裁切圖左上角在原始大圖中的 Y 像素座標")

    args = parser.parse_args()

    tfw_path = Path(args.tfw)
    json_path = Path(args.json)
    output_path = Path(args.output)

    # 1. 準備 WorldFile
    print(f"[Info] 讀取 World File: {tfw_path}")
    wf = WorldFile.from_file(tfw_path)

    # 2. 準備座標轉換器
    print(f"[Info] 初始化座標轉換: EPSG:{args.src_epsg} -> EPSG:4326")
    transformer = Transformer.from_crs(
        f"EPSG:{args.src_epsg}", 
        "EPSG:4326", 
        always_xy=True
    )

    # 3. 讀取輸入資料
    print(f"[Info] 讀取輸入 JSON: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[Info] 處理裁切偏移: X={args.crop_x}, Y={args.crop_y}")

    # 4. 執行轉換
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        print("[Info] 偵測為標準 GeoJSON 格式")
        result = convert_geojson_featurecollection(data, wf, transformer, args.crop_x, args.crop_y)
    elif isinstance(data, dict) and "objects" in data:
        print("[Info] 偵測為 YOLO 自訂格式")
        result = convert_yolo_json_to_geojson(data, wf, transformer, args.crop_x, args.crop_y)
    else:
        print("[Warning] 未知格式，嘗試通用遞迴轉換")
        result = _convert_coord_list(data, wf, transformer, args.crop_x, args.crop_y)

    # 5. 輸出
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Success] 轉換完成！檔案已儲存至: {output_path}")


if __name__ == "__main__":
    main()