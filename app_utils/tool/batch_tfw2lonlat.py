import argparse
import json
from pathlib import Path
import re
import sys
from pyproj import Transformer

# 為了確保可以找到 app_utils 模組，將專案根目錄加入路徑
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 匯入 tfw2lonlat 已經寫好的基礎邏輯
from app_utils.tool.tfw2lonlat import (
    WorldFile, 
    convert_yolo_json_to_geojson, 
    convert_geojson_featurecollection, 
    _convert_coord_list
)

def batch_process(json_dir: Path, output_dir: Path, tfw_dir: Path, src_epsg: int,
                  rotation_map: dict | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] 初始化座標轉換: EPSG:{src_epsg} -> EPSG:4326")
    transformer = Transformer.from_crs(
        f"EPSG:{src_epsg}", 
        "EPSG:4326", 
        always_xy=True
    )

    # 紀錄各個 TFW 檔案的快取，避免對於同一大圖的切片重複讀取 I/O
    tfw_cache = {}

    # 正規表達式：匹配 原檔名_X_Y[yolo後綴].json
    # .* 會貪婪匹配，所以原檔名如果有底線也能正確對應，(_.*)? 則是選填的 YOLO 版本號後綴
    pattern = re.compile(r"^(.*)_(\d+)_(\d+)(_.*)?\.json$")

    count = 0
    success_count = 0

    print(f"[Info] 開始處理資料夾: {json_dir}")
    print("-" * 40)

    for json_file in json_dir.glob("*.json"):
        count += 1
        match = pattern.match(json_file.name)
        
        if not match:
            # 如果沒有切圖的 _X_Y 後綴，預設 crop_x=0, crop_y=0，並嘗試直接找同名 tfw 
            base_name = json_file.stem
            crop_x, crop_y = 0.0, 0.0
            print(f"[Warning] 檔案 {json_file.name} 不符合 _X_Y 後綴格式，預設 crop=(0,0)")
        else:
            base_name = match.group(1)
            crop_x = float(match.group(2))
            crop_y = float(match.group(3))

        # 尋找對應的 TFW 檔案 (支援 .tfw 和 .jgw)
        tfw_file = tfw_dir / f"{base_name}.tfw"
        if not tfw_file.exists():
            tfw_file = tfw_dir / f"{base_name}.jgw"
        
        if not tfw_file.exists():
            print(f"[Error] 找不到檔案 {json_file.name} 對應的 World File ({base_name}.tfw / .jgw)，跳過處理。")
            continue

        if tfw_file not in tfw_cache:
            try:
                tfw_cache[tfw_file] = WorldFile.from_file(tfw_file)
            except Exception as e:
                print(f"[Error] 讀取 {tfw_file} 失敗: {e}")
                continue
        
        wf = tfw_cache[tfw_file]

        # 讀取 JSON 並處理
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] 讀取 {json_file.name} 失敗: {e}")
            continue

        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            result = convert_geojson_featurecollection(data, wf, transformer, crop_x, crop_y)
        elif isinstance(data, dict) and "objects" in data:
            result = convert_yolo_json_to_geojson(data, wf, transformer, crop_x, crop_y,
                                                  rotation_map=rotation_map)
        else:
            # 備用：通用遞迴
            result = _convert_coord_list(data, wf, transformer, crop_x, crop_y)

        # 輸出寫入
        out_file = output_dir / json_file.name
        try:
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            success_count += 1
        except Exception as e:
            print(f"[Error] 寫入 {out_file.name} 失敗: {e}")

    print("-" * 40)
    print(f"[Success] 處理完成！共掃描 {count} 個檔案，成功轉換 {success_count} 個。")
    print(f"[Info] 輸出路徑: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="批次處理 YOLO JSON 轉 GeoJSON (依據切圖檔名自動匹配 TFW 及 Crop 偏移)")
    
    parser.add_argument("--json-dir", required=True, type=Path, help="輸入的 YOLO JSON 資料夾路徑")
    parser.add_argument("--output-dir", required=True, type=Path, help="輸出的 GeoJSON 資料夾路徑")
    parser.add_argument("--tfw-dir", required=True, type=Path, help="原始 TFW/JGW 檔案所在資料夾路徑")
    parser.add_argument("--src-epsg", type=int, default=3826, help="原始 TIF 的 EPSG 代碼 (預設 3826)")
    parser.add_argument("--rotation-map", type=Path, default=None,
                        help="旋轉切片方向對照表（detect_slice_rotation.py 產出）")

    args = parser.parse_args()

    # 檢查輸入資料夾是否存在
    if not args.json_dir.is_dir():
        print(f"[Error] JSON 資料夾不存在: {args.json_dir}")
        return
    if not args.tfw_dir.is_dir():
        print(f"[Error] TFW 資料夾不存在: {args.tfw_dir}")
        return

    rotation_map = None
    if args.rotation_map:
        with args.rotation_map.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rotation_map = payload.get("directions", payload)  # 支援純 dict 或含 details 的格式
        print(f"[Info] 載入 rotation map: {len(rotation_map)} 個旋轉切片")

    batch_process(args.json_dir, args.output_dir, args.tfw_dir, args.src_epsg,
                  rotation_map=rotation_map)

if __name__ == "__main__":
    main()
