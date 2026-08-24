# 衛星影像推論管線教學（YOLO Detect → GeoJSON → 大圖驗證）

從大圖切片的 HD 影像出發，批次推論、轉換為 GPS 座標的 GeoJSON，
最後把零星切片的結果合併繪回原始大圖做最終驗證。

```
大圖 TIF ──(切片工具，外部)──> HD 切片 PNG
                                   │
        ┌──────────────────────────┘
        ▼
  [1] YOLO 批次推論 (app.py CLI)            ──> {圖名}__{模型}.json / .jpg
        ▼
  [2] TFW 定位檔對應 (stage_tfw_files.py)    ──> _tfw/ 暫存資料夾
  [3] 旋轉切片方向偵測 (detect_slice_rotation.py) ──> rotation_map.json
        ▼
  [4] GeoJSON 轉換 (batch_tfw2lonlat.py)     ──> 帶 GPS 座標的 GeoJSON
        ▼
  [5] 大圖驗證 (draw_geojson_on_bigmap.py)   ──> {圖幅}__validation.jpg
```

五個步驟都有獨立 CLI 腳本，`run_pipeline.py` 以 subprocess 依序串接（低耦合，
任一步驟也可單獨重跑）。

---

## 0. 前置需求

| 項目 | 說明 |
|---|---|
| 環境 | conda env `yolov12`（gradio 5.x、ultralytics、pyproj、opencv） |
| 模型 | 例：`D:\_Dataset_\yolov12\model\e7_b1_hd_a.pt` |
| config | GUI「配置管理」匯出的 JSON（顯示設定、類別篩選、Mask/Polygon 優化步驟） |
| 大圖資料 | 原始 TIF 與 TWD97 定位檔（.tfw/.jgw），例：`G:\temp\全市正射影像圖(109~111年度)\` |
| 切片影像 | 依命名規則切好的 HD PNG（見下方「檔名規則」） |

## 1. 一鍵執行（建議）

```powershell
conda activate yolov12
cd D:\yolov12

python run_pipeline.py `
    -i  "D:\_Dataset_\Satellite\HD-DS\110-1" `
    -o  "D:\output\110-1" `
    -m  "D:\_Dataset_\yolov12\model\e7_b1_hd_a.pt" `
    -c  "D:\...\config_20260611.json" `
    --imgsz 1600 `
    --tif-dirs "G:\...\110年度(北區)" "G:\...\109年度(東區)" "G:\...\111年度(香山區)" `
    --bigmap
```

重點參數：

- `--tif-dirs`：大圖 TIF／定位檔來源資料夾（**遞迴搜尋，依序先找到先用，正確年度放最前面**）。
  提供後自動執行步驟 2（TFW 對應）與步驟 3（旋轉偵測）。
- `--bigmap`：最後產出大圖驗證圖（`--bigmap-scale` 控制縮放，預設 0.5）。
- 不給 `--tif-dirs` 就只跑推論；`--skip-geojson` 強制只推論。
- 多個來源資料夾（如 109-1、109-2…）各跑一次，輸出互不干擾。

輸出結構：

```
D:\output\110-1\
├─ json\               # [1] YOLO JSON + 標註圖（{圖名}__{模型}.json/.jpg）
├─ _tfw\               # [2] 對應改名後的定位檔暫存
├─ rotation_map.json   # [3] 旋轉切片方向對照表
├─ geojson\            # [4] GeoJSON（EPSG:4326 經緯度）
└─ bigmap\             # [5] {圖幅}__validation.jpg（多切片合併、白框標切片範圍）
```

## 2. 分步執行（除錯或單獨重跑）

```powershell
# [1] 推論：資料夾批次（jpg/png/bmp/tif），每張圖輸出 JSON + 標註圖
python app.py -i <影像資料夾> -o <out>\json -m <模型.pt> -c <config.json> --imgsz 1600

# [2] TFW 對應：依 JSON 檔名找定位檔，複製改名到暫存資料夾
python app_utils\tool\stage_tfw_files.py --json-dir <out>\json `
    --tfw-dirs <年度資料夾...> --output <out>\_tfw

# [3] 旋轉偵測：影像比對判定每個旋轉切片的方向（cw/ccw）
python app_utils\tool\detect_slice_rotation.py --json-dir <out>\json `
    --png-dir <影像資料夾> --tif-dirs <年度資料夾...> --output <out>\rotation_map.json

# [4] GeoJSON 轉換（TWD97 EPSG:3826 -> WGS84）
python app_utils\tool\batch_tfw2lonlat.py --json-dir <out>\json --output-dir <out>\geojson `
    --tfw-dir <out>\_tfw --src-epsg 3826 --rotation-map <out>\rotation_map.json

# [5] 大圖驗證：同圖幅切片合併繪回 TIF
python app_utils\tool\draw_geojson_on_bigmap.py --geojson-dir <out>\geojson `
    --tif-dirs <年度資料夾...> --output-dir <out>\bigmap --scale 0.5
```

## 3. 檔名規則（整條管線的核心約定）

切片 PNG 必須命名為：

```
{圖幅編號}[_o]_{切片序號}_{crop_x}_{crop_y}_{WxH}.png
例： 3074571_o_0001_5932_4262_1920x1080.png
```

- `圖幅編號`：對應定位檔 `3074571.tfw`；`_o` 為「接邊外擴」版本（有自己的 `3074571_o.tfw`）
- `crop_x / crop_y`：切片在**原始大圖上的像素偏移**（GeoJSON 轉換的關鍵）
- `WxH`：在**原始大圖上裁切的真實區域尺寸**（旋轉偵測依據此值）

推論輸出會自動沿用：`{切片檔名}__{模型}.json`。

## 4. 旋轉切片（重要陷阱）

部分切片是「**直式裁切（檔名 `_1080x1920`）、旋轉 90° 後以橫式儲存**」。
推論座標基於儲存方向，轉 GeoJSON 前必須反旋轉，否則物件在地理上整批錯位
（多數仍落在圖幅內，屬於**無聲錯誤**，只有大圖驗證看得出來）。

- 偵測規則：檔名 WxH 與 JSON 記錄的實際影像尺寸**恰好對調** → 旋轉切片
- **方向無法由座標判斷**，且不同批次方向可能不同
  （實例：109-x/110-1/111-1 全為順時針、110-2 全為逆時針）
- `detect_slice_rotation.py` 以影像內容比對逐切片判定方向，
  輸出含 diff 與信心比值；標記「低信心」的切片請人工確認
- 轉換時不帶 `--rotation-map` 的旋轉切片會**預設視為順時針並印出警告**

## 5. 常見問題

| 問題 | 處理 |
|---|---|
| `[MISS] 找不到對應定位檔` | 確認 `--tif-dirs` 是否涵蓋該年度；`_o` 切片的定位檔在「接邊外擴製作」子資料夾 |
| 同圖幅跨年度重複測製 | `--tif-dirs` 把正確年度放最前（先找到先用）；「以年度較新為準」 |
| 大圖驗證出現「座標超出影像範圍」警告 | 通常是旋轉方向錯誤或 tfw 配對錯誤，重跑步驟 3 並檢查低信心切片 |
| GeoJSON 裡的環形 lane line | 閉合 LineString（首尾同點），非 Polygon；帶洞多邊形輸出 `[外環, 洞...]` |
| EPSG | 新竹市正射圖為 TWD97（EPSG:3826），其他來源用 `--src-epsg` 調整 |

## 6. 相關測試

```powershell
python -m pytest tests\test_donut_polygon.py tests\test_geojson_rotation.py tests\test_polygon_step_order.py -q
```

涵蓋：帶洞多邊形（甜甜圈）、環形 lane line、旋轉切片座標反推（與 cv2.rotate 互為反函數的機器驗證）、GeoJSON 帶洞輸出與舊格式相容。
