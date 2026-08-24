# TIF Folder → GeoJSON 串流式 Pipeline 指南

`run_tif_pipeline.py` 是 [run_pipeline.py](../run_pipeline.py)（吃外部預切 PNG 的 5 階段版本，見 [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)）的姊妹版本：**輸入直接是大圖 TIF 資料夾 + TFW 資料夾**，pipeline 自行切片、推論、縫合、輸出 GeoJSON。

## 與舊 pipeline 的差異

| | run_pipeline.py（舊） | run_tif_pipeline.py（新） |
|---|---|---|
| 輸入 | 外部工具預切的 PNG | 大圖 TIF + TFW 資料夾 |
| 切片 | 外部（含旋轉切片） | 內建，記憶體內 numpy view，**不落地磁碟** |
| 旋轉偵測 | 需要（detect_slice_rotation.py） | 不需要（自切全部軸對齊） |
| TFW staging | 需要 | 不需要（同 stem 直接配對） |
| 切縫處理 | 無去重（重疊區物件重複） | **mask 層級縫合**：同類別像素相交即 union，斷裂物件癒合、重複偵測去重 |
| 磁碟中間產物 | 每圖幅 1–2 GB 切片 | 無（除非 `--keep-tiles`） |

## 用法

```bash
python run_tif_pipeline.py \
    --tif-dir D:/maps/tif --tfw-dir D:/maps/tfw \
    -o ./out -m best.pt -c config_20260611_opt2.json \
    --imgsz 1600 --conf 0.25
```

主要參數（其餘見 `--help`）：

- `--tile-size 1600` / `--overlap 320`：切片邊長與重疊。重疊需大於典型標線寬度，物件跨縫時兩側切片才有共同像素可縫合。邊緣切片自動 clamp 保持正方形（不 padding）。
- `--stitch-gap 1`：縫合像素容差；`overlap=0` 時貼邊物件靠此合併。
- `--src-epsg 3826`：TIF 原始座標系（預設 TWD97）。
- `-c config.json`：與 app.py CLI 相同的 config（`mask_optimizations` / `polygon_optimizations` / `class_filter`），參見 [CLI_AND_CONFIG_GUIDE.md](../CLI_AND_CONFIG_GUIDE.md) 與 [roadmaker.md](../roadmaker.md)。
- `--keep-tiles`：除錯用，切片以既有命名約定（`{stem}_{seq}_{x}_{y}_{WxH}.png`）落地，可餵給舊工具鏈檢查。
- `--bigmap` / `--bigmap-scale 0.5`：輸出縮圖疊繪驗證 JPG。
- `--stems 3074571,3074572`：只處理指定圖幅。

## 輸出結構

```
OUT/
  {stem}__{model}.geojson   # 每圖幅每模型一份合併 FeatureCollection（EPSG:4326）
  tiles/                    # 僅 --keep-tiles
  bigmap/                   # 僅 --bigmap
```

Feature properties 額外含 `tile_seqs`（物件來源切片序號，跨縫物件會有多個）；metadata 記錄 tiling 參數。

## 處理流程（每圖幅）

1. `read_big_image`（[tif_tiler.py](../app_utils/tool/tif_tiler.py)）：unicode 安全讀取，cv2 失敗時 fallback `tifffile`（BigTIFF）。整張載入記憶體，一次一張。
2. `iter_tiles`：切片為 numpy view（零複製），`is_blank` 跳過圖幅外空白領域。
3. 逐片 `model.predict` → 只抽原始 instance mask 成「稀疏全域 instance」（全域 bbox 原點 + 裁切 mask），**此時不做任何優化**。
4. `merge_instances`（[mask_stitcher.py](../app_utils/tool/mask_stitcher.py)）：同類別、像素相交（union-find）合併 —— 這一步同時解決重疊區重複偵測與跨縫斷裂。
5. `instances_to_objects`：對合併後完整 mask 跑 config 的 mask 優化步驟 → 輪廓抽取（含洞）→ polygon 優化鏈。`pca` 等跨物件步驟因此以整張圖幅為單位運作。
6. `convert_yolo_json_to_geojson`（crop=0，座標已全域）→ WorldFile 仿射 + pyproj 轉 EPSG:4326。

## 已知語意與限制

- 同類別兩個「真實相鄰接觸」的實體會被縫合成一個；config 的 mask `split` 步驟可拆回（與既有 dilate/merge 後 split 的處理哲學一致）。
- mask 優化的 `merge` 步驟只作用於單一縫合 instance 拆出的元件之間，不跨縫合後的不同 instance。
- 記憶體峰值 ≈ 一張大圖（20000×20000×3 ≈ 1.2 GB）+ 稀疏 instance；如遇超大 BigTIFF 需 windowed read，可在 `read_big_image` 內以 tifffile `aszarr` 擴充，呼叫端不需改動。

## 測試

```bash
pytest tests/test_tif_tiler.py tests/test_mask_stitcher.py tests/test_tif_pipeline_e2e.py
```

端到端測試以 stub 模型 + 合成 TIF/TFW 驗證：跨縫矩形恰好輸出一次、GeoJSON 座標與獨立 pyproj 換算吻合、不落地切片。
