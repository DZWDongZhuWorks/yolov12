# YOLOv12 推論介面：配置匯出入與批次命令列支援

本專案 `app.py` 已經整合了 **設定檔（Configuration）匯出入機制** 以及 **純命令列介面（CLI）的批次推論功能**。
這讓使用者可以在友善的圖形介面（Web UI）中反覆除錯並找出最佳推論與顯示參數後，將設定打包保存，接著在伺服器或背景終端機中，針對大量資料夾或圖片進行穩定、自動化的推論處理。

---

## 一、 Web UI：配置管理 (Config) 匯出與匯入

在啟動圖形介面後（執行 `python app.py` 且不加 `--input` 參數），畫面的控制面板中新增了 **「配置管理 (Config)」** 頁籤。

### 1. 匯出目前配置
當您在「顯示 / 執行」、「Mask 優化」與「Polygon 優化」介面完成以下參數的微調後：
* **視覺顯示參數**：諸如標籤模式 (Label Mode)、外框顯示、Mask 顯示、Polygon 顯示、信心值標示等狀態。
* **類別篩選 (Class Filtering)**：有勾選要保留或剔除哪些模型偵測類別。
* **優化步驟表 (Dataframe)**：包含 Mask 開關與順序、Polygon 簡化（如 RDP、Visvalingam）與方向分群對齊等自訂流程。

可以直接點擊 **「匯出目前配置」** 按鈕。系統會自動產生一個 `.json` 格式的配置檔案供您下載。

### 2. 匯入配置檔 (.json)
若日後需要載入同一組複雜的優化參數，您可以直接將前述的 `.json` 檔案上傳到 **「匯入配置檔 (.json)」** 區塊。 
* 系統將立即還原面板上的各項參數數值（含 checkbox 與表格流程）。
* 為避免載入過程中因圖片或模型尚未備妥引起錯誤，**參數還原並不會強制立刻觸發畫面重繪**，請在確認各項設定無誤後，手動點擊推理按鈕或繼續後續操作。

---

## 二、 CLI 命令列：無介面批次推論

結合被匯出的 JSON 參數檔，您可以透過指令列將 YOLOv12 腳本轉為批次推論工具，它將自動讀取所有圖片作推論、套用多邊形優化，然後匯出對應的 `JSON 標註結果` 與 `標註視覺化圖片`。

### 基本指令範例

```bash
python app.py --input ./my_images_folder --output ./my_results --config my_config.json --models yolov12m.pt
```
*備註：一旦給予了 `--input` 參數，程式將自動進入 CLI 模式而不會啟動 Gradio 網頁介面。*

### 參數詳解

| 參數 | 縮寫 | 說明 | 預設值 |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | **(進入 CLI 必填)** 輸入的圖片檔案路徑或是包含多張圖片的**資料夾路徑**。 | `None` |
| `--config` | `-c` | **(CLI 必填)** 從 UI 產生的組態配置 JSON 檔案路徑。必須具備此擋案，程式才知道要套用何種優化。 | `None` |
| `--output` | `-o` | 輸出標註結果與對比圖片的資料夾目的地。資料夾若不存在會自動建立。 | `./output` |
| `--models` | `-m` | 欲執行的 YOLOv12 模型，可撰寫模型檔名或絕對路徑。若需跑「多模型比較」，請用逗號隔開 (如：`yolov12m.pt,yolov12s.pt`) | `yolov12m.pt` |
| `--imgsz` | | 推論輸入圖片尺寸 (Image size)。 | `640` |
| `--conf` | | 基礎偵測的信心值門檻 (Confidence threshold)。 | `0.25` |
| `--device` | | 運算硬體指令，例如 `cpu`, `cuda:0`, `mps` 等。 | `auto` |
| `--save-img` | | **[預設開啟]** 旗標。在輸出目錄中，除了產生 `.json` 標註檔案之外，也順便存下帶著所有視覺設定（如 Bbox、Polygon）的 `.jpg` 圖檔。 | `True` |
| `--no-save-img`| | 旗標。加上此參數將**關閉**影像繪製儲存，僅匯出 JSON 檔案以提升大批次運算的整體速度。 | `False` |

### 輸出結果展示

假設您針對一個圖像 `/my_images/test_01.jpg` 使用了 `--models yolov12m.pt` 進行推論，當處理完成後，`--output` 資料夾中將會產生：
1. **`test_01__yolov12m.json`**：經過所有 Mask 與 Polygon 優化階段處理後，乾淨且被過濾好的邊界多邊形頂點做為最終物件紀錄。
2. **`test_01__yolov12m.jpg`**（若無關閉 `--save-img`）：根據 Config 設定中所決定是否隱藏的標籤與邊界畫布。

這套操作流程為您的電腦從「參數調研」到「落地批次執行」帶來了最完全的自動化支援！

---

## 三、 Polygon 優化步驟：`small_object_fit`（小物件形狀貼合）

針對**菱形、倒三角形、箭頭、道路標字**等小型地面標線設計的 polygon 優化步驟。其他既有步驟（`rdp`、`min_area_rect`、`polygon_to_lane_line` 等）都是為長條形/矩形/線狀物件設計的；`small_object_fit` 則專門處理小物件：保留原本的凹凸形狀，但把點數壓到指定目標值。

### 運作邏輯

1. **大小門檻**：若 polygon 的 bbox 面積 > `max_area_px`（且 `max_area_px > 0`），原樣回傳（不動到大物件，例如車道線）。`max_area_px = 0` 表示完全不檢查。
2. **二分搜尋 epsilon**：在 `cv2.approxPolyDP` 上二分搜尋，把點數壓到 `[3, target_vertices]`。
3. **保留凹凸**：不做 convex hull，確保箭頭凹口與「T」字凹角等語意點得以存活。

### 專屬參數（dataframe 第 8、9 欄）

| 參數 | 預設 | 說明 |
| :--- | :--- | :--- |
| `max_area_px` | `5000.0` | bbox 面積上限（像素²）；0 = 不檢查 |
| `target_vertices` | `8` | 目標頂點數上限（3 ~ 32） |

### Config JSON 範例

Dataframe 欄位順序：`step, enabled, count, eps_coeff, min_aspect, pca_min_cosine, pca_cross_class, max_area_px, target_vertices, classes`

```json
{
  "polygon_optimizations": {
    "enabled": true,
    "steps": [
      ["small_object_fit", true, 1, 1.0, 0.0, 0.94, false, 5000.0, 8, "12, 13, 14, 15"]
    ]
  }
}
```

複合配置：箭頭與菱形先凸化再壓點、文字用較大目標頂點數保留筆畫：

```json
{
  "polygon_optimizations": {
    "enabled": true,
    "steps": [
      ["convex_hull",      true, 1, 1.0, 0.0, 0.94, false, 5000.0, 8,  "arrow, rhombus"],
      ["small_object_fit", true, 1, 1.0, 0.0, 0.94, false, 5000.0, 6,  "arrow, rhombus"],
      ["small_object_fit", true, 1, 1.0, 0.0, 0.94, false, 8000.0, 16, "road_text"]
    ]
  }
}
```

> 舊版 8 欄 config 仍可被自動讀入並補上預設的 `max_area_px = 5000.0` 與 `target_vertices = 8`。
