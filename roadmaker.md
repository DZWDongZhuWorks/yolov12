# RoadMaker 優化步驟操作指南

道路標線向量化管線的 **Mask 優化** 與 **Polygon 優化** 各步驟的功能、參數與使用時機。
所有建議都附 109-2 實測數據佐證(47 張、~6000 物件、imgsz=1600)。

---

## 1. 管線總覽

```
YOLO 分割
   │  輸出：每個實例一張機率遮罩 (mask)
   ▼
┌─────────────── ① Mask 優化（在「像素遮罩」上動刀）────────────────┐
│  erode / dilate / distance_* / blur / remove_small / fill_holes   │
│  split / merge                                                    │
│  目的：清理遮罩雜訊、補洞、分裂或合併實例                          │
└───────────────────────────────────────────────────────────────────┘
   │  遮罩 → 輪廓（外環 + 洞）
   ▼
┌─────────────── ② Polygon 優化（在「向量幾何」上動刀）────────────┐
│  convex_hull / rdp / visvalingam_whyatt / min_area_rect /         │
│  export_line / polygon_to_lane_line / small_object_fit /          │
│  smooth_spline / fit_lines_arcs / pca                             │
│  目的：簡化點數、平滑去鋸齒、擬合直線/弧、抽中心線、方向對齊       │
└───────────────────────────────────────────────────────────────────┘
   │
   ▼
JSON（影像座標）→ tfw2lonlat → GeoJSON
```

兩階段都是**逐步驟、依序套用**,且每個步驟都可用 `classes` 欄位**只作用於指定類別**(空=全部)。

---

## 2. Mask 優化步驟

遮罩階段欄位順序:`step, enabled, count, morph_kernel, blur_kernel, blur_threshold, min_component_area, max_hole_area, merge_iou_threshold, classes`

| 步驟 | 功能 | 主要參數 | 使用時機 |
|---|---|---|---|
| **erode** | 形態學侵蝕,遮罩往內縮 | `morph_kernel` | 標線被預測得偏胖、相鄰標線黏在一起時瘦身 |
| **dilate** | 形態學膨脹,遮罩往外長 | `morph_kernel` | 標線斷裂、太細、有小缺口時補強 |
| **distance_erode** | 距離轉換式侵蝕(較均勻) | `morph_kernel`(半徑=kernel/2) | 要等距內縮、比 erode 更平滑可控時 |
| **distance_dilate** | 距離轉換式膨脹 | `morph_kernel` | 要等距外擴、連接近距斷點時 |
| **blur** | 高斯模糊遮罩後再二值化 | `blur_kernel`, `blur_threshold` | **遮罩邊緣鋸齒/毛邊**,先在像素層柔化(治本的一種) |
| **remove_small** | 移除面積 < 門檻的連通元件 | `min_component_area` | **清掉零散雜點/誤判小塊**(常用,如本專案設 100) |
| **fill_holes** | 填補面積 ≤ 門檻的內部破洞 | `max_hole_area` | 標線內部有小破洞、想補實(注意:會吃掉真正的甜甜圈空心) |
| **split** | 把一張含多個分離塊的遮罩拆成多個獨立實例 | (無) | 一個偵測框內含多條不相連標線,要各自成物件時 |
| **merge** | 同類別、IoU ≥ 門檻的實例合併 | `merge_iou_threshold` | 同一條標線被切成多個重疊實例,要併回一條時 |

> 一般起手式:`split → remove_small`(拆開 + 去雜點),即本專案 config 的設定。

---

## 3. Polygon 優化步驟

向量階段欄位順序:`step, enabled, count, eps_coeff, min_aspect, pca_min_cosine, pca_cross_class, max_area_px, target_vertices, classes`

### 3.1 兩大類本質(最重要的觀念)

- **抽點 / 簡化(decimation)**:`rdp`、`visvalingam_whyatt`、`convex_hull`
  從原始點裡**挑子集**。點變少,但**留下的點仍釘在原本的像素階梯上 → 微觀依然鋸齒**。
- **擬合 / 平滑(fitting)**:`smooth_spline`、`fit_lines_arcs`、`min_area_rect`、`export_line`、`polygon_to_lane_line`
  把點**移到模型上**(直線/弧/曲線/中心線)→ 真正去鋸齒、得到平整幾何。

> 想消鋸齒,要用「擬合」類,不是「簡化」類。

### 3.2 各步驟

| 步驟 | 功能 | 主要參數 | 使用時機 / 備註 |
|---|---|---|---|
| **rdp** | Douglas–Peucker 抽點簡化 | `eps_coeff`(大=點更少、偏差更大) | 通用點數壓縮;直邊形狀夠用。**仍會殘留鋸齒** |
| **visvalingam_whyatt** | 依三角面積抽點簡化 | `eps_coeff` | 比 rdp 保留更多點、偏差更小,但**慢**、保點多 |
| **convex_hull** | 取凸包 | (無) | 只要外接凸輪廓、不在意凹處時 |
| **min_area_rect** | 用最小面積外接矩形取代 | `min_aspect`(長寬比門檻,0=一律) | 想把物件硬框成矩形 |
| **export_line** | 取長軸,輸出一條**直線**(2 點) | `min_aspect` | **筆直的長線**(白/黃線);彎的會被拉直而失真 |
| **polygon_to_lane_line** | 抽**中心線**(中軸/中點配對) | `eps_coeff` | **車道線、導引線**等帶狀標線轉成單線。**中心線本身偏鋸齒**,常需再平滑 |
| **small_object_fit** | 二分搜尋 eps,壓到目標頂點數、**保留凹凸** | `max_area_px`(面積>此值則略過)、`target_vertices` | **小符號**(箭頭、菱形、三角、文字)。精簡向量;8 頂點對箭頭偏少會削尖角 |
| **smooth_spline** ⭐ | B-spline 近似平滑(scipy),失敗回退 Chaikin | `eps_coeff`(大=更平滑、位移更大) | **彎曲帶狀**標線去鋸齒(尤其接在 polygon_to_lane_line 後平滑中心線)。**會磨圓銳角** |
| **fit_lines_arcs** ⭐ | RDP 找角 + 段內擬合**直線/圓弧** | `eps_coeff`(找角容差) | **直邊封閉環**(crosswalk/mesh/parking/rectangle)、直線+弧線標線。**保留銳角**、把鋸齒邊拉成乾淨直線 |
| **pca** | 線段方向分群、吸附到主軸 | `eps_coeff`(=α,0~1 吸附強度)、`pca_min_cosine`、`pca_cross_class` | 讓**近平行**的線變整齊(白/黃線)。⚠️ 勿開 Manhattan 正交吸附(會扭曲斜交路口) |

> `eps_coeff` 在多數步驟是「size 相對」的強度(`eps = 物件尺寸 × 0.01 × eps_coeff`);唯獨 **pca** 的 `eps_coeff` 是吸附比例 α(0~1)。

---

## 4. 排序規則與反模式(踩過的雷)

1. **平滑/擬合要排在 `rdp`/`visvalingam` 之前**(或取代它)。
   它們需要**稠密點**;若先 rdp 把點抽光,後面再平滑就沒材料了。
2. **🚫 不要 `rdp → smooth_spline`**。
   對稀疏的封閉環套樣條,會把直角**磨圓、鼓包**。實測直邊封閉環偏差從 ~4px **爆增到 41~156px**。
3. **`smooth_spline` 會磨圓真實銳角** → 只給**彎曲帶狀**類;箭頭/文字/直邊**不要**用。
4. **直邊封閉環殘留鋸齒** → 用 **`fit_lines_arcs`**(它先找角再把邊擬合成直線),不是 smooth_spline。
5. **`fit_lines_arcs` 不需要全域取代 `rdp`**:對純直邊矩形/格子,兩者差異小;它的價值在**把鋸齒的直邊/弧線拉乾淨**。
6. **sub-pixel 輪廓疊在 rdp/fit_lines_arcs 後幾乎無感**(簡化已吸收掉階梯,實測僅 −1~2%);只在「保留稠密輪廓」或需要次像素精度時才開。
7. **🚫 Manhattan 正交吸附對道路標線會失真**:道路常斜交/彎曲,強制垂直/平行會把線轉離真實油漆方向(實測 51% 線被旋轉 >2°)。維持一般 `pca` 即可。

---

## 5. 依標線型態的建議流程(已驗證)

| 標線型態 | 類別範例 | 建議步驟 | 為何 |
|---|---|---|---|
| **彎曲帶狀 / 車道線** | `_lane_line` 類(white/yallow/red_lane_line)、導引線、各式可能轉彎的 lane line | `polygon_to_lane_line` → **`smooth_spline`** | 抽中心線後**平滑掉鋸齒**(實測鋸齒 **−55.6%**,形狀不偏) |
| **直邊封閉環** | crosswalk、mesh_line、parking-grid、rectangle | **`fit_lines_arcs`** | 直邊拉直、保留直角(實測最壞鋸齒 **−15~26%**,其餘類別零影響) |
| **筆直長線** | `_line` 類:white_line、yallow_line | `export_line` → `pca` | 取長軸直線 + 近平行對齊。⚠️ 不開 Manhattan |
| **小符號** | 箭頭、菱形、三角、文字、X 標 | `small_object_fit` | 精簡保形;**箭頭**可把 `target_vertices` 8→14 改善尖角/凹口 |

> **命名規則判斷**:`_line` 結尾(white_line、yallow_line)= **兩點一直線** → 走 `export_line`;`_lane_line` 結尾(white/yallow/red_lane_line)= **可能轉彎** → 走 `polygon_to_lane_line`。兩者**不要混放**同一群,否則直線會被 lane 中心線邏輯多餘地重算端點。

---

## 6. 實證摘要(109-2)

| 比較 | 結果 |
|---|---|
| 車道中心線 `polygon_to_lane_line` + `smooth_spline` | 鋸齒(總轉角)**−55.6%**,中心線平均位移 <1px,非車道類別 100% 不變 |
| 直邊封閉環 `rdp` → `fit_lines_arcs` | 最壞偏差:rectangle 4.46→**3.29**、parking 9.04→**6.88**;點數 +23% |
| 直邊封閉環 `rdp → smooth_spline`(反模式) | 偏差暴增 4→**41**、9→**156** px(鼓包) ❌ |
| 直邊封閉環疊 sub-pixel | 鋸齒僅 **−1~2%**,不划算 |
| 白/黃線開 Manhattan | 51% 線旋轉 >2°、max 18°,**扭曲真實方向** ❌ |

---

## 7. 完整範例設定(本專案驗證版)

> 對應 `config_20260611_opt2.json`。重點:車道線抽中心線後平滑、直邊封閉環用 fit_lines_arcs、白黃線取直線對齊、小符號精簡。

```json
{
  "mask_optimizations": {
    "enabled": true,
    "steps": [
      ["split",        true, 1, 0, 0, 0,   0, 0, 0, ""],
      ["remove_small", true, 1, 0, 0, 0, 100, 0, 0, ""]
    ]
  },
  "polygon_optimizations": {
    "enabled": true,
    "steps": [
      ["fit_lines_arcs",       true, 1, 1.0, 0, 0,    false, 0, 0, "17,31,32,33,34,36,37,38"],
      ["rdp",                  true, 1, 1.0, 0, 0,    false, 0, 0, "18,19"],
      ["export_line",          true, 1, 0,   0, 0,    false, 0, 0, "0,4"],
      ["pca",                  true, 1, 1.0, 0, 0.94, false, 0, 0, "0,4"],
      ["polygon_to_lane_line", true, 1, 1.0, 0, 0,    false, 0, 0, "1,2,3,5,6,7,8,9,10,11,12,13,14,15"],
      ["smooth_spline",        true, 1, 0.5, 0, 0,    false, 0, 0, "1,2,3,5,6,7,8,9,10,11,12,13,14,15"],
      ["small_object_fit",     true, 1, 0,   0, 0,    false, 5000, 8, "20,21,22,24,25,26,27,28,30,35,39"]
    ]
  }
}
```

> 微調建議:
> - 車道線想更平滑 → `smooth_spline` 的 `eps_coeff` 0.5→1.0(會放大位移,大物件留意)。
> - 箭頭尖角被削 → `small_object_fit` 對箭頭那組 `target_vertices` 8→14。

---

## 8. 對比評估工具

落地前可先量化/目視比較(內建批次彙整):

```bash
# 單張:疊圖 + 偏差表
python -m app_utils.polygon_comparison --weights best.pt --image a.png --imgsz 1600 --out cmp

# 整個資料夾平均(跨圖彙整偏差,寫 batch_report.md)
python -m app_utils.polygon_comparison --weights best.pt --image-dir ./tiles --imgsz 1600 --out cmp_dir
```

偏差表欄位:總點數、簡化率、面積保留率、**與稠密輪廓的 max/mean 偏差(px)**、處理時間——數字越小越忠實。
