"""旋轉切片方向偵測工具。

部分 HD 切片是「直式裁切（檔名 _WxH）、旋轉 90° 後以橫式儲存」，
且不同批次的旋轉方向可能不同（順時針或逆時針）。方向無法由座標判斷，
必須以影像內容比對：把儲存的切片分別往兩個方向轉回直式，
與原始大圖上 (X, Y, WxH) 區域比對，取差異小的方向。

輸出 rotation map（JSON），供 batch_tfw2lonlat.py --rotation-map 使用：
  {
    "directions": { "<切片檔名去副檔名>": "cw" | "ccw", ... },
    "details":    [ {..diff 數值與判定信心..}, ... ]
  }
  "cw"  = 儲存影像為原始區域「順時針」轉 90°
  "ccw" = 儲存影像為原始區域「逆時針」轉 90°

用法範例：
  python detect_slice_rotation.py ^
      --json-dir "D:/.../20260611/110-2" ^
      --png-dir  "D:/.../HD-DS/110-2" ^
      --tif-dirs "G:/.../110年度(北區)" "G:/.../109年度(東區)" "G:/.../111年度(香山區)" ^
      --output   "D:/.../20260611/rotation_map_110-2.json"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_DIM_TOKEN_PATTERN = re.compile(r"_(\d+)x(\d+)(?:\.|_|$)")
_SLICE_PATTERN = re.compile(r"^(?P<stem>.+?)_(?P<slice>\d+)_(?P<x>\d+)_(?P<y>\d+)_(?P<w>\d+)x(?P<h>\d+)$")

# 兩方向 diff 比值需達此倍數才視為可信判定
MIN_CONFIDENCE_RATIO = 2.0


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _build_tif_index(dirs: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            print(f"[Warning] 找不到資料夾，略過: {d}")
            continue
        for p in sorted(d.rglob("*.tif")):
            index.setdefault(p.stem.lower(), p)
        for p in sorted(d.rglob("*.tiff")):
            index.setdefault(p.stem.lower(), p)
    return index


def detect_directions(json_dir: Path, png_dir: Path, tif_dirs: List[Path]):
    tif_index = _build_tif_index(tif_dirs)
    tif_cache: Dict[str, Optional[np.ndarray]] = {}

    directions: Dict[str, str] = {}
    details: List[dict] = []

    for jp in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = data.get("image") or {}
        fname = str(info.get("file_name") or "")
        dm = _DIM_TOKEN_PATTERN.search(fname)
        if not dm:
            continue
        w_name, h_name = int(dm.group(1)), int(dm.group(2))
        img_w, img_h = info.get("width"), info.get("height")
        if w_name == h_name or (img_w, img_h) != (h_name, w_name):
            continue  # 非旋轉切片

        png_stem = fname.rsplit(".", 1)[0]
        sm = _SLICE_PATTERN.match(png_stem)
        if not sm:
            print(f"[Warning] 無法解析切片檔名: {png_stem}")
            continue
        map_stem = sm.group("stem")
        crop_x, crop_y = int(sm.group("x")), int(sm.group("y"))

        png = _imread_unicode(png_dir / f"{png_stem}.png")
        if png is None:
            print(f"[Error] 找不到切片影像: {png_dir / (png_stem + '.png')}")
            continue

        if map_stem not in tif_cache:
            tif_path = tif_index.get(map_stem.lower())
            tif_cache[map_stem] = _imread_unicode(tif_path) if tif_path else None
        tif = tif_cache[map_stem]
        if tif is None:
            print(f"[Error] 找不到大圖 TIF: {map_stem}")
            continue

        region = tif[crop_y:crop_y + h_name, crop_x:crop_x + w_name]
        if region.shape[:2] != (h_name, w_name):
            print(f"[Warning] 區域超出大圖範圍: {png_stem} -> {region.shape}")
            continue

        region16 = region.astype(np.int16)
        # 逆時針轉回一致 => 儲存時為「順時針」；反之為「逆時針」
        diff_back_ccw = float(np.mean(np.abs(
            cv2.rotate(png, cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.int16) - region16)))
        diff_back_cw = float(np.mean(np.abs(
            cv2.rotate(png, cv2.ROTATE_90_CLOCKWISE).astype(np.int16) - region16)))

        if diff_back_ccw <= diff_back_cw:
            direction = "cw"
            ratio = diff_back_cw / max(diff_back_ccw, 1e-6)
        else:
            direction = "ccw"
            ratio = diff_back_ccw / max(diff_back_cw, 1e-6)

        confident = ratio >= MIN_CONFIDENCE_RATIO
        directions[png_stem] = direction
        details.append({
            "slice": png_stem,
            "direction": direction,
            "diff_if_cw": round(diff_back_ccw, 2),
            "diff_if_ccw": round(diff_back_cw, 2),
            "confidence_ratio": round(ratio, 1),
            "confident": confident,
        })
        flag = "" if confident else "  <-- 低信心，建議人工確認"
        print(f"[{direction.upper():>3}] {png_stem} (ratio={ratio:.1f}){flag}")

    return directions, details


def main():
    parser = argparse.ArgumentParser(
        description="以影像比對偵測旋轉切片的旋轉方向，輸出 rotation map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json-dir", required=True, type=Path, help="推論輸出 JSON 資料夾")
    parser.add_argument("--png-dir", required=True, type=Path, help="原始切片 PNG 資料夾")
    parser.add_argument("--tif-dirs", nargs="+", required=True, type=Path,
                        help="大圖 TIF 搜尋資料夾（遞迴；依序先找到先用）")
    parser.add_argument("--output", required=True, type=Path, help="輸出 rotation map JSON")
    args = parser.parse_args()

    directions, details = detect_directions(args.json_dir, args.png_dir, args.tif_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"directions": directions, "details": details}, f, ensure_ascii=False, indent=2)

    n_cw = sum(1 for v in directions.values() if v == "cw")
    n_ccw = len(directions) - n_cw
    low_conf = sum(1 for d in details if not d["confident"])
    print("-" * 40)
    print(f"[Success] 旋轉切片 {len(directions)} 個（cw={n_cw}, ccw={n_ccw}，低信心 {low_conf} 個）")
    print(f"[Info] rotation map 已輸出: {args.output}")


if __name__ == "__main__":
    main()
