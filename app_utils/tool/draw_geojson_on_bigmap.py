"""最終驗證工具：將零星切片的 GeoJSON 推論結果重新繪製回原始大圖（TIF）。

同一張大圖（圖幅）的多個推論切片會合併繪製在同一張輸出圖上，
並以白框標出每個切片的裁切範圍，方便核對 GeoJSON 座標是否落回正確位置。

繪製邏輯重用 draw_geojson.py 的 draw_on_image（經緯度 -> 投影座標 -> 大圖全域像素，
crop 偏移 = 0），切片範圍則由 GeoJSON 檔名的 {圖幅}_{序號}_{X}_{Y}_{WxH} 規則解析。

用法範例：
  python draw_geojson_on_bigmap.py ^
      --geojson-dir "D:/.../geojson/109-1" ^
      --tif-dirs "G:/.../109年度(東區)" "G:/.../110年度(北區)" "G:/.../111年度(香山區)" ^
      --output-dir "D:/.../bigmap/109-1"

  * --tif-dirs 依序搜尋（遞迴），先找到先用 → 把正確年度放最前面
  * --tfw-dirs 未指定時沿用 --tif-dirs
  * 同一圖幅跨資料夾的切片要合併時，把多個 --geojson-dir 一起傳入即可
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from pyproj import Transformer

from app_utils.tool.draw_geojson import WorldFile, draw_on_image

# {圖幅stem}_{切片序號}_{crop_x}_{crop_y}_{WxH}[__模型].json
SLICE_PATTERN = re.compile(
    r"^(?P<stem>.+?)_(?P<slice>\d+)_(?P<x>\d+)_(?P<y>\d+)_(?P<w>\d+)x(?P<h>\d+)(?:__.*)?\.json$"
)
# 後備：{stem}_{X}_{Y}[後綴].json（無 WxH，無法畫切片框）
LOOSE_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<x>\d+)_(?P<y>\d+)(?:_.*)?\.json$")

SLICE_BOX_COLOR = (255, 255, 255)  # 白色切片框
SLICE_BOX_THICKNESS = 4


def build_file_index(dirs: List[Path], exts: Tuple[str, ...]) -> Dict[str, Path]:
    """遞迴建立 {stem(小寫): 路徑} 索引；dirs 依序處理，先找到先用。"""
    index: Dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            print(f"[Warning] 找不到資料夾，略過: {d}")
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in exts:
                index.setdefault(p.stem.lower(), p)
    return index


def collect_geojsons(geojson_dirs: List[Path]) -> Dict[str, List[dict]]:
    """掃描 GeoJSON 並依圖幅 stem 分組。"""
    groups: Dict[str, List[dict]] = {}
    for d in geojson_dirs:
        for p in sorted(d.glob("*.json")):
            m = SLICE_PATTERN.match(p.name)
            if m:
                info = {
                    "path": p,
                    "slice": m.group("slice"),
                    "x": int(m.group("x")), "y": int(m.group("y")),
                    "w": int(m.group("w")), "h": int(m.group("h")),
                }
                stem = m.group("stem")
            else:
                m2 = LOOSE_PATTERN.match(p.name)
                if not m2:
                    print(f"[Warning] 檔名無法解析切片資訊，略過: {p.name}")
                    continue
                info = {"path": p, "slice": None,
                        "x": int(m2.group("x")), "y": int(m2.group("y")),
                        "w": None, "h": None}
                stem = m2.group("stem")
            groups.setdefault(stem, []).append(info)
    return groups


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray, quality: int) -> bool:
    ok, buf = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def process_stem(
    stem: str,
    items: List[dict],
    tif_index: Dict[str, Path],
    tfw_index: Dict[str, Path],
    transformer: Transformer,
    out_dir: Path,
    draw_labels: bool,
    scale: float,
    quality: int,
) -> bool:
    tif_path = tif_index.get(stem.lower())
    tfw_path = tfw_index.get(stem.lower())
    if tif_path is None:
        print(f"[Error] {stem}: 找不到大圖 TIF，略過（{len(items)} 個切片）")
        return False
    if tfw_path is None:
        print(f"[Error] {stem}: 找不到定位檔 (.tfw/.jgw)，略過")
        return False

    t0 = time.perf_counter()
    image = imread_unicode(tif_path)
    if image is None:
        print(f"[Error] {stem}: 大圖讀取失敗 {tif_path}")
        return False

    wf = WorldFile.from_file(tfw_path)

    # 1) 逐切片把 GeoJSON 畫回大圖（全域像素 → crop 偏移為 0）
    for info in items:
        with info["path"].open("r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        draw_on_image(image, geojson_data, wf, transformer, 0.0, 0.0, draw_labels)

    # 2) 標出每個切片的裁切範圍與名稱
    for info in items:
        if info["w"] is None:
            continue
        x, y, w, h = info["x"], info["y"], info["w"], info["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), SLICE_BOX_COLOR, SLICE_BOX_THICKNESS)
        tag = f"{info['slice'] or ''} ({x},{y})"
        cv2.putText(image, tag, (x + 8, max(40, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(image, tag, (x + 8, max(40, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, SLICE_BOX_COLOR, 2, cv2.LINE_AA)

    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    out_path = out_dir / f"{stem}__validation.jpg"
    if not imwrite_unicode(out_path, image, quality):
        print(f"[Error] {stem}: 影像存檔失敗")
        return False

    print(f"[OK] {stem}: {len(items)} 個切片 -> {out_path.name} "
          f"({image.shape[1]}x{image.shape[0]}, {time.perf_counter() - t0:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="將切片推論的 GeoJSON 合併繪製回原始大圖 TIF（最終驗證）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--geojson-dir", nargs="+", required=True, type=Path,
                        help="GeoJSON 資料夾（可多個，會合併同圖幅的切片）")
    parser.add_argument("--tif-dirs", nargs="+", required=True, type=Path,
                        help="大圖 TIF 搜尋資料夾（遞迴；依序先找到先用）")
    parser.add_argument("--tfw-dirs", nargs="*", type=Path, default=None,
                        help="定位檔搜尋資料夾（未指定時沿用 --tif-dirs）")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--src-epsg", type=int, default=3826, help="TFW 投影座標 EPSG")
    parser.add_argument("--no-label", action="store_true", help="不繪製類別標籤")
    parser.add_argument("--scale", type=float, default=1.0, help="輸出縮放倍率（如 0.5）")
    parser.add_argument("--quality", type=int, default=90, help="輸出 JPG 品質")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tfw_dirs = args.tfw_dirs if args.tfw_dirs else args.tif_dirs

    print("[Info] 建立 TIF / TFW 檔案索引...")
    tif_index = build_file_index(args.tif_dirs, (".tif", ".tiff"))
    tfw_index = build_file_index(tfw_dirs, (".tfw", ".jgw"))
    print(f"[Info] TIF: {len(tif_index)} 張, 定位檔: {len(tfw_index)} 個")

    groups = collect_geojsons(args.geojson_dir)
    total_slices = sum(len(v) for v in groups.values())
    print(f"[Info] GeoJSON 切片 {total_slices} 個，分屬 {len(groups)} 張大圖")

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{args.src_epsg}", always_xy=True)

    ok = 0
    for stem in sorted(groups):
        if process_stem(stem, groups[stem], tif_index, tfw_index, transformer,
                        args.output_dir, not args.no_label, args.scale, args.quality):
            ok += 1

    print("-" * 40)
    print(f"[Success] 完成 {ok}/{len(groups)} 張大圖。輸出: {args.output_dir}")


if __name__ == "__main__":
    main()
