"""TIF Folder -> GeoJSON 串流式 Pipeline（mask 層級縫合）。

輸入：大圖 TIF 資料夾 + TFW 資料夾（同 stem 配對）。
輸出：每張圖幅、每個模型一份合併的 GeoJSON（EPSG:4326）。

與 run_pipeline.py（吃外部預切 PNG 的 5 階段版本）不同，本 pipeline：
1. 自行在記憶體內切片（預設 1600x1600、重疊 320px），切片不落地磁碟。
2. 各切片只取原始 instance mask，於全域像素座標縫合（同類別像素相交即
   union）—— 跨切片斷裂的物件自然癒合、重疊區重複偵測自然去重。
3. 縫合完成後才對完整 mask 跑 mask/polygon 優化鏈（config 與 app.py 相同），
   再經 WorldFile + pyproj 轉為經緯度。

用法範例：
    python run_tif_pipeline.py --tif-dir D:/maps/tif --tfw-dir D:/maps/tfw \
        -o ./out -m best.pt -c config.json --imgsz 1600
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from pyproj import Transformer

sys.path.append(str(Path(__file__).resolve().parent))

from app_utils.cli_config import load_inference_config
from app_utils.tool import mask_stitcher
from app_utils.tool.tfw2lonlat import WorldFile, convert_yolo_json_to_geojson
from app_utils.tool.tif_tiler import (
    imwrite_unicode,
    is_blank,
    iter_tiles,
    read_big_image,
    tile_file_name,
)

TIF_EXTS = (".tif", ".tiff")
TFW_EXTS = (".tfw", ".jgw")


def build_file_index(root: Path, exts) -> Dict[str, Path]:
    """遞迴建立 {stem(小寫): 路徑} 索引，先找到先用。"""
    index: Dict[str, Path] = {}
    if not root.is_dir():
        print(f"[Warning] 找不到資料夾: {root}")
        return index
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            index.setdefault(p.stem.lower(), p)
    return index


def sanitize_model_name(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(model_id).stem)


def _draw_bigmap(
    img: np.ndarray,
    geojson: dict,
    wf: WorldFile,
    src_epsg: int,
    out_path: Path,
    scale: float,
    quality: int = 85,
) -> None:
    """把 GeoJSON 疊繪在縮小後的大圖上供目視驗證（重用 draw_geojson.draw_on_image）。"""
    from app_utils.tool.draw_geojson import WorldFile as DrawWorldFile, draw_on_image

    scale = max(0.01, min(1.0, float(scale)))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img.copy()
    # 縮小圖的像素座標 = 原圖像素 * scale，等價於把 world file 的像素尺寸放大 1/scale
    wf_draw = DrawWorldFile(wf.A / scale, wf.D / scale, wf.B / scale, wf.E / scale, wf.C, wf.F)
    reverse_transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{src_epsg}", always_xy=True)
    draw_on_image(small, geojson, wf_draw, reverse_transformer, 0.0, 0.0, draw_labels=False)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(out_path))
        print(f"[Info] Bigmap 驗證圖: {out_path}")


def process_stem(
    stem: str,
    tif_path: Path,
    tfw_path: Path,
    loaded_models: Dict[str, object],
    cfg: dict,
    args,
    transformer: Transformer,
    out_dir: Path,
) -> None:
    t_start = time.perf_counter()
    wf = WorldFile.from_file(tfw_path)
    img = read_big_image(tif_path)
    h, w = img.shape[:2]
    print(f"[Info] {stem}: {w}x{h}，tile={args.tile_size} overlap={args.overlap}")

    tiles_dir = out_dir / "tiles"
    if args.keep_tiles:
        tiles_dir.mkdir(parents=True, exist_ok=True)

    device_val = None if args.device == "auto" else args.device

    for mid, model in loaded_models.items():
        instances: List[dict] = []
        n_tiles = n_blank = 0
        for seq, x0, y0, view in iter_tiles(img, args.tile_size, args.overlap):
            n_tiles += 1
            if is_blank(view):
                n_blank += 1
                continue
            if args.keep_tiles:
                name = tile_file_name(stem, seq, x0, y0, view.shape[1], view.shape[0])
                imwrite_unicode(tiles_dir / name, view)

            predict_kwargs = {
                "source": np.ascontiguousarray(view),
                "imgsz": args.imgsz,
                "conf": args.conf,
                "verbose": False,
            }
            if device_val:
                predict_kwargs["device"] = device_val
            # 移到 CPU 後立即釋放 GPU 端結果，避免密集場景 OOM
            results = [r.cpu() for r in model.predict(**predict_kwargs)]
            instances.extend(
                mask_stitcher.extract_instances_from_result(
                    results[0], x0, y0, tile_seq=seq,
                    allowed_class_ids=cfg["allowed_class_ids"],
                )
            )
            del results
        print(f"[Info] {stem} x {mid}: {n_tiles} 片（空白跳過 {n_blank}），raw instances={len(instances)}")

        merged = mask_stitcher.merge_instances(instances, gap=args.stitch_gap)
        print(f"[Info] {stem} x {mid}: 縫合後 instances={len(merged)}")

        names = getattr(model, "names", {}) or {}
        objects = mask_stitcher.instances_to_objects(
            merged,
            names=names,
            mask_steps=cfg["mask_steps_parsed"],
            polygon_steps=cfg["polygon_steps_parsed"],
        )

        payload = {
            "image": {"file_name": tif_path.name, "width": w, "height": h},
            "model": {"name": mid},
            "objects": objects,
        }
        geojson = convert_yolo_json_to_geojson(payload, wf, transformer, crop_x=0.0, crop_y=0.0)
        # objects 與 features 依序一一對應，回填切片追溯資訊
        for feat, obj in zip(geojson.get("features", []), objects):
            feat["properties"]["tile_seqs"] = obj.get("tile_seqs", [])
        geojson["metadata"]["tiling"] = {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stitch_gap": args.stitch_gap,
            "src_epsg": args.src_epsg,
        }

        out_path = out_dir / f"{stem}__{sanitize_model_name(mid)}.geojson"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        print(f"[Info] GeoJSON 輸出: {out_path}（{len(geojson['features'])} features）")

        if args.bigmap:
            bigmap_dir = out_dir / "bigmap"
            bigmap_dir.mkdir(parents=True, exist_ok=True)
            _draw_bigmap(
                img, geojson, wf, args.src_epsg,
                bigmap_dir / f"{stem}__{sanitize_model_name(mid)}.jpg",
                args.bigmap_scale,
            )

    print(f"[Info] {stem} 完成，耗時 {time.perf_counter() - t_start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="TIF Folder -> GeoJSON 串流式 Pipeline（自動切片 + mask 縫合）")
    parser.add_argument("--tif-dir", required=True, help="大圖 TIF 資料夾（遞迴搜尋 .tif/.tiff）")
    parser.add_argument("--tfw-dir", default=None, help="World File 資料夾（.tfw/.jgw；預設同 --tif-dir）")
    parser.add_argument("-o", "--output", default="./tif_pipeline_output", help="輸出資料夾")
    parser.add_argument("-m", "--models", default="yolov12m.pt", help="逗號分隔的模型路徑")
    parser.add_argument("-c", "--config", default=None, help="app.py 匯出的 config JSON（mask/polygon 優化）")
    parser.add_argument("--tile-size", type=int, default=1600, help="切片邊長（預設 1600）")
    parser.add_argument("--overlap", type=int, default=320, help="相鄰切片重疊像素（預設 320）")
    parser.add_argument("--imgsz", type=int, default=1600, help="模型推論尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="信心閾值")
    parser.add_argument("--device", default="auto", help="推論裝置（auto/cpu/cuda:0...）")
    parser.add_argument("--src-epsg", type=int, default=3826, help="TIF 原始座標系 EPSG（預設 TWD97/3826）")
    parser.add_argument("--stitch-gap", type=int, default=1, help="縫合像素容差（預設 1px）")
    parser.add_argument("--stems", default=None, help="只處理指定圖幅（逗號分隔 stem）")
    parser.add_argument("--keep-tiles", action="store_true", help="除錯：切片 PNG 落地（既有命名約定）")
    parser.add_argument("--bigmap", action="store_true", help="輸出大圖疊繪驗證 JPG")
    parser.add_argument("--bigmap-scale", type=float, default=0.5, help="bigmap 縮放比例（預設 0.5）")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_dir = Path(args.tif_dir)
    tfw_dir = Path(args.tfw_dir) if args.tfw_dir else tif_dir
    tif_index = build_file_index(tif_dir, TIF_EXTS)
    tfw_index = build_file_index(tfw_dir, TFW_EXTS)
    if not tif_index:
        print(f"[Error] {tif_dir} 中找不到任何 TIF")
        sys.exit(1)

    if args.stems:
        wanted = {s.strip().lower() for s in args.stems.split(",") if s.strip()}
        tif_index = {k: v for k, v in tif_index.items() if k in wanted}
        missing = wanted - set(tif_index)
        if missing:
            print(f"[Warning] 指定的圖幅找不到 TIF: {sorted(missing)}")

    cfg = load_inference_config(args.config)

    from app_utils.model_cache import get_model

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    loaded_models = {mid: get_model(mid) for mid in models}

    transformer = Transformer.from_crs(f"EPSG:{args.src_epsg}", "EPSG:4326", always_xy=True)

    failed: List[str] = []
    for stem_key in sorted(tif_index):
        tif_path = tif_index[stem_key]
        tfw_path = tfw_index.get(stem_key)
        if tfw_path is None:
            print(f"[Error] 圖幅 {tif_path.stem} 缺少 world file，略過")
            failed.append(tif_path.stem)
            continue
        try:
            process_stem(tif_path.stem, tif_path, tfw_path, loaded_models, cfg, args, transformer, out_dir)
        except Exception as exc:  # 單一圖幅失敗不中斷整批
            print(f"[Error] 圖幅 {tif_path.stem} 處理失敗: {exc}")
            failed.append(tif_path.stem)

    total = len(tif_index)
    print(f"[Done] {total - len(failed)}/{total} 圖幅完成" + (f"，失敗: {failed}" if failed else ""))
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
