"""大圖 TIF 讀取與記憶體內切片。

供 run_tif_pipeline.py 串流式使用：切片以 numpy view 產生（零複製），
不落地磁碟；亦提供獨立 CLI 可將切片以既有命名約定
（{stem}_{seq}_{x}_{y}_{WxH}.png）輸出供除錯。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


def imread_unicode(path) -> Optional[np.ndarray]:
    """cv2.imread 的 unicode 路徑安全版（Windows 中文路徑）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image: np.ndarray) -> bool:
    suffix = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(suffix, image)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def _to_bgr_uint8(arr: np.ndarray) -> np.ndarray:
    """任意波段/位深的影像陣列 -> BGR uint8（模型輸入格式，同 cv2.imread）。"""
    arr = np.squeeze(np.asarray(arr))
    if arr.dtype == np.uint16:
        arr = (arr / 257.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        vmax = float(arr_f.max()) or 1.0
        arr = (arr_f * (255.0 / vmax)).astype(np.uint8) if vmax > 255 else arr_f.astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    # tifffile 讀出為 RGB 順序
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


def read_big_image(path) -> np.ndarray:
    """讀取大圖為 BGR uint8。cv2 無法解碼（BigTIFF / 特殊壓縮）時改用 tifffile。"""
    img = imread_unicode(path)
    if img is not None:
        return img
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            f"cv2 無法解碼 {path}，且未安裝 tifffile 可供備援（pip install tifffile）"
        ) from exc
    print(f"[Info] cv2 解碼失敗，改用 tifffile 讀取: {path}")
    return _to_bgr_uint8(tifffile.imread(str(path)))


def compute_axis_origins(size: int, tile: int, overlap: int) -> List[int]:
    """一維切片原點。stride = tile - overlap；最後一片 clamp 到 size - tile。

    size <= tile 時回傳 [0]（整段一片，不 padding）。
    """
    if tile <= 0:
        raise ValueError("tile 必須為正")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap 需滿足 0 <= overlap < tile")
    if size <= tile:
        return [0]
    stride = tile - overlap
    origins = list(range(0, size - tile, stride))
    last = size - tile
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


def compute_tile_origins(width: int, height: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    """回傳 (x0, y0) 清單，列優先（由上而下、由左而右），完整覆蓋整張圖。"""
    xs = compute_axis_origins(width, tile, overlap)
    ys = compute_axis_origins(height, tile, overlap)
    return [(x0, y0) for y0 in ys for x0 in xs]


def iter_tiles(
    img: np.ndarray, tile: int, overlap: int
) -> Iterator[Tuple[int, int, int, np.ndarray]]:
    """逐片產生 (seq, x0, y0, view)。view 是 img 的 numpy view，不複製。"""
    h, w = img.shape[:2]
    for seq, (x0, y0) in enumerate(compute_tile_origins(w, h, tile, overlap), start=1):
        yield seq, x0, y0, img[y0 : y0 + min(tile, h), x0 : x0 + min(tile, w)]


def is_blank(tile: np.ndarray, nodata_fraction: float = 0.98, std_threshold: float = 1.0) -> bool:
    """判斷切片是否為空白（圖幅外的 nodata 領域），可跳過推論。

    純黑/純白像素占比 >= nodata_fraction，或整片近乎均勻（std < std_threshold）視為空白。
    """
    if tile.size == 0:
        return True
    if float(tile.std()) < std_threshold:
        return True
    if tile.ndim == 3:
        black = (tile == 0).all(axis=2)
        white = (tile == 255).all(axis=2)
    else:
        black = tile == 0
        white = tile == 255
    return float((black | white).mean()) >= nodata_fraction


def tile_file_name(stem: str, seq: int, x0: int, y0: int, w: int, h: int, ext: str = ".png") -> str:
    """既有 pipeline 的切片命名約定：{stem}_{seq:04d}_{x}_{y}_{WxH}.png"""
    return f"{stem}_{seq:04d}_{x0}_{y0}_{w}x{h}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="將大圖 TIF 切片為固定尺寸 PNG（除錯用；pipeline 本身不落地切片）")
    parser.add_argument("--tif", required=True, help="輸入 TIF 路徑")
    parser.add_argument("--output", required=True, help="輸出資料夾")
    parser.add_argument("--tile-size", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=320)
    parser.add_argument("--keep-blank", action="store_true", help="連空白切片也輸出")
    args = parser.parse_args()

    tif_path = Path(args.tif)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = read_big_image(tif_path)
    written = skipped = 0
    for seq, x0, y0, view in iter_tiles(img, args.tile_size, args.overlap):
        if not args.keep_blank and is_blank(view):
            skipped += 1
            continue
        name = tile_file_name(tif_path.stem, seq, x0, y0, view.shape[1], view.shape[0])
        imwrite_unicode(out_dir / name, view)
        written += 1
    print(f"[Done] 輸出 {written} 片，跳過空白 {skipped} 片 -> {out_dir}")


if __name__ == "__main__":
    main()
