"""TFW 定位檔對應與暫存工具。

batch_tfw2lonlat.py 依 JSON 檔名解析出 base（如 3074571_o_0001），
並尋找同名定位檔（3074571_o_0001.tfw）——但定位檔實際是以圖幅命名
（3074571.tfw 或外擴版 3074571_o.tfw）。本工具負責對應與複製：

  1. base 去掉切片序號 -> stem（3074571_o_0001 -> 3074571_o）
  2. 依序在 --tfw-dirs（遞迴）尋找 {stem}.tfw/.jgw，找不到再退而求 {圖幅編號}.tfw/.jgw
  3. 複製並改名為 {base}.tfw/.jgw 到 --output 暫存資料夾

不修改任何原始資料，也不修改轉換工具本身。

用法範例：
  python stage_tfw_files.py ^
      --json-dir "D:/.../20260611/109-1" ^
      --tfw-dirs "G:/.../109年度(東區)" "G:/.../110年度(北區)" ^
      --output   "D:/.../20260611/_tfw/109-1"
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 與 batch_tfw2lonlat.py 相同的檔名解析規則
JSON_PATTERN = re.compile(r"^(.*)_(\d+)_(\d+)(_.*)?\.json$")


def build_tfw_index(dirs: List[Path]) -> Dict[str, Path]:
    """遞迴建立 {stem(小寫): 路徑} 索引；dirs 依序處理，先找到先用。"""
    index: Dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            print(f"[Warning] 找不到資料夾，略過: {d}")
            continue
        for ext in ("*.tfw", "*.jgw"):
            for p in sorted(d.rglob(ext)):
                index.setdefault(p.stem.lower(), p)
    return index


def find_source(base: str, index: Dict[str, Path]) -> Optional[Path]:
    stem = re.sub(r"_\d+$", "", base)   # 去切片序號：3074571_o_0001 -> 3074571_o
    tile = base.split("_")[0]           # 圖幅編號：3074571
    candidates = [stem] if stem == tile else [stem, tile]  # 精確 stem 優先
    for cand in candidates:
        hit = index.get(cand.lower())
        if hit is not None:
            return hit
    return None


def stage(json_dir: Path, tfw_dirs: List[Path], output_dir: Path) -> Tuple[int, List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = build_tfw_index(tfw_dirs)
    print(f"[Info] 定位檔索引: {len(index)} 個")

    bases = set()
    for p in sorted(json_dir.glob("*.json")):
        m = JSON_PATTERN.match(p.name)
        if m:
            bases.add(m.group(1))

    matched, missing = 0, []
    for base in sorted(bases):
        src = find_source(base, index)
        if src is None:
            missing.append(base)
            print(f"[MISS] {base}: 找不到對應定位檔")
            continue
        shutil.copyfile(src, output_dir / f"{base}{src.suffix}")
        matched += 1

    print("-" * 40)
    print(f"[{'Success' if not missing else 'Warning'}] 對應 {matched}/{len(bases)} 個 base"
          + (f"，缺漏 {len(missing)} 個" if missing else ""))
    print(f"[Info] 暫存資料夾: {output_dir}")
    return matched, missing


def main():
    parser = argparse.ArgumentParser(
        description="依 YOLO JSON 檔名對應定位檔，複製改名到暫存資料夾供 batch_tfw2lonlat 使用",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json-dir", required=True, type=Path, help="YOLO 推論 JSON 資料夾")
    parser.add_argument("--tfw-dirs", nargs="+", required=True, type=Path,
                        help="定位檔來源資料夾（遞迴；依序先找到先用，正確年度放最前）")
    parser.add_argument("--output", required=True, type=Path, help="暫存輸出資料夾")
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        print(f"[Error] JSON 資料夾不存在: {args.json_dir}")
        sys.exit(1)

    _, missing = stage(args.json_dir, args.tfw_dirs, args.output)
    sys.exit(2 if missing else 0)


if __name__ == "__main__":
    main()
