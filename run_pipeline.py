"""端到端批次管線：YOLO 推論 → TFW 對應 → 旋轉偵測 → GeoJSON 轉換 → 大圖驗證。

以 subprocess 依序執行各個獨立 CLI，彼此不 import、保持低耦合：

  1) app.py                                  影像資料夾批次推論 -> YOLO JSON + 標註圖
  2) app_utils/tool/stage_tfw_files.py       依 JSON 檔名對應定位檔到暫存資料夾
  3) app_utils/tool/detect_slice_rotation.py 旋轉切片方向偵測 -> rotation map
  4) app_utils/tool/batch_tfw2lonlat.py      YOLO JSON -> GeoJSON（含反旋轉）
  5) app_utils/tool/draw_geojson_on_bigmap.py GeoJSON 繪回大圖（--bigmap 時執行）

使用範例：
  # 完整端到端（自動 tfw 對應 + 旋轉偵測 + GeoJSON + 大圖驗證）
  python run_pipeline.py -i D:/imgs -o D:/result -m best.pt -c config.json ^
      --tif-dirs "G:/.../110年度(北區)" "G:/.../109年度(東區)" --bigmap

  # 只推論
  python run_pipeline.py -i D:/imgs -o D:/result -m best.pt -c config.json

  # 推論 + GeoJSON（定位檔已預先對應好檔名時，可改用 --tfw-dir）
  python run_pipeline.py -i D:/imgs -o D:/result -m best.pt --tfw-dir D:/tfw_staged
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TOOL = PROJECT_ROOT / "app_utils" / "tool"


def run_step(name: str, cmd: list) -> None:
    print(f"\n=== [{name}] ===")
    print(" ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode != 0:
        print(f"[Error] {name} 失敗（exit code {proc.returncode}），管線中止。")
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="批次管線：YOLO 推論 -> GeoJSON -> 大圖驗證（各步驟為獨立 CLI 依序執行）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 推論參數（轉交 app.py CLI） ---
    parser.add_argument("-i", "--input", required=True, help="輸入影像資料夾（或單一影像）")
    parser.add_argument("-o", "--output", default="./pipeline_output",
                        help="輸出根目錄（json/ geojson/ bigmap/ 等子目錄）")
    parser.add_argument("-m", "--models", default="yolov12m.pt", help="模型，逗號分隔可多個")
    parser.add_argument("-c", "--config", default=None, help="GUI 匯出的 config JSON")
    parser.add_argument("--imgsz", type=int, default=None, help="影像尺寸（未指定用 app.py 預設）")
    parser.add_argument("--conf", type=float, default=None, help="信心閾值（未指定用 app.py 預設）")
    parser.add_argument("--device", default=None, help="auto / cpu / cuda:0 ...")
    parser.add_argument("--no-save-img", action="store_true", help="不輸出標註圖，只輸出 JSON")
    # --- GeoJSON / 大圖參數 ---
    parser.add_argument("--tif-dirs", nargs="*", type=Path, default=None,
                        help="大圖 TIF / 定位檔來源資料夾（遞迴搜尋；正確年度放最前）。"
                             "提供時自動執行 tfw 對應與旋轉偵測")
    parser.add_argument("--tfw-dir", default=None,
                        help="已預先對應好檔名的定位檔資料夾（不需自動對應時使用）")
    parser.add_argument("--src-epsg", type=int, default=3826, help="原始影像座標系 EPSG")
    parser.add_argument("--skip-geojson", action="store_true", help="只推論，略過 GeoJSON 轉換")
    parser.add_argument("--bigmap", action="store_true",
                        help="GeoJSON 轉換後繪回大圖驗證（需 --tif-dirs）")
    parser.add_argument("--bigmap-scale", type=float, default=0.5, help="大圖輸出縮放倍率")

    args = parser.parse_args()

    out_root = Path(args.output)
    json_dir = out_root / "json"
    geo_dir = out_root / "geojson"
    py = sys.executable

    # === 步驟 1：推論 ===
    infer_cmd = [py, PROJECT_ROOT / "app.py", "-i", args.input, "-o", json_dir, "-m", args.models]
    if args.config:
        infer_cmd += ["-c", args.config]
    if args.imgsz is not None:
        infer_cmd += ["--imgsz", args.imgsz]
    if args.conf is not None:
        infer_cmd += ["--conf", args.conf]
    if args.device:
        infer_cmd += ["--device", args.device]
    if args.no_save_img:
        infer_cmd += ["--no-save-img"]
    run_step("步驟 1：YOLO 推論", infer_cmd)

    if args.skip_geojson or (not args.tif_dirs and not args.tfw_dir):
        if not args.skip_geojson:
            print("\n[Info] 未提供 --tif-dirs / --tfw-dir，略過 GeoJSON 轉換。")
        print(f"[Info] 推論結果：{json_dir}")
        return

    # === 步驟 2、3：TFW 對應 + 旋轉偵測（提供 --tif-dirs 時） ===
    rotation_map_path = None
    if args.tif_dirs:
        tfw_staged = out_root / "_tfw"
        run_step("步驟 2：TFW 定位檔對應",
                 [py, TOOL / "stage_tfw_files.py",
                  "--json-dir", json_dir, "--output", tfw_staged,
                  "--tfw-dirs", *args.tif_dirs])
        tfw_dir = tfw_staged

        png_dir = Path(args.input)
        if png_dir.is_file():
            png_dir = png_dir.parent
        rotation_map_path = out_root / "rotation_map.json"
        run_step("步驟 3：旋轉切片方向偵測",
                 [py, TOOL / "detect_slice_rotation.py",
                  "--json-dir", json_dir, "--png-dir", png_dir,
                  "--output", rotation_map_path,
                  "--tif-dirs", *args.tif_dirs])
    else:
        tfw_dir = Path(args.tfw_dir)

    # === 步驟 4：GeoJSON 轉換 ===
    geo_cmd = [py, TOOL / "batch_tfw2lonlat.py",
               "--json-dir", json_dir, "--output-dir", geo_dir,
               "--tfw-dir", tfw_dir, "--src-epsg", args.src_epsg]
    if rotation_map_path is not None:
        geo_cmd += ["--rotation-map", rotation_map_path]
    run_step("步驟 4：GeoJSON 轉換", geo_cmd)

    # === 步驟 5：大圖驗證（可選） ===
    if args.bigmap:
        if not args.tif_dirs:
            print("[Warning] --bigmap 需要 --tif-dirs（大圖來源），略過大圖驗證。")
        else:
            run_step("步驟 5：GeoJSON 繪回大圖驗證",
                     [py, TOOL / "draw_geojson_on_bigmap.py",
                      "--geojson-dir", geo_dir,
                      "--output-dir", out_root / "bigmap",
                      "--scale", args.bigmap_scale,
                      "--tif-dirs", *args.tif_dirs])

    print("\n[Success] 管線完成。")
    print(f"[Info] 推論結果：{json_dir}")
    print(f"[Info] GeoJSON ：{geo_dir}")
    if args.bigmap and args.tif_dirs:
        print(f"[Info] 大圖驗證：{out_root / 'bigmap'}")


if __name__ == "__main__":
    main()
