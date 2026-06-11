"""端到端批次管線：推論 CLI → GeoJSON 轉換 CLI。

以 subprocess 依序執行兩個既有 CLI，彼此不 import、保持低耦合：

  1) python app.py -i <影像資料夾> -o <輸出>/json -m <模型> -c <config.json>
  2) python app_utils/tool/batch_tfw2lonlat.py --json-dir <輸出>/json
         --output-dir <輸出>/geojson --tfw-dir <tfw資料夾> --src-epsg <EPSG>

使用範例：
  # 推論 + GeoJSON 轉換
  python run_pipeline.py -i D:/imgs -o D:/result -m best.pt -c config.json --tfw-dir D:/tfws

  # 只推論（未提供 --tfw-dir 時自動略過 GeoJSON 轉換）
  python run_pipeline.py -i D:/imgs -o D:/result -m best.pt -c config.json
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(name: str, cmd: list) -> None:
    print(f"\n=== [{name}] ===")
    print(" ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode != 0:
        print(f"[Error] {name} 失敗（exit code {proc.returncode}），管線中止。")
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="批次管線：YOLO 推論 -> GeoJSON 轉換（兩個獨立 CLI 依序執行）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 推論參數（轉交 app.py CLI） ---
    parser.add_argument("-i", "--input", required=True, help="輸入影像資料夾（或單一影像）")
    parser.add_argument("-o", "--output", default="./pipeline_output",
                        help="輸出根目錄（json/ 放推論結果、geojson/ 放轉換結果）")
    parser.add_argument("-m", "--models", default="yolov12m.pt", help="模型，逗號分隔可多個")
    parser.add_argument("-c", "--config", default=None, help="GUI 匯出的 config JSON")
    parser.add_argument("--imgsz", type=int, default=None, help="影像尺寸（未指定用 app.py 預設）")
    parser.add_argument("--conf", type=float, default=None, help="信心閾值（未指定用 app.py 預設）")
    parser.add_argument("--device", default=None, help="auto / cpu / cuda:0 ...")
    parser.add_argument("--no-save-img", action="store_true", help="不輸出標註圖，只輸出 JSON")
    # --- GeoJSON 轉換參數（轉交 batch_tfw2lonlat.py） ---
    parser.add_argument("--tfw-dir", default=None,
                        help="World file（.tfw/.jgw）資料夾；未提供則略過 GeoJSON 轉換")
    parser.add_argument("--src-epsg", type=int, default=3826, help="原始影像座標系 EPSG")
    parser.add_argument("--skip-geojson", action="store_true", help="強制略過 GeoJSON 轉換")

    args = parser.parse_args()

    out_root = Path(args.output)
    json_dir = out_root / "json"
    geo_dir = out_root / "geojson"

    # === 步驟 1：推論 ===
    infer_cmd = [sys.executable, PROJECT_ROOT / "app.py",
                 "-i", args.input, "-o", json_dir, "-m", args.models]
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
    run_step("步驟 1/2：YOLO 推論", infer_cmd)

    # === 步驟 2：GeoJSON 轉換（可選） ===
    if args.skip_geojson or not args.tfw_dir:
        print("\n[Info] 未提供 --tfw-dir（或指定 --skip-geojson），略過 GeoJSON 轉換。")
        print(f"[Info] 推論結果：{json_dir}")
        return

    geo_cmd = [sys.executable, PROJECT_ROOT / "app_utils" / "tool" / "batch_tfw2lonlat.py",
               "--json-dir", json_dir,
               "--output-dir", geo_dir,
               "--tfw-dir", args.tfw_dir,
               "--src-epsg", args.src_epsg]
    run_step("步驟 2/2：GeoJSON 轉換", geo_cmd)

    print("\n[Success] 管線完成。")
    print(f"[Info] 推論結果：{json_dir}")
    print(f"[Info] GeoJSON ：{geo_dir}")


if __name__ == "__main__":
    main()
