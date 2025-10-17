import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

from ultralytics import YOLO


# ---------- 工具函式 ----------

def _literal(s: str):
    """聰明的型別轉換：'True'->True, '0.5'->0.5, 'None'->None，其餘保留字串"""
    try:
        return ast.literal_eval(s)
    except Exception:
        ls = s.lower()
        if ls == "true":
            return True
        if ls == "false":
            return False
        if ls == "none":
            return None
        return s


def parse_unknown_kv(unknown_tokens) -> Dict[str, Any]:
    """
    支援：
      --epochs=100 / epochs=100
      --cos_lr      (無值視為 True)
      --optimizer=SGD
    建議把額外參數放在 `--` 之後，避免被 argparse 吃掉，例如：
      python main.py ... -- --optimizer=SGD cos_lr=True lr0=0.01
    """
    out = {}
    for tok in unknown_tokens:
        tok = tok.lstrip("-")  # 去掉前導 -
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.replace("-", "_")] = _literal(v)
        else:
            out[tok.replace("-", "_")] = True  # 無值旗標 -> True
    return out


def load_config_overrides(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if p.exists():
        if p.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("pyyaml 未安裝，無法讀取 YAML。請先 `pip install pyyaml`")
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if p.suffix.lower() == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        raise ValueError("不支援的 config 格式，請用 .yaml/.yml/.json 檔或 JSON 字串")
    # 也允許直接給 JSON 字串
    try:
        return json.loads(path)
    except Exception:
        raise FileNotFoundError(f"Config 檔不存在，且也不是有效的 JSON 字串：{path}")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model(model_spec: str, pretrained: Optional[str] = None) -> YOLO:
    """
    統一建模入口：
    - 若給 .yaml/.yml：會先建模，再視情況 .load(pretrained)
    - 若給 .pt：直接載入權重
    - 若給任意能被 Ultralytics 辨識的字串也可（例如官方模型別名）
    """
    m = YOLO(model_spec)
    if model_spec.lower().endswith((".yml", ".yaml")) and pretrained:
        # 以 YAML 建模後載入預訓練權重（常見於分割用 detect 權重轉移學習）
        m = m.load(pretrained)
    return m


# ---------- pipeline 函式 ----------

def train(project: str, name: str, model_spec: str, pretrained: Optional[str], base_overrides: Dict[str, Any]):
    """
    訓練：
    - 預設將輸出到 {project}/train_{name}
    - base_overrides 可帶入 Ultralytics 的 train 參數（data/epochs/batch/imgsz 等）
    """
    model = build_model(model_spec, pretrained)

    # 安全預設，可被外部覆蓋
    defaults = dict(
        exist_ok=True,
        plots=True,
        project=project,
        name="train_" + name,
    )

    # 去掉 None，避免傳入 YOLO 造成副作用
    clean = {k: v for k, v in base_overrides.items() if v is not None}
    merged = {**defaults, **clean}
    return model.train(**merged)


def _best_pt_path(project: str, name: str) -> Path:
    return (Path(project) / f"train_{name}" / "weights" / "best.pt").resolve()


def val(project: str, name: str, **val_overrides):
    model_path = _best_pt_path(project, name)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 best.pt：{model_path}")
    model = YOLO(str(model_path))
    # 不帶 data 時，會沿用訓練時的 data
    metrics = model.val(project=project, name="val_" + name, **val_overrides)
    return metrics


def test(project: str, name: str, test_dir: str, **pred_overrides):
    model_path = _best_pt_path(project, name)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 best.pt：{model_path}")
    model = YOLO(str(model_path))

    test_dir_p = Path(test_dir).resolve()
    if not test_dir_p.exists():
        raise FileNotFoundError(f"測試資料夾不存在：{test_dir_p}")

    # 快速跑一次（如需 warmup 等）
    _ = model(str(test_dir_p))
    # 正式推論，支援覆蓋
    return model.predict(str(test_dir_p), project=project, name="test_" + name, **pred_overrides)


def export(project: str, name: str, export_format: str, **exp_overrides):
    model_path = _best_pt_path(project, name)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 best.pt：{model_path}")
    model = YOLO(str(model_path))
    # 統一放在 runs/export_* 只是名稱標示，Ultralytics 會輸出到當前目錄
    return model.export(format=export_format, **exp_overrides)


# ---------- CLI 入口 ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultralytics YOLO 12 路口腳本（修正版）")

    # 主要超參
    parser.add_argument("--model", type=str, default="yolo12x-seg.yaml",
                        help="模型規格：可為 .yaml/.pt 或官方別名，預設 yolo12x-seg.yaml")
    parser.add_argument("--pretrained", type=str, default="yolo12x.pt",
                        help="若 --model 為 .yaml，則可透過此指定預訓練權重（例如 yolo12x.pt）。給空字串可略過載入")
    parser.add_argument("--batch", type=int, default=160)
    parser.add_argument("--epoch", type=int, default=600)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--data", type=str, help="資料集 YAML 或別名（如 coco8.yaml / coco8-seg.yaml）")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--config", type=str, help="YAML/JSON 檔路徑，或 JSON 字串")

    # pipeline 控制
    parser.add_argument("--test", action="store_true", help="訓練與驗證後進行推論")
    parser.add_argument("--test-dir", type=str, help="推論資料夾")
    parser.add_argument("--export", action="store_true", help="訓練與驗證後進行匯出")
    parser.add_argument("--export-format", type=str, help="匯出格式（如 onnx、engine、torchscript 等）")

    # 先吃掉能解析的，其餘未知參數保留
    args, unknown = parser.parse_known_args()

    # 1) 基本參數
    base = dict(
        data=args.data,
        batch=args.batch,
        epochs=args.epoch,
        imgsz=args.imgsz,
        # 你原本硬編的 device=1 建議移到外部覆蓋，例如：-- device=0
    )

    # 2) config 檔（若提供）
    file_overrides = load_config_overrides(args.config) if args.config else {}

    # 3) 未知 kv 直通（建議放在 -- 之後，例如 `-- cos_lr=True optimizer='SGD'`）
    passthrough = parse_unknown_kv(unknown)

    # 最終優先序：基本 < config < 未知 kv（符合註解）
    train_overrides = {**base, **file_overrides, **passthrough}

    # 執行期健全性檢查
    if not train_overrides.get("data"):
        parser.error("--data 為必填（可由 CLI 或 config/未知 kv 提供，例如 coco8.yaml 或 coco8-seg.yaml 或你的 dataset.yaml）")

    if args.test and not args.test_dir:
        parser.error("--test 需要搭配 --test-dir")

    if args.export and not args.export_format:
        parser.error("--export 需要搭配 --export-format（如 onnx、engine、torchscript）")

    # 空字串視為未指定
    pretrained = args.pretrained if (args.pretrained is not None and args.pretrained.strip() != "") else None

    # 走 pipeline
    _ = train(args.project, args.name, args.model, pretrained, train_overrides)
    _ = val(args.project, args.name)

    if args.test:
        # 允許把推論參數放在未知 kv，一併使用（只挑 predict 會用到的鍵）
        test_allowed = {
            "imgsz", "conf", "iou", "max_det", "classes", "agnostic_nms",
            "device", "half", "save", "save_txt", "save_conf", "save_crop",
        }
        test_kwargs = {k: v for k, v in passthrough.items() if k in test_allowed}
        _ = test(args.project, args.name, args.test_dir, **test_kwargs)

    if args.export:
        # 匯出參數：避免把訓練用的鍵也帶進來
        export_kwargs = {k: v for k, v in passthrough.items() if k not in train_overrides}
        _ = export(args.project, args.name, args.export_format, **export_kwargs)
