import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import suppress
from torch.utils.tensorboard import SummaryWriter
import torch

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
    解析 argparse 無法處理的額外參數，支援：
      --epochs=100 / epochs=100
      --cos_lr（無值視為 True）
      --optimizer=SGD
      --device 1（成對寫法）
    建議把額外參數放在 `--` 之後，避免被 argparse 吃掉，例如：
      python train.py ... -- --optimizer=SGD cos_lr=True lr0=0.01 device=1
    """
    out: Dict[str, Any] = {}
    i = 0
    while i < len(unknown_tokens):
        tok = unknown_tokens[i]
        if not tok:
            i += 1
            continue
        s = tok.lstrip("-")  # 去掉前導 -
        if not s:
            i += 1
            continue

        if "=" in s:  # e.g. --device=1 或 device=1
            k, v = s.split("=", 1)
            out[k.replace("-", "_")] = _literal(v)
            i += 1
            continue

        # 嘗試成對寫法：--device 1 / device 1
        nxt = unknown_tokens[i + 1] if (i + 1) < len(unknown_tokens) else None
        if nxt is not None and (not nxt.startswith("-")) and ("=" not in nxt):
            out[s.replace("-", "_")] = _literal(nxt)
            i += 2
        else:
            # 無值旗標 -> True
            out[s.replace("-", "_")] = True
            i += 1

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


# ---- DDP/裝置輔助 ----

def _is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_main_process() -> bool:
    """在單卡或未初始化 DDP 時也回傳 True。"""
    if not _is_dist_avail_and_initialized():
        # 若未初始化，仍可透過環境變數判斷；默認視為主進程
        rank_env = os.getenv("RANK")
        return (rank_env is None) or (rank_env == "0")
    try:
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


def world_info_str() -> str:
    rank = os.getenv("RANK", "?")
    local_rank = os.getenv("LOCAL_RANK", "?")
    world_size = os.getenv("WORLD_SIZE", "?")
    ddp = _is_dist_avail_and_initialized()
    return f"[DDP] RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size} initialized={ddp}"


def _norm_device(d: Optional[Any]) -> Optional[str]:
    """把 device 正規化為 Ultralytics 友善格式：
    - int → "0"
    - list/tuple/set → "0,1,2"
    - 純數字/逗號字串（"0,1"）→ 原樣
    - 其他字串（"cpu"/"cuda:0"/"mps"）→ 原樣
    """
    if d is None:
        return None
    # 序列 → 逗號字串
    if isinstance(d, (list, tuple, set)):
        def _to_idx(x):
            xs = str(x)
            # 支援 "cuda:0" 這類
            return int(''.join(ch for ch in xs if ch.isdigit()))
        try:
            return ','.join(str(_to_idx(x)) for x in d)
        except Exception:
            return ','.join(str(x) for x in d)
    if isinstance(d, int):
        return str(d)
    ds = str(d).strip()
    # 處理像 "(0, 1)" 這種從 shell 進來的字串
    if ds.startswith('(') and ds.endswith(')'):
        ds = ds[1:-1].replace(' ', '')
    # 純由數字與逗號組成（如 "0"、"0,1"）
    if all(ch.isdigit() or ch == ',' for ch in ds):
        return ds
    return ds  # 已是 'cpu'、'cuda:0'、'mps' 等


def _coerce_device_to_visible(d: Optional[Any]) -> Optional[str]:
    """依據實際可見 GPU 數，調整多卡要求以避免 Ultralytics select_device 直接報錯。
    規則：
    - 若 torch 只看到 0 張 GPU → 回傳 'cpu'
    - 若只看到 1 張 GPU → 僅保留第一張（'0'）
    - 若看到 N 張 GPU → 過濾掉超出 [0, N-1] 的索引，並去重排序
    """
    ds = _norm_device(d)
    if ds is None:
        return None
    # CPU 或明確的非數字裝置不動
    if any(tok.isalpha() for tok in ds.replace(',', '')):
        return ds

    vis = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if vis <= 0:
        print("[WARN] CUDA 不可用，改用 CPU 訓練")
        return 'cpu'

    # 解析出索引清單
    try:
        req = [int(x) for x in ds.split(',') if x != '']
    except Exception:
        return ds

    if vis == 1:
        if len(req) > 1 or req[0] != 0:
            print(f"[WARN] 目前僅有 1 張可見 GPU（CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}），強制使用 device=0")
        return '0'

    # 多卡情境，過濾非法索引
    legal = sorted({i for i in req if 0 <= i < vis})
    if not legal:
        print(f"[WARN] 要求的 device={ds} 與可見 GPU 數 vis={vis} 不相容，改用 device=0")
        return '0'
    if legal != req:
        print(f"[WARN] 過濾非法索引：要求 {req} → 使用 {legal}")
    return ','.join(str(i) for i in legal)


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

def _maybe_write_tensorboard_scaffold(project: str, run_name: str, model: YOLO, imgsz: int):
    """僅在主進程寫入一點點 TensorBoard scaffold，避免多進程搶寫。"""
    if not is_main_process():
        return

    run_dir = ensure_dir(Path(project) / run_name)
    writer = None
    try:
        writer = SummaryWriter(log_dir=str(run_dir))
        # 影像：保證面板出現（之後可改成真實樣本/預測）
        _imgsz = int(imgsz)
        H = W = max(64, min(_imgsz, 512))
        writer.add_images("sanity/dummy_batch", torch.rand(4, 3, H, W), global_step=0)

        # Graph：逐步嘗試，能寫則寫；失敗就跳過
        try:
            # 嘗試抓取當前 model device（可能尚未移動，通常是 CPU）
            dev = next(model.model.parameters()).device
        except Exception:
            dev = torch.device("cpu")

        _graphsz = max(256, min(640, int(imgsz)))  # 用小一點即可
        dummy = torch.zeros(1, 3, _graphsz, _graphsz, device=dev)
        model.model.eval()

        ok = False
        with torch.inference_mode():
            with suppress(Exception):
                writer.add_graph(model.model, dummy)
                print("[TB] add_graph ok (direct)")
                ok = True
            if not ok:
                with suppress(TypeError, Exception):
                    writer.add_graph(model.model, dummy, use_strict_trace=False)
                    print("[TB] add_graph ok (use_strict_trace=False)")
                    ok = True
            if not ok:
                with suppress(Exception):
                    traced = torch.jit.trace(model.model, dummy, strict=False, check_trace=False)
                    writer.add_graph(traced, (dummy,))
                    print("[TB] add_graph ok (jit.trace fallback)")
                    ok = True
            if not ok:
                from torch.fx import symbolic_trace
                with suppress(Exception):
                    gm = symbolic_trace(model.model)
                    writer.add_graph(gm, (dummy,))
                    print("[TB] add_graph ok (torch.fx)")
                    ok = True
        if not ok:
            print("[TB] add_graph failed, skip (dynamic models are common)")
    except Exception as e:
        print(f"[TB] writer/init failed: {e}")
    finally:
        if writer is not None:
            # 這裡主動 flush/close，後續 Ultralytics 自己的 TB logger 仍會在同目錄追加
            writer.flush()
            writer.close()

def _debug_devices():
    """列印目前裝置/環境資訊，失敗時不中斷流程。"""
    print("=== Device Debug ===")
    try:
        print(world_info_str())
    except Exception as e:
        print(f"[DBG] world_info_str() failed: {e}")

    # 版本與後端
    try:
        import platform
        print(f"python={platform.python_version()} torch={torch.__version__}")
    except Exception as e:
        print(f"[DBG] version check failed: {e}")

    # CUDA 狀態
    try:
        cuda_ok = torch.cuda.is_available()
        print(f"cuda_available={cuda_ok} CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
        if cuda_ok:
            n = torch.cuda.device_count()
            print(f"visible_gpu_count={n}")
            for i in range(n):
                try:
                    p = torch.cuda.get_device_properties(i)
                    print(f"  [GPU {i}] {p.name} cc={p.major}.{p.minor} vram={p.total_memory/1024**3:.1f}GiB")
                except Exception as ie:
                    print(f"  [GPU {i}] <err: {ie}>")
            with suppress(Exception):
                print(f"current_device={torch.cuda.current_device()}")
        else:
            print("running_on=CPU")
    except Exception as e:
        print(f"[DBG] cuda probe failed: {e}")

    # Apple MPS（若有）
    try:
        mps_ok = getattr(torch.backends.mps, "is_available", lambda: False)()
        print(f"mps_available={mps_ok}")
    except Exception as e:
        print(f"[DBG] mps probe failed: {e}")

    # 可選套件：xformers / flash-attn
    for mod in ("xformers", "flash_attn"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "<unknown>")
            print(f"{mod}={ver}")
        except Exception:
            print(f"{mod}=not_installed")

def train(project: str, name: str, model_spec: str, pretrained: Optional[str], base_overrides: Dict[str, Any]):
    model = build_model(model_spec, pretrained)

    defaults = dict(
        exist_ok=True,
        plots=True,
        project=project,
        name="train_" + name,
    )
    clean = {k: v for k, v in base_overrides.items() if v is not None}
    if "device" in clean:
        clean["device"] = _norm_device(clean["device"])
    merged = {**defaults, **clean}

    orig_dev = merged.get('device')
    merged['device'] = _coerce_device_to_visible(orig_dev)
    print(f"[DEBUG] train() device: requested={orig_dev} -> using={merged.get('device')}")

    # 只在主進程寫入一次 TensorBoard scaffold，避免 DDP 多進程搶寫
    _maybe_write_tensorboard_scaffold(merged["project"], merged["name"], model, int(merged.get("imgsz", 640)))

    # === 開始訓練 ===
    return model.train(**merged)


def _best_pt_path(project: str, name: str) -> Path:
    return (Path(project) / f"train_{name}" / "weights" / "best.pt").resolve()


def val(project: str, name: str, **val_overrides):
    model_path = _best_pt_path(project, name)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 best.pt：{model_path}")
    model = YOLO(str(model_path))
    # 正規化 device 並傳入
    if "device" in val_overrides:
        val_overrides["device"] = _coerce_device_to_visible(val_overrides["device"])
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

    if "device" in pred_overrides:
        pred_overrides["device"] = _coerce_device_to_visible(pred_overrides["device"])

    # 快速跑一次（如需 warmup 等）
    _ = model(str(test_dir_p))
    # 正式推論，支援覆蓋
    return model.predict(str(test_dir_p), project=project, name="test_" + name, **pred_overrides)


def export(project: str, name: str, export_format: str, **exp_overrides):
    model_path = _best_pt_path(project, name)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 best.pt：{model_path}")
    model = YOLO(str(model_path))

    if "device" in exp_overrides:
        exp_overrides["device"] = _coerce_device_to_visible(exp_overrides["device"])

    # 統一放在 runs/export_* 只是名稱標示，Ultralytics 會輸出到當前目錄
    return model.export(format=export_format, **exp_overrides)


# ---------- CLI 入口 ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultralytics YOLO 12 路口腳本（DDP/TensorBoard 修正版）")

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

    _debug_devices()  # 顯示目前可見裝置與 torch 狀態（除錯用）

    # 1) 基本參數
    base = dict(
        data=args.data,
        batch=args.batch,
        epochs=args.epoch,
        imgsz=args.imgsz,
        # device 由 unknown kv 或 config 覆蓋（更直覺）
    )

    # 2) config 檔（若提供）
    file_overrides = load_config_overrides(args.config) if args.config else {}

    # 3) 未知 kv 直通（建議放在 -- 之後，例如 `-- cos_lr=True optimizer='SGD' device=0`）
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

    # 將 device 傳給 val（避免驗證階段跑到錯卡）
    val_device = train_overrides.get("device")
    _ = val(args.project, args.name, device=val_device)

    if args.test:
        # 允許把推論參數放在未知 kv，一併使用（只挑 predict 會用到的鍵）
        test_allowed = {
            "imgsz", "conf", "iou", "max_det", "classes", "agnostic_nms",
            "device", "half", "save", "save_txt", "save_conf", "save_crop",
        }
        test_kwargs = {k: (_norm_device(v) if k == "device" else v) for k, v in passthrough.items() if k in test_allowed}
        _ = test(args.project, args.name, args.test_dir, **test_kwargs)

    if args.export:
        # 匯出也傳遞同一張卡（若指定）
        exp_kwargs: Dict[str, Any] = {}
        if val_device is not None:
            exp_kwargs["device"] = _norm_device(val_device)
        _ = export(args.project, args.name, args.export_format, **exp_kwargs)
