# app_utils/model_cache.py
"""共用 YOLO 模型快取。

避免每次推論 / 讀取 class names 時重新從磁碟載入模型
（重複 instantiate 也可能導致 CUDA Out Of Memory）。
"""
import os
import threading
from typing import Dict, Tuple

from ultralytics import YOLO

_lock = threading.Lock()
_cache: Dict[Tuple[str, float], YOLO] = {}


def _cache_key(model_id: str) -> Tuple[str, float]:
    # 以 mtime 作為 key 的一部分：自訂權重檔被覆蓋時會自動重新載入
    try:
        mtime = os.path.getmtime(model_id)
    except OSError:
        mtime = -1.0
    return (str(model_id), mtime)


def get_model(model_id: str) -> YOLO:
    key = _cache_key(model_id)
    with _lock:
        model = _cache.get(key)
        if model is None:
            model = YOLO(model_id)
            # 清掉同一 model_id 的舊版本（檔案已更新）
            for stale in [k for k in _cache if k[0] == key[0] and k != key]:
                _cache.pop(stale, None)
            _cache[key] = model
        return model


def clear_model_cache() -> None:
    with _lock:
        _cache.clear()
