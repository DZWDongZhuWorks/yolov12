# app_utils/model_registry.py
import json
from typing import List, Tuple
from .config import DEFAULT_MODELS, PERSIST_FILE


def load_model_choices() -> Tuple[List[str], List[str]]:
    """
    回傳 (下拉列表 choices, 已儲存的自訂模型 saved_custom)
    """
    try:
        with open(PERSIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            saved = data.get("models", [])
    except Exception:
        saved = []
    seen = set()
    choices: List[str] = []
    for x in DEFAULT_MODELS + saved:
        if x not in seen:
            seen.add(x)
            choices.append(x)
    return choices, saved


def _persist_model_choice(saved_list: List[str], model_id: str):
    if model_id and model_id not in DEFAULT_MODELS and model_id not in saved_list:
        saved_list.append(model_id)
        try:
            with open(PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump({"models": saved_list}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    seen = set()
    choices: List[str] = []
    for x in DEFAULT_MODELS + saved_list:
        if x not in seen:
            seen.add(x)
            choices.append(x)
    return choices, saved_list


def persist_model_choices(
    saved_list: List[str],
    model_ids: List[str],
) -> Tuple[List[str], List[str]]:
    """
    一次處理多個模型 id，回傳 (新的 choices, 更新後 saved_list)
    """
    choices, saved = None, saved_list
    for mid in (model_ids or []):
        choices, saved = _persist_model_choice(saved, mid)

    # 沒有新的自訂模型時，仍組出 choices
    if choices is None:
        seen = set()
        choices = []
        for x in DEFAULT_MODELS + saved:
            if x not in seen:
                seen.add(x)
                choices.append(x)

    return choices, saved
