from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO


def get_model_names(model_id: str) -> Dict[Any, str]:
    """Load model and return class-name mapping."""
    try:
        model = YOLO(model_id)
        return model.names or {}
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return {}


def names_to_choice_list(names: Dict[Any, str]) -> Tuple[List[str], List[int]]:
    """Convert {id: name} map into sorted checkbox choices and IDs."""
    pairs: List[Tuple[int, str]] = []
    for k, v in (names or {}).items():
        try:
            cid = int(k)
        except Exception:
            cid = int(str(k))
        pairs.append((cid, v))

    pairs.sort(key=lambda x: x[0])
    choices = [f"{cid}: {name}" for cid, name in pairs]
    ids = [cid for cid, _ in pairs]
    return choices, ids


def parse_selected_to_ids(selected_items: Optional[List[str]]) -> List[int]:
    """Parse checkbox labels like ['0: person'] into class IDs."""
    if not selected_items:
        return []

    out: List[int] = []
    for s in selected_items:
        try:
            cid = int(str(s).split(":")[0].strip())
            out.append(cid)
        except Exception:
            pass
    return out
