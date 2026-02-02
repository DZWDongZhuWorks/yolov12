
from typing import Optional, List, Union
import numpy as np

def _normalize_class_filter(raw_value) -> Optional[List[object]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple, set)):
        tokens = [t for t in raw_value if t not in (None, "")]
    else:
        text = str(raw_value).strip()
        if not text or text.lower() in {"all", "*", "any"}:
            return None
        tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    if not tokens:
        return None
    normalized: List[object] = []
    for token in tokens:
        if isinstance(token, (int, np.integer)):
            normalized.append(int(token))
            continue
        token_str = str(token).strip()
        if ":" in token_str:
            prefix = token_str.split(":", 1)[0].strip()
            try:
                normalized.append(int(prefix))
                continue
            except Exception:
                token_str = token_str
        try:
            normalized.append(int(token_str))
        except Exception:
            normalized.append(token_str)
    return normalized or None

# Test Cases
print("Test 1: Normal comma string")
print(_normalize_class_filter("0: person, 1: car"))

print("\nTest 2: List (Backend direct)")
print(_normalize_class_filter(["0: person", "1: car"]))

print("\nTest 3: List string representation (The bug)")
bug_input = "['0: person', '1: car']"
print(f"Input: {bug_input}")
parsed = _normalize_class_filter(bug_input)
print(f"Parsed: {parsed}")

print("\nTest 4: Empty list string")
print(_normalize_class_filter("[]"))
