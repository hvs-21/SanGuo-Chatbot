"""Read offline model candidates without importing a model or starting training."""

import json
import math
from pathlib import Path
from config import SANGUO_DATA_DIR
from prediction_review import attach_answers


def validate_predictions(data: dict) -> None:
    """Reject incompatible exports and misleading/invalid candidate rankings."""
    if (data["schema_version"], data["split"], data["task"], data["protocol"]) != (
        1, "valid", "tail_prediction", "single_step_raw"
    ):
        raise ValueError("Unsupported prediction protocol")
    if set(data["models"]) != {"tirgn", "regcn"} or not data["cases"]:
        raise ValueError("Both models and at least one case are required")
    top_k = data["top_k"]
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("Invalid candidate count")
    seen = set()
    for case in data["cases"]:
        if case["id"] in seen or case["id"] != f'{case["head_id"]}:{case["relation_id"]}:{case["time_id"]}':
            raise ValueError("Invalid or duplicate query")
        seen.add(case["id"])
        if not all(isinstance(case[key], str) and case[key] for key in ("head", "relation", "time")):
            raise ValueError("Missing Chinese query labels")
        for model in data["models"]:
            candidates = case["predictions"][model]
            if len(candidates) != top_k or len({item["entity_id"] for item in candidates}) != top_k:
                raise ValueError("Missing or duplicate candidates")
            previous = float("inf")
            for rank, item in enumerate(candidates, 1):
                score = item["score"]
                if (item["rank"] != rank or not isinstance(item["entity"], str) or not item["entity"]
                        or not math.isfinite(score) or score > previous):
                    raise ValueError("Invalid candidate ranking")
                previous = score


def load_predictions(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_predictions(data)
    except FileNotFoundError:
        return {"available": False, "message": "尚未生成模型预测案例，事实问答仍可使用。"}
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, OverflowError):
        return {"available": False, "message": "预测文件无法读取或格式不正确，请重新导出。"}
    try:
        # Labels are attached only for explanation, after the saved ranking is loaded.
        label_dir = path.parent / "prediction_data" if "data_version" in data else SANGUO_DATA_DIR
        attach_answers(data, label_dir)
        data["review_available"] = True
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        data["review_available"] = False
    return dict(data, available=True)
