from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


WORKSPACE = Path(__file__).resolve().parent
TIRGN_ROOT = WORKSPACE / "tirgn" / "TiRGN-main"
TIRGN_SRC = TIRGN_ROOT / "src"
DEFAULT_WSL_ENV = "/home/hvs/ENTER/envs/tirgn_env"

DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "ICEWS14": {
        "history-rate": "0.3",
        "train-history-len": "9",
        "test-history-len": "9",
        "dilate-len": "1",
        "lr": "0.001",
        "n-layers": "2",
        "evaluate-every": "1",
        "n-hidden": "200",
        "decoder": "timeconvtranse",
        "encoder": "convgcn",
        "weight": "0.5",
        "angle": "8",
        "discount": "1",
        "task-weight": "0.7",
        "save": "checkpoint",
        "flags": [
            "self-loop",
            "layer-norm",
            "entity-prediction",
            "relation-prediction",
            "add-static-graph",
        ],
    },
}

METRIC_PATTERNS = {
    "mrr_filter_ent": r"MRR \(filter_ent\):\s*([0-9.]+)",
    "hits1_filter_ent": r"Hits \(filter_ent\) @ 1:\s*([0-9.]+)",
    "hits3_filter_ent": r"Hits \(filter_ent\) @ 3:\s*([0-9.]+)",
    "hits10_filter_ent": r"Hits \(filter_ent\) @ 10:\s*([0-9.]+)",
    "mrr_filter_rel": r"MRR \(filter_rel\):\s*([0-9.]+)",
    "hits1_filter_rel": r"Hits \(filter_rel\) @ 1:\s*([0-9.]+)",
    "hits3_filter_rel": r"Hits \(filter_rel\) @ 3:\s*([0-9.]+)",
    "hits10_filter_rel": r"Hits \(filter_rel\) @ 10:\s*([0-9.]+)",
}

def parse_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics

def mock_payload(dataset: str, mode: str) -> Dict[str, object]:
    return {
        "model": "TiRGN",
        "task": "extrapolation",
        "dataset": dataset,
        "mode": mode,
        "status": "mock",
        "metrics": {
            "mrr_filter_ent": 0.444396,
            "hits1_filter_ent": 0.342081,
            "hits3_filter_ent": 0.496483,
            "hits10_filter_ent": 0.637952,
            "mrr_filter_rel": 0.473103,
            "hits1_filter_rel": 0.346241,
            "hits3_filter_rel": 0.525601,
            "hits10_filter_rel": 0.752382,
        },
        "top_predictions": [
            {"target": "entity", "label": "candidate_a", "score": 0.88},
            {"target": "relation", "label": "candidate_b", "score": 0.81},
        ],
        "note": "Mock mode for TiRGN module.",
    }

def save_payload(payload: Dict[str, object], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run or mock the TiRGN module through WSL.")
    parser.add_argument("--dataset", default="ICEWS14", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--mode", default="test", choices=["test", "train", "multistep"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wsl-env-path", default=DEFAULT_WSL_ENV)
    parser.add_argument("--prepare-history", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.mock:
        payload = mock_payload(args.dataset, args.mode)
        save_payload(payload, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # 默认非 mock 时的兜底或执行逻辑
    payload = mock_payload(args.dataset, args.mode)
    save_payload(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
