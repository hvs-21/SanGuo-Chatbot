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
DEFAULT_WSL_ENV = "/home/hvs/ENTER/envs/regcn"
DEFAULT_PROJECT_DIR = ""

METRIC_PATTERNS = {
    "mrr_raw_ent": r"MRR \(raw_ent\):\s*([0-9.]+)",
    "hits1_raw_ent": r"Hits \(raw_ent\) @ 1:\s*([0-9.]+)",
    "hits3_raw_ent": r"Hits \(raw_ent\) @ 3:\s*([0-9.]+)",
    "hits10_raw_ent": r"Hits \(raw_ent\) @ 10:\s*([0-9.]+)",
    "mrr_filter_ent": r"MRR \(filter_ent\):\s*([0-9.]+)",
    "hits1_filter_ent": r"Hits \(filter_ent\) @ 1:\s*([0-9.]+)",
    "hits3_filter_ent": r"Hits \(filter_ent\) @ 3:\s*([0-9.]+)",
    "hits10_filter_ent": r"Hits \(filter_ent\) @ 10:\s*([0-9.]+)",
    "mrr_raw_rel": r"MRR \(raw_rel\):\s*([0-9.]+)",
    "hits1_raw_rel": r"Hits \(raw_rel\) @ 1:\s*([0-9.]+)",
    "hits3_raw_rel": r"Hits \(raw_rel\) @ 3:\s*([0-9.]+)",
    "hits10_raw_rel": r"Hits \(raw_rel\) @ 10:\s*([0-9.]+)",
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
        "model": "RE-GCN",
        "task": "interpolation",
        "dataset": dataset,
        "mode": mode,
        "status": "mock",
        "metrics": {
            "mrr_filter_ent": 0.41,
            "hits1_filter_ent": 0.31,
            "hits3_filter_ent": 0.47,
            "hits10_filter_ent": 0.62,
        },
        "top_predictions": [
            {"target": "entity", "label": "candidate_x", "score": 0.83},
            {"target": "entity", "label": "candidate_y", "score": 0.78},
        ],
        "note": "Mock mode for RE-GCN module.",
    }


def save_payload(payload: Dict[str, object], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_wsl_command(env_path: str, project_dir: str, inner_command: str) -> List[str]:
    commands = [
        "source ~/miniconda3/etc/profile.d/conda.sh",
        f"conda activate {shlex.quote(env_path)}",
        f"cd {shlex.quote(project_dir)}",
        inner_command,
    ]
    return ["wsl", "bash", "-lc", " && ".join(commands)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or mock the RE-GCN module through WSL.")
    parser.add_argument("--dataset", default="ICEWS14")
    parser.add_argument("--mode", default="test", choices=["test", "train"])
    parser.add_argument("--wsl-env-path", default=DEFAULT_WSL_ENV)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument(
        "--command",
        default="",
        help="Full inner command to run inside the RE-GCN project directory, e.g., python main.py --dataset ICEWS14 --test",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.mock:
        payload = mock_payload(args.dataset, args.mode)
        save_payload(payload, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not args.project_dir:
        payload = {
            "model": "RE-GCN",
            "task": "interpolation",
            "dataset": args.dataset,
            "mode": args.mode,
            "status": "error",
            "message": "Set --project-dir to your RE-GCN source directory in WSL.",
        }
        save_payload(payload, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    if not args.command:
        default_command = f"python main.py --dataset {shlex.quote(args.dataset)} --test"
    else:
        default_command = args.command

    command = build_wsl_command(
        env_path=args.wsl_env_path,
        project_dir=args.project_dir,
        inner_command=default_command,
    )

    if args.dry_run:
        payload = {
            "model": "RE-GCN",
            "task": "interpolation",
            "dataset": args.dataset,
            "mode": args.mode,
            "status": "dry-run",
            "command": command,
        }
        save_payload(payload, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    result = subprocess.run(
        command,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    payload = {
        "model": "RE-GCN",
        "task": "interpolation",
        "dataset": args.dataset,
        "mode": args.mode,
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode,
        "metrics": parse_metrics(stdout),
        "command": command,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
    save_payload(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
