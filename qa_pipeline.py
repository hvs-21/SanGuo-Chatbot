from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


WORKSPACE = Path(__file__).resolve().parent
TIRGN_SCRIPT = WORKSPACE / "predict_with_tirgn.py"
REGCN_SCRIPT = WORKSPACE / "predict_with_regcn.py"


def extract_keywords(question: str) -> List[str]:
    raw_parts = re.split(r"[\s,，。！？；：、]+", question.strip())
    return [part for part in raw_parts if part]


def query_graph_stub(question: str) -> Dict[str, object]:
    keywords = extract_keywords(question)
    return {
        "status": "stub",
        "matched_keywords": keywords[:5],
        "facts": [
            f"Potential graph fact related to '{keyword}'." for keyword in keywords[:3]
        ],
    }


def retrieve_docs_stub(question: str) -> Dict[str, object]:
    keywords = extract_keywords(question)
    return {
        "status": "stub",
        "snippets": [
            f"Retrieved text snippet for keyword '{keyword}'." for keyword in keywords[:2]
        ],
    }


def run_model(script: Path, dataset: str, use_mock: bool, extra_args: List[str] | None = None) -> Dict[str, object]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        output_path = Path(tmp.name)

    command = [sys.executable, str(script), "--dataset", dataset, "--output", str(output_path)]
    if use_mock:
        command.append("--mock")
    if extra_args:
        command.extend(extra_args)

    result = subprocess.run(
        command,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    payload: Dict[str, object]
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        output_path.unlink(missing_ok=True)
    else:
        payload = {
            "status": "error",
            "message": f"No payload produced by {script.name}.",
        }

    payload["runner_stdout"] = (result.stdout or "")[-1000:]
    payload["runner_stderr"] = (result.stderr or "")[-1000:]
    payload["runner_return_code"] = result.returncode
    return payload


def build_final_answer(
    question: str,
    graph_result: Dict[str, object],
    retrieval_result: Dict[str, object],
    tirgn_result: Dict[str, object] | None,
    regcn_result: Dict[str, object] | None,
) -> str:
    lines = [f"Question: {question}"]

    keywords = graph_result.get("matched_keywords", [])
    if keywords:
        lines.append("Graph keywords: " + ", ".join(str(item) for item in keywords))

    if regcn_result:
        status = regcn_result.get("status", "unknown")
        lines.append(f"RE-GCN status: {status}")
        metrics = regcn_result.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            if "mrr_filter_ent" in metrics:
                lines.append(f"RE-GCN filter entity MRR: {metrics['mrr_filter_ent']}")

    if tirgn_result:
        status = tirgn_result.get("status", "unknown")
        lines.append(f"TiRGN status: {status}")
        metrics = tirgn_result.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            if "mrr_filter_ent" in metrics:
                lines.append(f"TiRGN filter entity MRR: {metrics['mrr_filter_ent']}")
            if "mrr_filter_rel" in metrics:
                lines.append(f"TiRGN filter relation MRR: {metrics['mrr_filter_rel']}")

    snippets = retrieval_result.get("snippets", [])
    if snippets:
        lines.append("Retrieved evidence: " + " | ".join(str(item) for item in snippets))

    lines.append(
        "Pipeline note: this prototype already wires graph lookup, retrieval, and both model wrappers into one flow. "
        "Replace the stub graph/retrieval parts with your real Neo4j and text retrieval modules later."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype QA pipeline for your SRTP system.")
    parser.add_argument("question", help="Natural language question.")
    parser.add_argument("--dataset", default="ICEWS14")
    parser.add_argument("--skip-tirgn", action="store_true")
    parser.add_argument("--skip-regcn", action="store_true")
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument("--regcn-project-dir", default="")
    parser.add_argument("--regcn-command", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    graph_result = query_graph_stub(args.question)
    retrieval_result = retrieve_docs_stub(args.question)

    tirgn_result = None
    regcn_result = None

    if not args.skip_tirgn:
        tirgn_result = run_model(
            script=TIRGN_SCRIPT,
            dataset=args.dataset,
            use_mock=args.mock_models,
        )

    if not args.skip_regcn:
        regcn_extra_args: List[str] = []
        if args.regcn_project_dir:
            regcn_extra_args.extend(["--project-dir", args.regcn_project_dir])
        if args.regcn_command:
            regcn_extra_args.extend(["--command", args.regcn_command])
        regcn_result = run_model(
            script=REGCN_SCRIPT,
            dataset=args.dataset,
            use_mock=args.mock_models,
            extra_args=regcn_extra_args,
        )

    payload = {
        "question": args.question,
        "dataset": args.dataset,
        "graph_result": graph_result,
        "retrieval_result": retrieval_result,
        "regcn_result": regcn_result,
        "tirgn_result": tirgn_result,
    }
    payload["final_answer"] = build_final_answer(
        question=args.question,
        graph_result=graph_result,
        retrieval_result=retrieval_result,
        tirgn_result=tirgn_result,
        regcn_result=regcn_result,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
