from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import QA_FACTS_PATH
from fact_qa import Answer, answer_question, load_evidence_facts
from graph_repository import SanguoDataError
from query_graph import configure_console_encoding


def format_answer(result: Answer) -> str:
    """Print the answer alongside its source records, without model predictions."""
    lines = [f"问题：{result.question}", f"回答：{result.answer}"]
    for fact in result.facts:
        lines.extend(
            (
                "",
                f"事实 [{fact.fact_id}]：{fact.head} —{fact.relation}→ {fact.tail}",
                f"时间：{fact.time_text}",
                f"证据：{fact.source_text}",
                f"来源：《三国演义》{fact.chapter_label}（{fact.source_chunk}）",
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地三国图谱事实及文本证据回答问题。")
    parser.add_argument("question", help="例如：刘备和关羽在桃园结义阶段是什么关系？")
    parser.add_argument("--facts-file", type=Path, default=QA_FACTS_PATH, help="带来源的事实主表 CSV")
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        facts = load_evidence_facts(args.facts_file)
    except SanguoDataError as exc:
        print(f"问答失败：{exc}", file=sys.stderr)
        return 1

    print(format_answer(answer_question(args.question, facts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
