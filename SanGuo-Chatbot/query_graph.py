from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from config import SANGUO_DATA_DIR
from graph_repository import ResolvedTemporalFact, SanguoDataError, SanguoGraphRepository


def configure_console_encoding() -> None:
    """让 Windows 与 VS Code 终端稳定显示 UTF-8 中文。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def query_entity(query_name: str, data_dir: Path) -> tuple[int, List[ResolvedTemporalFact]]:
    """加载数据并按中文实体名查询，供命令行或其他模块调用。"""
    repository = SanguoGraphRepository(data_dir)
    repository.load_mappings()
    entity_id = repository.get_entity_id(query_name)
    facts = repository.find_facts_by_entity_id(entity_id)
    matches = [repository.resolve_fact(fact) for fact in facts]
    return entity_id, matches


def print_query_result(
    query_name: str,
    entity_id: int,
    matches: List[ResolvedTemporalFact],
) -> None:
    """以“原始 ID → 中文事实”的形式输出结果。"""
    print(f"查询实体：{query_name} → ID {entity_id}")
    print(f"共找到 {len(matches)} 条关联事实：")

    for match in matches:
        print(
            f"[{match.split}] {match.source.as_id_text()} "
            f"→ {match.as_chinese_text()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 SANGUO 的 train、valid、test 四元组中查询实体。"
    )
    parser.add_argument(
        "query_name",
        nargs="?",
        help="要查询的完整中文实体名；不传时进入交互输入。",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SANGUO_DATA_DIR,
        help=f"SANGUO 数据目录，默认：{SANGUO_DATA_DIR}",
    )
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    query_name = args.query_name or input("请输入要查询的实体名称：").strip()

    try:
        entity_id, matches = query_entity(query_name, args.data_dir)
        print_query_result(query_name, entity_id, matches)
    except SanguoDataError as exc:
        print(f"查询失败：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
