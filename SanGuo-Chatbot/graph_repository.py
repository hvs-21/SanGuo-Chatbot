from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

from config import (
    DATA_SPLITS,
    ENTITY_MAPPING_FILE,
    RELATION_MAPPING_FILE,
    TEXT_ENCODING,
    TIME_MAPPING_FILE,
)


class SanguoDataError(Exception):
    """三国知识图谱数据读取或查询失败。"""


class EntityNotFoundError(SanguoDataError):
    """查询的实体名称不在实体映射表中。"""


class DataFormatError(SanguoDataError):
    """映射表或四元组文件格式不符合约定。"""


@dataclass(frozen=True)
class TemporalFact:
    """模型使用的整数 ID 四元组。"""

    head_id: int
    relation_id: int
    tail_id: int
    time_id: int
    split: str

    def as_id_text(self) -> str:
        return f"{self.head_id} {self.relation_id} {self.tail_id} {self.time_id}"


@dataclass(frozen=True)
class ResolvedTemporalFact:
    """翻译为中文后的时态知识图谱事实。"""

    head_name: str
    relation_name: str
    tail_name: str
    time_name: str
    split: str
    source: TemporalFact

    def as_chinese_text(self) -> str:
        return (
            f"（{self.head_name}，{self.relation_name}，"
            f"{self.tail_name}，{self.time_name}）"
        )


def load_id_name_mapping(
    csv_path: Path,
    id_column: str,
    name_column: str,
) -> Dict[int, str]:
    """读取一张 CSV 对照表，返回“整数 ID → 中文名称”。"""
    if not csv_path.is_file():
        raise SanguoDataError(f"找不到映射表：{csv_path}")

    mapping: Dict[int, str] = {}
    try:
        with csv_path.open("r", encoding=TEXT_ENCODING, newline="") as file:
            reader = csv.DictReader(file)
            required_columns = {id_column, name_column}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise DataFormatError(
                    f"{csv_path.name} 缺少列：{sorted(required_columns)}"
                )

            for line_number, row in enumerate(reader, start=2):
                try:
                    item_id = int(row[id_column])
                    item_name = row[name_column].strip()
                except (TypeError, ValueError, AttributeError) as exc:
                    raise DataFormatError(
                        f"{csv_path.name} 第 {line_number} 行格式错误"
                    ) from exc

                if not item_name:
                    raise DataFormatError(
                        f"{csv_path.name} 第 {line_number} 行名称为空"
                    )
                if item_id in mapping:
                    raise DataFormatError(
                        f"{csv_path.name} 中存在重复 ID：{item_id}"
                    )
                mapping[item_id] = item_name
    except UnicodeError as exc:
        raise SanguoDataError(f"无法按 UTF-8 读取：{csv_path}") from exc

    return mapping


def build_name_id_mapping(id_to_name: Dict[int, str]) -> Dict[str, int]:
    """把“ID → 名称”反转为查询需要的“名称 → ID”。"""
    name_to_id: Dict[str, int] = {}
    for item_id, item_name in id_to_name.items():
        if item_name in name_to_id:
            raise DataFormatError(f"实体映射表中存在重复名称：{item_name}")
        name_to_id[item_name] = item_id
    return name_to_id


def iter_quadruples(file_path: Path, split: str) -> Iterator[TemporalFact]:
    """逐行读取四元组，避免把整个数据文件重复装入内存。"""
    if not file_path.is_file():
        raise SanguoDataError(f"找不到四元组文件：{file_path}")

    try:
        with file_path.open("r", encoding=TEXT_ENCODING) as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                parts = stripped_line.split()
                if len(parts) != 4:
                    raise DataFormatError(
                        f"{file_path.name} 第 {line_number} 行不是四元组"
                    )
                try:
                    head_id, relation_id, tail_id, time_id = map(int, parts)
                except ValueError as exc:
                    raise DataFormatError(
                        f"{file_path.name} 第 {line_number} 行包含非整数 ID"
                    ) from exc

                yield TemporalFact(
                    head_id=head_id,
                    relation_id=relation_id,
                    tail_id=tail_id,
                    time_id=time_id,
                    split=split,
                )
    except UnicodeError as exc:
        raise SanguoDataError(f"无法按 UTF-8 读取：{file_path}") from exc


class SanguoGraphRepository:
    """负责加载映射、扫描四元组并把查询结果翻译为中文。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.entity_id_to_name: Dict[int, str] = {}
        self.relation_id_to_name: Dict[int, str] = {}
        self.time_id_to_name: Dict[int, str] = {}
        self.entity_name_to_id: Dict[str, int] = {}

    def load_mappings(self) -> None:
        """一次性读取实体、关系和时间三张对照表。"""
        if not self.data_dir.is_dir():
            raise SanguoDataError(f"数据目录不存在：{self.data_dir}")

        self.entity_id_to_name = load_id_name_mapping(
            self.data_dir / ENTITY_MAPPING_FILE,
            id_column="model_entity_id",
            name_column="entity_name",
        )
        self.relation_id_to_name = load_id_name_mapping(
            self.data_dir / RELATION_MAPPING_FILE,
            id_column="model_relation_id",
            name_column="model_relation",
        )
        self.time_id_to_name = load_id_name_mapping(
            self.data_dir / TIME_MAPPING_FILE,
            id_column="model_time_id",
            name_column="time_text",
        )
        self.entity_name_to_id = build_name_id_mapping(self.entity_id_to_name)

    def get_entity_id(self, query_name: str) -> int:
        """把一个完整、精确的中文实体名转换为模型 ID。"""
        normalized_name = query_name.strip()
        if not normalized_name:
            raise EntityNotFoundError("查询名称不能为空")
        if not self.entity_name_to_id:
            raise SanguoDataError("映射表尚未加载，请先调用 load_mappings()")

        entity_id = self.entity_name_to_id.get(normalized_name)
        if entity_id is None:
            similar_names = [
                name for name in self.entity_name_to_id if normalized_name in name
            ][:5]
            suggestion = (
                f"；可能的名称：{', '.join(similar_names)}" if similar_names else ""
            )
            raise EntityNotFoundError(
                f"实体映射表中没有“{normalized_name}”{suggestion}"
            )
        return entity_id

    def find_facts_by_entity_id(self, entity_id: int) -> List[TemporalFact]:
        """依次扫描 train、valid、test，寻找头或尾等于目标 ID 的事实。"""
        matches: List[TemporalFact] = []
        for split in DATA_SPLITS:
            split_path = self.data_dir / f"{split}.txt"
            for fact in iter_quadruples(split_path, split):
                if fact.head_id == entity_id or fact.tail_id == entity_id:
                    matches.append(fact)
        return matches

    def resolve_fact(self, fact: TemporalFact) -> ResolvedTemporalFact:
        """把一个整数四元组翻译成中文事实。"""
        try:
            return ResolvedTemporalFact(
                head_name=self.entity_id_to_name[fact.head_id],
                relation_name=self.relation_id_to_name[fact.relation_id],
                tail_name=self.entity_id_to_name[fact.tail_id],
                time_name=self.time_id_to_name[fact.time_id],
                split=fact.split,
                source=fact,
            )
        except KeyError as exc:
            raise DataFormatError(
                f"四元组 {fact.as_id_text()} 引用了不存在的 ID：{exc.args[0]}"
            ) from exc

    def query_by_entity_name(self, query_name: str) -> List[ResolvedTemporalFact]:
        """完成“名称 → ID → 匹配四元组 → 中文结果”的完整查询。"""
        entity_id = self.get_entity_id(query_name)
        facts = self.find_facts_by_entity_id(entity_id)
        return [self.resolve_fact(fact) for fact in facts]

