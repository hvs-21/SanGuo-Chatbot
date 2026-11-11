"""Answer a small set of question patterns with reviewed, cited facts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from functools import cache
from pathlib import Path
from typing import Iterable

from config import (BOOK_INDEX_PATH, BOOK_PATH, BOOK_PERSON_NAMES, QA_DIRECTED_RELATIONS, QA_EVENT_RELATIONS,
                    QA_MAX_RESULTS, QA_REVIEWED_DEATHS, TEXT_ENCODING)
from graph_repository import DataFormatError, SanguoDataError


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    head: str
    relation: str
    tail: str
    time_text: str
    source_chapter: str
    source_chunk: str
    source_text: str
    time_id: str = ""

    @property
    def chapter_label(self) -> str:
        match = re.fullmatch(r"chapter_(\d+)", self.source_chapter)
        return f"第{int(match[1])}回" if match else self.source_chapter


@dataclass(frozen=True)
class FactQuery:
    person: str
    other_person: str = ""
    period: str = ""
    relations: tuple[str, ...] = ()
    direction: str = "either"
    time_question: bool = False
    path_question: bool = False
    chapter: str = ""
    confirmation: bool = False


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    facts: tuple[EvidenceFact, ...] = ()
    query_parser: str = ""
    fact_query: FactQuery | None = None
    evidence_origin: str = ""


FACT_COLUMNS = tuple(field.name for field in fields(EvidenceFact) if field.name != "time_id")
PAIR_RELATION = re.compile(r"(.+?)[和与跟](.+?)(?:在(.+?))?(?:是|有)(?:什么|何种)关系")
PAIR_EVENT_TIME = re.compile(r"(.+?)[和与跟](.+?)(?:什么时候|何时)(.+)")
PERSON_LOOKUP = re.compile(r"查询([\u3400-\u9fff·]+?)(?:在(.+?)的事迹)?")
NAME = r"[\u3400-\u9fff·（）]+?"
VERB = "|".join(sorted(QA_DIRECTED_RELATIONS, key=len, reverse=True))
DIRECTED_TIME = re.compile(rf"({NAME})(?:什么时候|何时)({VERB})(?:了)?({NAME})")
DIRECTED_CHAPTERS = re.compile(rf"({NAME})({VERB})(?:了)?({NAME})(?:是)?(?:在)?哪几回")
DIRECTED_OBJECT = re.compile(rf"({NAME})({VERB})(?:了|过)?(?:谁|何人|哪里|何处|什么官职|什么爵位)")
CHAPTER_OBJECT = re.compile(rf"(?:在)?第(\d{{1,3}})回[，,:：里中]?{DIRECTED_OBJECT.pattern}")
CHAPTER_IN_QUESTION = re.compile(rf"^({NAME})在第(\d{{1,3}})回[里中]?(?={VERB})")
DIRECTED_SUBJECT = re.compile(rf"(?:谁|何人)({VERB})(?:了|过)?({NAME})")
DIRECTED_CONFIRM = re.compile(rf"(?:请问)?({NAME})({VERB})(?:了|过)?({NAME})(?:吗|么)")
PAIR_PATH = re.compile(rf"({NAME})[和与跟]({NAME})(?:在(.+?))?(?:有哪些|有什么|有何)间接(?:关联|关系)")
INSUFFICIENT = "现有资料不足，无法回答这个问题。"


def load_evidence_facts(path: Path) -> list[EvidenceFact]:
    """Load reviewed facts with a complete citation from the semantic master table."""
    facts: list[EvidenceFact] = []
    try:
        with path.open(encoding=TEXT_ENCODING, newline="") as source:
            rows = csv.DictReader(source, strict=True)
            required = {*FACT_COLUMNS, "quality_status"}
            if not required.issubset(rows.fieldnames or ()):
                raise DataFormatError(f"问答数据缺少必要字段：{path.name}")

            for row in rows:
                if None in row or any(row[column] is None for column in required):
                    raise DataFormatError(f"问答数据第 {rows.line_num} 行列数不正确：{path.name}")
                if row["quality_status"].strip() != "PASS":
                    continue

                values = {column: row[column].strip() for column in FACT_COLUMNS}
                if all(values.values()):
                    facts.append(EvidenceFact(**values, time_id=(row.get("time_id") or "").strip()))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SanguoDataError(f"无法读取问答数据 {path}：{exc}") from exc

    return facts


def load_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    """Collect every candidate name for each alias so ambiguity stays visible."""
    entries = defaultdict(set)
    with path.open(encoding=TEXT_ENCODING, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"].strip() != "active":
                continue
            canonical = row["entity_name"].strip()
            if not canonical:
                continue
            entries[canonical].add(canonical)
            for alias in re.split(r"[;；、|，,]", row["aliases"]):
                if alias.strip():
                    entries[alias.strip()].add(canonical)
    for canonical, aliases in BOOK_PERSON_NAMES.items():
        entries[canonical].add(canonical)
        for alias in aliases:
            entries[alias].add(canonical)
    return {alias: tuple(sorted(names)) for alias, names in entries.items()}


def evidence_fingerprint(facts: Iterable[EvidenceFact]) -> str:
    rows = [dict(asdict(fact), ordinal=index) for index, fact in enumerate(facts)]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def parse_question(question: str) -> FactQuery | None:
    """Accept explicit question patterns; never fall back to arbitrary name matches."""
    normalized = re.sub(r"\s+", "", question).rstrip("？?。！!")
    normalized = CHAPTER_IN_QUESTION.sub(r"第\2回，\1", normalized)
    if match := CHAPTER_OBJECT.fullmatch(normalized):
        chapter = int(match[1])
        if 1 <= chapter <= 120:
            return FactQuery(match[2], relations=QA_DIRECTED_RELATIONS[match[3]], direction="head",
                             chapter=f"chapter_{chapter:03d}")
        return None
    if match := PAIR_PATH.fullmatch(normalized):
        return FactQuery(match[1], match[2], match[3] or "", path_question=True)
    if match := PAIR_RELATION.fullmatch(normalized):
        return FactQuery(match[1], match[2], match[3] or "")
    if match := PAIR_EVENT_TIME.fullmatch(normalized):
        relation = QA_EVENT_RELATIONS.get(match[3])
        if relation:
            return FactQuery(match[1], match[2], relations=(relation,), time_question=True)
    if match := DIRECTED_TIME.fullmatch(normalized):
        return FactQuery(match[1], match[3], relations=QA_DIRECTED_RELATIONS[match[2]],
                         direction="head", time_question=True)
    if match := DIRECTED_CHAPTERS.fullmatch(normalized):
        return FactQuery(match[1], match[3], relations=QA_DIRECTED_RELATIONS[match[2]],
                         direction="head", time_question=True)
    if match := DIRECTED_CONFIRM.fullmatch(normalized):
        return FactQuery(match[1], match[3], relations=QA_DIRECTED_RELATIONS[match[2]],
                         direction="head", confirmation=True)
    if match := DIRECTED_OBJECT.fullmatch(normalized):
        return FactQuery(match[1], relations=QA_DIRECTED_RELATIONS[match[2]], direction="head")
    if match := DIRECTED_SUBJECT.fullmatch(normalized):
        return FactQuery(match[2], relations=QA_DIRECTED_RELATIONS[match[1]], direction="tail")
    if match := PERSON_LOOKUP.fullmatch(normalized):
        return FactQuery(match[1], period=match[2] or "")
    return None


def find_matching_facts(query: FactQuery, facts: Iterable[EvidenceFact]) -> list[EvidenceFact]:
    """Require every supplied slot to match; keep the fact's original direction."""
    matches: list[EvidenceFact] = []
    period = query.period.removesuffix("阶段") or query.period
    if query.path_question:
        return find_two_hop_path(query.person, query.other_person, period, facts)
    for fact in facts:
        if query.direction == "head":
            matched = fact.head == query.person and (not query.other_person or fact.tail == query.other_person)
        elif query.direction == "tail":
            matched = fact.tail == query.person and (not query.other_person or fact.head == query.other_person)
        else:
            participants = {fact.head, fact.tail}
            matched = query.person in participants and (not query.other_person or participants == {query.person, query.other_person})
        if not matched:
            continue
        if query.relations and fact.relation not in query.relations:
            continue
        if query.chapter and query.chapter != fact.source_chapter:
            continue
        # For example, “桃园结义阶段” is part of the recorded combined stage.
        if period and period not in fact.time_text:
            continue
        matches.append(fact)
        if len(matches) >= QA_MAX_RESULTS:
            break
    return matches


def find_two_hop_path(start: str, finish: str, period: str, facts: Iterable[EvidenceFact]) -> list[EvidenceFact]:
    """Find a simple two-edge chain. Each edge keeps its own recorded direction."""
    if start == finish:
        return []
    first_edges = []
    last_edges = {}
    for fact in facts:
        if period not in fact.time_text or fact.head == fact.tail:
            continue
        for name, neighbor in ((fact.head, fact.tail), (fact.tail, fact.head)):
            if name == start and neighbor != finish:
                first_edges.append((neighbor, fact))
            elif name == finish and neighbor != start:
                last_edges.setdefault(neighbor, fact)
    for neighbor, first in first_edges:
        if neighbor in last_edges:
            return [first, last_edges[neighbor]]
    return []


@cache
def reviewed_death(person: str) -> EvidenceFact:
    actor, chapter, start, end, anchor = QA_REVIEWED_DEATHS[person]
    raw = BOOK_PATH.read_bytes()
    index = json.loads(BOOK_INDEX_PATH.read_text(encoding="utf-8"))
    bounds = index["chapters"][str(chapter)]
    quote = "".join(line.strip() for line in raw.decode("utf-8-sig").splitlines()[start - 1:end])
    if (hashlib.sha256(raw).hexdigest() != index["sha256"] or anchor not in quote
            or not bounds["start_line"] < start <= end <= bounds["end_line"]):
        raise ValueError("关羽遇害原文定位无效")
    return EvidenceFact(f"BOOK-{chapter:03d}-{start}", actor, "下令处死", person, f"第{chapter}回叙事",
                        f"chapter_{chapter:03d}", f"第{start}–{end}行", quote)


def answer_reviewed_death(question, query, parser, aliases, facts):
    target = query.person if query.direction == "tail" else query.other_person
    if (target not in QA_REVIEWED_DEATHS or not {"斩杀", "杀害"}.intersection(query.relations)
            or query.period or query.chapter):
        return None
    try:
        fact = reviewed_death(target)
    except (OSError, ValueError, KeyError):
        return None
    if query.direction == "tail" and query.other_person and query.other_person != fact.head:
        return None
    summary = f"《三国演义》{fact.chapter_label}记载，{fact.head}下令处死{fact.tail}。"
    if query.confirmation and query.person != fact.head:
        if query.person not in (aliases or {}) and not any(query.person in (row.head, row.tail) for row in facts):
            return None
        summary = "没有。" + summary
    elif query.direction != "tail" and query.person != fact.head:
        return None
    return Answer(question, summary, (fact,), parser, query, "review")


def answer_question(question: str, facts: Iterable[EvidenceFact], *, aliases=None, lookup=None,
                    semantic_parse=None) -> Answer:
    """Return a template answer and the exact records supporting it."""
    query = parse_question(question)
    parser = "rule"
    if query is None and semantic_parse:
        query = semantic_parse(question)
        parser = "bert"
    if query is None:
        return Answer(question, f"{INSUFFICIENT} 当前支持人物、关系和有明确对象的事件问题。")

    if aliases:
        names = []
        for value in (query.person, query.other_person):
            candidates = aliases.get(value, (value,))
            if len(candidates) > 1:
                return Answer(question, f"“{value}”可能指{'、'.join(candidates)}，请使用完整名称。", query_parser=parser)
            names.append(candidates[0])
        query = replace(query, person=names[0], other_person=names[1])

    matches = tuple(lookup(query) if lookup else find_matching_facts(query, facts))
    if not matches:
        return answer_reviewed_death(question, query, parser, aliases, facts) or Answer(
            question, INSUFFICIENT, query_parser=parser, fact_query=query
        )

    if query.path_question:
        first, second = matches
        via = first.tail if first.head == query.person else first.head
        summary = (f"找到一条经由{via}的关联路径。"
                   f"{first.head} —{first.relation}→ {first.tail}。"
                   f"{second.head} —{second.relation}→ {second.tail}。"
                   "两条记录各有出处和时间。")
    elif query.time_question:
        periods = "、".join(dict.fromkeys(fact.time_text for fact in matches))
        chapters = "、".join(dict.fromkeys(fact.chapter_label for fact in matches))
        summary = f"资料记载的时间阶段：{periods}。见《三国演义》{chapters}。"
    elif query.other_person or query.relations:
        relations = "；".join(
            dict.fromkeys(f"{fact.head} —{fact.relation}→ {fact.tail}" for fact in matches)
        )
        summary = f"资料记载的关系：{relations}。"
        if query.confirmation:
            summary = "是。" + summary
    else:
        summary = f"以下是与{query.person}有关的有来源记录。"
    return Answer(question, summary, matches, parser, query)
