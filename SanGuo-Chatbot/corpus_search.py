"""Search the pinned full text and preserve original wording and line references."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

from config import QA_MAX_RESULTS, WEB_CORPUS_MAX_LINES, WEB_MAX_SEARCH_TERMS
from book_search import PassageIndex


def windows_simplify(text: str) -> str:
    if sys.platform != "win32":
        return text
    if not text:
        return ""
    # Windows maps a search copy. The source text returned to the browser stays intact.
    mapper = ctypes.windll.kernel32.LCMapStringEx
    args = ("zh-CN", 0x02000000, text, -1)
    size = mapper(*args, None, 0, None, None, 0)
    if not size:
        raise OSError("Windows could not normalize Chinese text")
    buffer = ctypes.create_unicode_buffer(size)
    if not mapper(*args, buffer, size, None, None, 0):
        raise OSError("Windows could not normalize Chinese text")
    return buffer.value


def normalized(text: str) -> str:
    return windows_simplify("".join(text.split())).casefold()


@dataclass(frozen=True)
class Passage:
    chapter: int
    heading: str
    first_line: int
    last_line: int
    text: str
    searchable: str

    def citation(self) -> dict[str, str]:
        return {"id": "Gutenberg 23950", "fact": self.heading, "time": "",
                "chapter": f"第{self.chapter}回", "chunk": f"第{self.first_line}–{self.last_line}行",
                "quote": self.text}


def chapter_passages(lines: list[str], first: int, last: int):
    group = []
    for index in range(first - 1, last):
        if not lines[index].strip() or lines[index].startswith("　　"):
            if group:
                yield group
                group = []
        if lines[index].strip():
            group.append((index + 1, lines[index].strip()))
    if group:
        yield group


class BookCorpus:
    def __init__(self, passages: tuple[Passage, ...], chapter_count: int):
        self.passages = passages
        self.chapter_count = chapter_count
        self.index = PassageIndex(passages)

    @classmethod
    def load(cls, book_path: Path, index_path: Path) -> BookCorpus:
        raw = book_path.read_bytes()
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if hashlib.sha256(raw).hexdigest() != index["sha256"]:
            raise ValueError("原文与章节索引版本不一致")
        lines = raw.decode("utf-8-sig").splitlines()
        chapters = index["chapters"]
        if list(map(int, chapters)) != list(range(1, 121)):
            raise ValueError("章节索引必须包含完整的 120 回")
        passages = []
        for chapter, bounds in chapters.items():
            start, end = bounds["start_line"], bounds["end_line"]
            if not 1 <= start < end <= len(lines) or lines[start - 1] != bounds["heading"]:
                raise ValueError(f"第 {chapter} 回的原文定位无效")
            for group in chapter_passages(lines, start + 1, end):
                for offset in range(0, len(group), WEB_CORPUS_MAX_LINES - 2):
                    part = group[offset:offset + WEB_CORPUS_MAX_LINES]
                    text = "".join(line for _, line in part)
                    passages.append(Passage(int(chapter), bounds["heading"], part[0][0], part[-1][0], text, normalized(text)))
                    if offset + WEB_CORPUS_MAX_LINES >= len(group):
                        break
        if {p.chapter for p in passages} != set(range(1, 121)):
            raise ValueError("有章节缺少正文")
        return cls(tuple(passages), len(chapters))

    def search(self, query: str) -> tuple[Passage, ...]:
        chapter = None
        if match := re.match(r"^第\s*(\d{1,3})\s*回\s*", query):
            chapter = int(match[1])
            query = query[match.end():]
        terms = tuple(dict.fromkeys(normalized(term) for term in query.split()))
        if not terms or len(terms) > WEB_MAX_SEARCH_TERMS:
            return ()
        matches = []
        for passage in self.passages:
            if chapter is not None and passage.chapter != chapter:
                continue
            if all(term in passage.searchable for term in terms):
                matches.append(passage)
                if len(matches) == QA_MAX_RESULTS:
                    break
        return tuple(matches)
