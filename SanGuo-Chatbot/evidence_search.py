"""Literal passage search over the reviewed excerpts in the SANGUO table."""

from __future__ import annotations

from typing import Iterable

from config import QA_MAX_RESULTS, WEB_MAX_SEARCH_TERMS
from fact_qa import EvidenceFact


def search_passages(keywords: str, facts: Iterable[EvidenceFact]) -> tuple[EvidenceFact, ...]:
    """Require every search term to occur in the excerpt; deduplicate shared passages."""
    terms = tuple(dict.fromkeys(keywords.casefold().split()))
    if not terms or len(terms) > WEB_MAX_SEARCH_TERMS:
        return ()

    matches: list[EvidenceFact] = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        text = fact.source_text.casefold()
        if not all(term in text for term in terms):
            continue
        passage = (fact.source_chunk, fact.source_text)
        if passage in seen:
            continue
        seen.add(passage)
        matches.append(fact)
        if len(matches) == QA_MAX_RESULTS:
            break
    return tuple(matches)
