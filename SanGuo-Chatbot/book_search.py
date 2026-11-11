"""Rank source passages using words and names from a question."""

from collections import Counter, defaultdict
from math import log
import re

from config import QA_MAX_RESULTS


CHAPTER = re.compile(r"第\s*(\d{1,3})\s*回")
QUESTION_WORDS = re.compile(
    r"请问|三国演义|原文|书中|发生在哪|什么时候|是怎么|叫什么|为什么|怎么办|怎么样|"
    r"哪一位|哪一个|哪一回|第几回|在哪|哪座|哪条|哪里|何处|什么|多少|几位|多重|多长|"
    r"有哪些|是谁|哪个|什么关系|有多|是什么|的是|请|的时候|(?:了吗|吗|呢)$"
)
# Common wording differences between questions and the source text.
WORD_VARIANTS = {
    "赠送": ("献与", "送与"),
    "送给": ("献与", "递与"),
    "弹琴": ("操琴", "抚琴"),
    "城楼": ("敌楼", "城上"),
    "骑兵": ("百骑", "轻骑", "铁骑"),
    "条船": ("只船", "艘船"),
    "偷渡": ("进兵", "度过", "小路"),
    "东风": ("东南风",),
    "空城计": ("空城", "开四门"),
    "长坂": ("长板", "长阪"),
    "喊了": ("喝曰", "大叫"),
}


def grams(text):
    """Return overlapping Chinese bigrams and trigrams without crossing punctuation."""
    return {word[start:start + size]
            for word in re.findall(r"[\u3400-\u9fff]+", text)
            for size in (2, 3) for start in range(len(word) - size + 1)}


def name_pattern(aliases):
    names = sorted((name for name, targets in aliases.items() if len(targets) == 1 and len(name) >= 2),
                   key=lambda name: (-len(name), name))
    return re.compile("|".join(map(re.escape, names))) if names else None


class PassageIndex:
    def __init__(self, passages):
        self.passages = passages
        postings = defaultdict(list)
        for index, passage in enumerate(passages):
            for term in grams(passage.searchable):
                postings[term].append(index)
        self.postings = {term: tuple(indices) for term, indices in postings.items()}

    def search(self, text, aliases):
        chapters = {int(match[1]) for match in CHAPTER.finditer(text)}
        text = CHAPTER.sub(" ", text)
        if pattern := name_pattern(aliases):
            names = {aliases[match[0]][0] for match in pattern.finditer(text)}
            text = pattern.sub(" ", text)
        else:
            names = set()
        terms = grams(QUESTION_WORDS.sub(" ", text))
        variants = {part for word, alternatives in WORD_VARIANTS.items() if word in text
                    for alternative in alternatives for part in grams(alternative)}
        terms.update(variants)
        scores = Counter()
        overlaps = Counter()
        for term in sorted(terms):
            matches = self.postings.get(term, ())
            if not matches:
                continue
            weight = log(1 + len(self.passages) / len(matches)) * (1.6 if len(term) == 3 else 1)
            for index in matches:
                if chapters and self.passages[index].chapter not in chapters:
                    continue
                scores[index] += weight
                overlaps[index] += 1
        forms = {name: {alias for alias, targets in aliases.items() if targets == (name,) and len(alias) >= 2}
                 for name in names}
        for index in tuple(scores):
            passage = self.passages[index]
            names_found = sum(any(alias in passage.searchable for alias in spellings) for spellings in forms.values())
            variant_found = names_found and any(term in passage.searchable for term in variants)
            if overlaps[index] < 2 and not (names_found >= 2 or variant_found or (chapters and names_found)):
                del scores[index]
                continue
            scores[index] *= 1 + 0.42 * names_found
        return tuple(self.passages[index] for index, _ in scores.most_common(QA_MAX_RESULTS))
