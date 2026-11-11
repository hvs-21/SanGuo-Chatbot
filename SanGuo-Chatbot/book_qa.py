"""Retrieve full-text passages and distinguish direct narrative evidence from clues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import islice

from book_search import CHAPTER, name_pattern
from config import BOOK_RELATION_EVENTS, QA_MAX_RESULTS
from corpus_search import Passage, normalized


OUTSIDE = re.compile("智能手机|手机|电子邮件|身份证|互联网|飞机票")
TOPIC_TERMS = ("卧病", "病情", "染病", "病重", "病危", "永安宫", "吐血", "短刀", "乱箭", "刺杀",
               "火攻", "水寨", "借箭", "投降", "盟约", "交战", "出兵", "首级", "结义", "中箭",
               "击败", "救援", "护送", "追赶", "追击", "攻打", "被擒", "射死", "劝降", "死", "病")
DEATH = r"(?:去世|遇害|病逝|病故|死亡|身亡|死|驾崩)"
DEATH_QUESTION = re.compile(
    rf"(?:是)?(?:(?:在)?(?:哪(?:一|几)?回|第几回|什么时候|何时)(?:{DEATH})?|"
    rf"(?:怎么|如何){DEATH}|{DEATH}(?:时)?(?:是)?(?:在哪(?:一|几)?回|在哪儿|在哪里|在什么地方|"
    rf"第几回|什么时候|何时|多少岁|几岁))(?:的|呢)?"
)
KILL = re.compile(r"(?:亲手)?(?:杀死|斩杀|杀害|杀)(?:了|过)?")
RELATION_QUESTION = re.compile(r"(?:是|有)(?:什么|何种|何)关系|(?:的)?关系(?:如何|怎么样)")
TITLE_RANGE = {"刘备": ("先主", 80, 85), "刘禅": ("后主", 85, 120)}
LATEST_OFFICE_QUESTION = re.compile(
    r"(?:生前)?最后(?:(?:的)?(?:一个|一任)?(?:官职|职务)是什么|"
    r"(?:担任|当)(?:了|过)?什么(?:官职|职务|官))(?:呢)?"
)
OFFICE = r"[\u3400-\u9fff]{0,8}?(?:将军|校尉|太守|牧|刺史|尚书令|丞相|都督|长史|司马|太傅|侍中)"
OFFICE_DETAIL = re.compile(rf"(?P<duty>(?:兼领|兼|领){OFFICE})|(?:封)?(?P<rank>[\u3400-\u9fff]{{1,5}}侯)")
SHIP_COUNT = re.compile(r"借(?:了|到|来)?(?:多少|几)(?:条|艘|只)船$")


@dataclass(frozen=True)
class BookAnswer:
    answer: str
    passages: tuple[Passage, ...]
    kind: str = "direct"


def names_in(text, aliases):
    pattern = name_pattern(aliases)
    return [(match, aliases[match[0]][0]) for match in pattern.finditer(text)] if pattern else []


def forms_for(person, aliases, chapter):
    forms = {person, *(alias for alias, targets in aliases.items() if targets == (person,) and len(alias) >= 2)}
    title = TITLE_RANGE.get(person)
    if title and title[1] <= chapter <= title[2]:
        forms.add(title[0])
    return "|".join(map(re.escape, sorted(forms, key=lambda value: (-len(value), value))))


def answer_latest_office(person, corpus, aliases):
    """Find the last explicit appointment in the book's narrative order."""
    forms = {chapter: forms_for(person, aliases, chapter) for chapter in range(1, corpus.chapter_count + 1)}
    for passage in reversed(corpus.passages):
        pattern = (rf"(?:^|[，。；;])(?:遂|乃|即|朝廷)?(?:迁|加|拜|升|授|任命|封)"
                   rf"(?:{forms[passage.chapter]})为(?P<office>{OFFICE})(?=$|[，。；;」])")
        appointments = list(re.finditer(pattern, passage.searchable))
        if not appointments:
            continue
        appointment = appointments[-1]
        details = re.split(r"[。；;」]", passage.searchable[appointment.end():], maxsplit=1)[0]
        answer = f"按原文能查到的最后一处明确任命，{person}任{appointment['office']}"
        for clause in details.split("，")[1:]:
            detail = OFFICE_DETAIL.fullmatch(clause)
            if not detail:
                break
            answer += f"，{detail['duty']}" if detail["duty"] else f"，爵位为{detail['rank']}"
        return BookAnswer(f"{answer}。见《三国演义》第{passage.chapter}回。", (passage,))
    return None


def death_passages(person, corpus, aliases):
    """Find actual death narration. Past references have lower priority."""
    found = []
    patterns = {chapter: forms_for(person, aliases, chapter) for chapter in range(1, corpus.chapter_count + 1)}
    for passage in corpus.passages:
        text = passage.searchable
        forms = patterns[passage.chapter]
        if not re.search(forms, text):
            continue
        direct = re.search(
            rf"(?:{forms})(?P<event>(?:竟|遂|当即)?(?:驾崩|遇害|身故|病故|病重而死|缢死|绞死|"
            r"死[于於][^。；]{1,15}|连叫数声而亡)|(?:既|已)死)", text
        )
        if direct:
            event = direct["event"]
            score = 1 if event in ("既死", "已死", "身故") else 4
            found.append((score, passage, event))
        elif re.match(rf"(?:却说|且说)?(?:{forms})", text) and re.search(r"连叫数声而亡|吐血[^。；]{0,8}而死", text):
            event = re.search(r"连叫数声而亡|吐血[^。；]{0,8}而死", text)[0]
            found.append((5, passage, event))
        elif re.search(rf"[。；」]{re.escape(person[-1])}[^。；]{{0,14}}吐血[^。；]{{0,8}}而死", text):
            found.append((5, passage, "吐血而死"))
        elif re.search(rf"原来(?:{forms})[^。]{{0,12}}不合眼", text) and "以短刀刺入" in text and "而亡" in text:
            found.append((5, passage, "遇害"))
    return sorted(found, key=lambda item: (-item[0], item[1].chapter, item[1].first_line))


def death_context(passage, corpus, person, aliases):
    nearby = [item for item in corpus.passages if item.chapter == passage.chapter
              and passage.first_line - 12 <= item.last_line and item.first_line <= passage.last_line]
    forms = forms_for(person, aliases, passage.chapter)
    illness = next((item for item in nearby if re.search(
        rf"(?:{forms})(?:自|在[^，。]{{1,12}})?(?:染病|病重|病危|旧病复发)|某父[^。]{{0,14}}病重而死",
        item.searchable)), None)
    # An earlier illness can be stated well before the death in the same chapter.
    if illness is None and re.search(rf"(?:{forms})驾崩", passage.searchable):
        illness = next((item for item in corpus.passages if item.chapter == passage.chapter
                        and item.first_line < passage.first_line
                        and re.search(rf"(?:{forms})在[^，。]{{1,12}}染病", item.searchable)), None)
    return nearby, illness


def describe_death(person, passage, event, corpus, aliases):
    nearby, illness = death_context(passage, corpus, person, aliases)
    forms = forms_for(person, aliases, passage.chapter)
    if "缢死" in event:
        actor = re.search(rf"(?:^|[。；：。！」])([\u3400-\u9fff]{{1,8}})令将(?:{forms})缢死", passage.searchable)
        if actor:
            canonical = aliases.get(actor[1], ())
            if actor[1] == "操" and "曹操" in aliases:
                canonical = ("曹操",)
            if len(canonical) == 1:
                return f"{canonical[0]}下令将{person}缢死", (passage,), canonical[0]
        return f"{person}被缢死", (passage,), ""
    if "吐血" in event:
        return f"{person}吐血而死", (passage,), "illness" if "旧病复发" in passage.searchable else ""
    if re.search("死[于於]乱箭", event):
        return f"{person}死于乱箭", (passage,), ""
    if "以短刀刺入" in passage.searchable:
        return f"{person}在帐中遭短刀刺杀", (passage,), ""
    if illness:
        location = re.search(rf"(?:{forms})在([^，。]{{1,12}})染病", illness.searchable)
        where = f"在{location[1]}" if location else ""
        return f"{person}{where}染病，随后去世", tuple(dict.fromkeys((illness, passage))), "illness"
    return f"{person}去世", (passage,), ""


def age_number(text):
    digits = "零一二三四五六七八九"
    if text.isdigit():
        return int(text)
    if "十" not in text:
        return digits.index(text) if text in digits else None
    if not re.fullmatch(r"[一二三四五六七八九]?十[一二三四五六七八九]?", text):
        return None
    before, after = text.split("十", 1)
    return (digits.index(before) if before else 1) * 10 + (digits.index(after) if after else 0)


def answer_death(person, intent, corpus, aliases, actor=""):
    for score, passage, event in death_passages(person, corpus, aliases):
        chapter = f"第{passage.chapter}回"
        if actor or "怎么" in intent or "如何" in intent:
            summary, evidence, cause = describe_death(person, passage, event, corpus, aliases)
            if actor:
                if not cause or cause == actor:
                    continue
                return BookAnswer(f"没有。{chapter}记载，{summary}。", evidence, "correction")
            if summary != f"{person}去世":
                return BookAnswer(f"{chapter}记载，{summary}。", evidence)
        elif "地方" in intent or "哪里" in intent or "哪儿" in intent:
            location = re.search(r"此处地名([\u3400-\u9fff]{2,7})[，。]", passage.searchable)
            if score > 1 and location:
                return BookAnswer(f"{person}死在{location[1]}。见《三国演义》{chapter}。", (passage,))
        elif "岁" in intent:
            nearby, _ = death_context(passage, corpus, person, aliases)
            for item in reversed(nearby):
                age = re.search(r"(?:而逝|而亡|而死|驾崩)[。；，：](?:时?年止?|寿)([一二三四五六七八九十\d]{1,3})岁",
                                item.searchable)
                if age and (item == passage or event == "既死") and (years := age_number(age[1])):
                    return BookAnswer(f"{person}去世时{years}岁。见《三国演义》{chapter}。",
                                      tuple(dict.fromkeys((item, passage))))
        else:
            return BookAnswer(f"{person}去世的情节见《三国演义》{chapter}。", (passage,))
    return None


def retrieve_clues(text, people, corpus, aliases):
    terms = [word for word in TOPIC_TERMS if word in text]
    if not people or not terms:
        return ()
    patterns = {chapter: tuple(forms_for(person, aliases, chapter) for person in people)
                for chapter in range(1, corpus.chapter_count + 1)}
    scored = []
    for passage in corpus.passages:
        people_count = sum(bool(re.search(pattern, passage.searchable)) for pattern in patterns[passage.chapter])
        weight = sum(len(word) for word in terms if word in passage.searchable)
        if people_count and weight:
            scored.append((people_count * 3 + weight, passage))
    return tuple(item for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].first_line))[:3])


def indexed_answer(text, corpus, aliases):
    passages = corpus.index.search(text, aliases)
    if not passages:
        return None
    if SHIP_COUNT.search(text) and (answer := answer_ship_count(text, passages, aliases)):
        return answer
    return BookAnswer("资料不足以确认完整答案。下方列出相关原文供核对。", passages, "clue")


def answer_ship_count(text, passages, aliases):
    """Extract a number only when the original explicitly identifies both speakers."""
    mentions = names_in(text, aliases)
    if len(mentions) != 2:
        return None
    (first, borrower), (second, lender) = mentions
    if text[first.end():second.start()] not in ("向", "找", "请", "跟"):
        return None
    counts = {}
    for passage in passages:
        speaker = forms_for(borrower, aliases, passage.chapter)
        addressee = forms_for(lender, aliases, passage.chapter)
        pattern = (rf"(?:{speaker})曰[：:]「?(?:望|请)?(?:{addressee})借我"
                   r"([一二两三四五六七八九十百千万零\d]+)(?:只|隻|条|艘)船")
        for count in re.findall(pattern, passage.searchable):
            counts.setdefault(count, passage)
    if len(counts) != 1:
        return None
    count, passage = next(iter(counts.items()))
    return BookAnswer(f"第{passage.chapter}回写{borrower}向{lender}借{count}只船。", (passage,))


def answer_person_relationship(people, corpus, aliases):
    """Report reviewed relationships, or show passages mentioning both people."""
    events = []
    for participants, chapter, start, end, anchor, answer in BOOK_RELATION_EVENTS:
        if people.issubset(participants):
            source = next((passage for passage in corpus.passages if passage.chapter == chapter
                           and passage.first_line <= start <= end <= passage.last_line
                           and anchor in passage.text), None)
            if source:
                events.append((source, answer))
    if events:
        events.sort(key=lambda event: event[0].first_line)
        summary = "".join(f"第{source.chapter}回，{answer}" for source, answer in events)
        return BookAnswer(summary, tuple(dict.fromkeys(source for source, _ in events)))

    names = [re.compile(forms_for(person, aliases, 0)) for person in sorted(people)]
    passages = tuple(islice((passage for passage in corpus.passages
                             if all(name.search(passage.searchable) for name in names)), QA_MAX_RESULTS))
    if passages:
        return BookAnswer("资料不足以确认两人的关系。下方列出同时提及两人的原文。", passages, "clue")
    return None


def resolve_from_book(question, corpus, aliases, *, scoped=False):
    text = normalized(question).rstrip("？?。！!")
    if OUTSIDE.search(text):
        return None
    aliases = aliases or {}
    if CHAPTER.search(text):
        return indexed_answer(text, corpus, aliases)
    if scoped:
        return None
    mentions = names_in(text, aliases)
    if len(mentions) == 1:
        match, person = mentions[0]
        prefix, intent = text[:match.start()], text[match.end():]
        if prefix in ("", "请问") and LATEST_OFFICE_QUESTION.fullmatch(intent):
            if result := answer_latest_office(person, corpus, aliases):
                return result
        # Accept the same death question both before and after the name.
        if prefix in ("", "请问") and re.search(DEATH, intent) and DEATH_QUESTION.fullmatch(intent):
            if result := answer_death(person, intent, corpus, aliases):
                return result
        if re.fullmatch(r"在?哪(?:一)?回", prefix) and re.fullmatch(DEATH + "(?:的|呢)?", intent):
            if result := answer_death(person, prefix, corpus, aliases):
                return result
        if prefix == "" and intent in ("是谁", "指谁", "指的是谁"):
            for passage in corpus.passages:
                if re.search(rf"{re.escape(match[0])}(?:乃|正是|即|就是)[^，。]{{0,8}}{re.escape(person)}",
                             passage.searchable):
                    return BookAnswer(f"{match[0]}指{person}。见《三国演义》第{passage.chapter}回。", (passage,))
    elif len(mentions) == 2:
        (subject, actor), (target, person) = mentions
        if (actor != person and text[:subject.start()] in ("", "请问")
                and text[subject.end():target.start()] in ("和", "与", "跟", "同")
                and RELATION_QUESTION.fullmatch(text[target.end():])):
            return answer_person_relationship({actor, person}, corpus, aliases)
        if (actor != person and text[:subject.start()] in ("", "请问")
                and KILL.fullmatch(text[subject.end():target.start()]) and text[target.end():] in ("吗", "么")):
            if result := answer_death(person, "", corpus, aliases, actor=actor):
                return result
    clues = retrieve_clues(text, {name for _, name in mentions}, corpus, aliases)
    if clues:
        return BookAnswer("资料不足以确认。以下是可核对的原文线索。", clues, "clue")
    return indexed_answer(text, corpus, aliases)
