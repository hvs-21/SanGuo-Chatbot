"""Interpret eligible paraphrases without letting similarity invent query slots."""

import re

from config import (
    BERT_MAX_TOKENS, BERT_MIN_MARGIN, BERT_MIN_SIMILARITY, BERT_QUERY_TIMEOUT_SECONDS,
    BERT_RUNTIME_DIR, BERT_STARTUP_TIMEOUT_SECONDS, QA_DIRECTED_RELATIONS,
)
from fact_qa import FactQuery, VERB
from local_worker import LocalWorker, uses_wsl


PAIR = re.compile(r"甲[和与跟同]乙(.+)")
CONNECTION = re.compile(r"关系|关联|交情|交往|往来|来往|交集|联系|瓜葛|牵连")
PATH = re.compile(r"通过|经由|间接|中间|第三人")
SCOPE_CHANGE = re.compile(r"如果|假如|假设|倘若|是否|能否|为什么|为何|原因|有没有|好不好|如何评价|如何看待|未曾|没有|无关|并非|谁更|[，,；;：:、？?。！!]")
PAIR_FILTER = re.compile(r"在|之前|之后|后来|期间|早年|晚年|最早|最后|何时|什么时间|[年月日回章节\d]|兄弟|结义|联盟|亲属|父子|母子|夫妻|师徒|敌对|血缘|同僚|上下级|合作|因果")
WHEN = r"(?:(?:是)?(?:在|发生在|发生于|发生的|大约在)?)?(?:什么时候|什么时间|何时|哪一回|哪回|第几回|哪一章|第几章|哪个阶段)(?:的|呢)?"
ACTIVE_TIME = re.compile(rf"甲({VERB})(?:了)?乙{WHEN}")
PASSIVE_TIME = re.compile(rf"甲(?:被|遭|为)乙({VERB})(?:了)?{WHEN}")
ACTOR = re.compile(rf"甲(?:是)?(?:被|遭|为)(?:谁|何人|哪个人|哪位人物)({VERB})(?:的|过|呢)?")
OBJECT = re.compile(rf"甲(?:曾经|曾)?({VERB})(?:了|过)?(?:哪些人|什么人|哪几个人|哪几位)(?:呢)?")
PERSON = re.compile(r"(?:(?:想|我想)?(?:了解|看看|查看|听听|讲讲|说说|介绍(?:一下)?)甲的?(?:事迹|故事|经历|生平|相关记载)|甲(?:都)?(?:有过|有)(?:哪些|什么)(?:事迹|故事|经历|相关记载))(?:呢)?")


class BertWorker(LocalWorker):
    def __init__(self):
        prefix = "SANGUO_BERT"
        path = "../.runtime/bert/python" if uses_wsl(prefix) else str(BERT_RUNTIME_DIR / "python")
        super().__init__(prefix, "semantic_worker.py", BERT_STARTUP_TIMEOUT_SECONDS, BERT_QUERY_TIMEOUT_SECONDS,
                         {"PYTHONPATH": path, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                          "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})

    def similarities(self, question):
        return self.request(question)


class SemanticParser:
    def __init__(self, aliases, worker):
        self.worker = worker
        alternatives = sorted(aliases, key=lambda name: (-len(name), name))
        self.names = re.compile("|".join(map(re.escape, alternatives)))

    def parse(self, question):
        clean = re.sub(r"\s+", "", question).rstrip("？?。！!")
        clean = clean.removeprefix("请问")
        if not 0 < len(clean) <= BERT_MAX_TOKENS - 2 or SCOPE_CHANGE.search(clean) or "甲" in clean or "乙" in clean:
            return None
        spans = list(self.names.finditer(clean))
        if len(spans) not in (1, 2):
            return None
        names = [span[0] for span in spans]
        masked = clean
        for span, marker in reversed(list(zip(spans, "甲乙"))):
            masked = masked[:span.start()] + marker + masked[span.end():]
        candidates = eligible_queries(masked, names)
        if not candidates:
            return None
        scores = self.worker.similarities(masked)
        ranking = sorted(((scores[label], label) for label in (*candidates, "unsupported") if label in scores), reverse=True)
        if len(ranking) < 2:
            return None
        (best_score, label), (second_score, _) = ranking[:2]
        if best_score >= BERT_MIN_SIMILARITY and best_score - second_score >= BERT_MIN_MARGIN:
            return candidates.get(label)
        return None


def eligible_queries(masked, names):
    if len(names) == 2:
        first, second = names
        if match := PAIR.fullmatch(masked):
            suffix = match[1]
            if not CONNECTION.search(suffix) or PAIR_FILTER.search(suffix):
                return {}
            if PATH.search(suffix):
                return {"path": FactQuery(first, second, path_question=True)}
            return {"pair": FactQuery(first, second)}
        if match := ACTIVE_TIME.fullmatch(masked):
            return {"time": FactQuery(first, second, relations=QA_DIRECTED_RELATIONS[match[1]],
                                      direction="head", time_question=True)}
        if match := PASSIVE_TIME.fullmatch(masked):
            return {"time": FactQuery(first, second, relations=QA_DIRECTED_RELATIONS[match[1]],
                                      direction="tail", time_question=True)}
    else:
        first = names[0]
        if match := ACTOR.fullmatch(masked):
            return {"actor": FactQuery(first, relations=QA_DIRECTED_RELATIONS[match[1]], direction="tail")}
        if match := OBJECT.fullmatch(masked):
            return {"object": FactQuery(first, relations=QA_DIRECTED_RELATIONS[match[1]], direction="head")}
        if PERSON.fullmatch(masked):
            return {"person": FactQuery(first)}
    return {}
