"""Bridge a chapter-scoped question to RE-GCN and verify its ranked suggestions."""

import csv
from hashlib import sha256
import io
import json
import re
from collections import defaultdict

from config import (
    BOOK_INDEX_PATH, BOOK_PATH, COMPLETION_QUERY_SECONDS, COMPLETION_RELEASE_SHA256, COMPLETION_REVIEW_SHA256,
    COMPLETION_RUN, COMPLETION_SEED, COMPLETION_STARTUP_SECONDS, COMPLETION_THREADS, COMPLETION_TOP_K,
)
from fact_qa import EvidenceFact
from local_worker import LocalWorker


def verified_bytes(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"文件摘要不匹配：{path.name}")
    return raw


def locked_protocol():
    release = json.loads(verified_bytes(COMPLETION_RUN / "test_release.json", COMPLETION_RELEASE_SHA256))
    protocol = json.loads(verified_bytes(COMPLETION_RUN / "protocol.json", release["protocol_sha256"]))
    return protocol, release["checkpoints"]["regcn"][str(COMPLETION_SEED)]


def read_mapping(protocol, name):
    raw = verified_bytes(COMPLETION_RUN / name, protocol["frozen_files"][name])
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def load_review(protocol):
    manifest = json.loads(verified_bytes(BOOK_INDEX_PATH.with_name("manifest.json"), COMPLETION_REVIEW_SHA256))
    lines = verified_bytes(BOOK_PATH, manifest["corpus_sha256"]).decode("utf-8-sig").splitlines()
    chapters = json.loads(BOOK_INDEX_PATH.read_text(encoding="utf-8"))["chapters"]
    times = {row["time_id"]: row for row in read_mapping(protocol, "time_mapping.csv")}
    for item in manifest["additions"]:
        if item["decision"] != "include_in_candidate":
            continue
        evidence = item["evidence"]
        chapter, start, end = (evidence[key] for key in ("chapter", "start_line", "end_line"))
        bounds = chapters[str(chapter)]
        quote = "\n".join(line.strip() for line in lines[start - 1:end])
        if not bounds["start_line"] < start <= end <= bounds["end_line"] or evidence["anchor"] not in quote:
            raise ValueError("原文复核定位不一致")
        yield EvidenceFact(f"v2-{item['id']}", item["head"], item["relation"], item["tail"],
                           times[item["time_id"]]["time_text"], f"chapter_{chapter:03d}",
                           f"第{start}–{end}行", quote, item["time_id"])


class CompletionRepository:
    def __init__(self, facts, worker):
        protocol, _ = locked_protocol()
        self.names = {int(row["model_entity_id"]): row["entity_name"]
                      for row in read_mapping(protocol, "entity_mapping.csv")}
        self.ids = {name: identifier for identifier, name in self.names.items()}
        self.relations = {row["model_relation"]: int(row["model_relation_id"])
                          for row in read_mapping(protocol, "relation_mapping.csv")}
        relation_map = BOOK_INDEX_PATH.with_name("original_to_model_relation.csv")
        digest = json.loads(verified_bytes(BOOK_INDEX_PATH.with_name("manifest.json"), COMPLETION_REVIEW_SHA256))
        raw = verified_bytes(relation_map, digest["output_sha256"][relation_map.name])
        self.original_relations = {row["original_relation"]: row["model_relation"]
                                   for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))}
        self.by_chapter = defaultdict(list)
        for row in read_mapping(protocol, "time_mapping.csv"):
            if match := re.match(r"chapter_\d{3}", row["source_scope"]):
                self.by_chapter[match[0]].append(row)
        self.evidence = defaultdict(list)
        for fact in (*facts, *load_review(protocol)):
            self.evidence[fact.head, fact.tail, fact.time_id, fact.source_chapter].append(fact)
        self.worker = worker

    def suggest(self, query):
        if not query or not query.chapter or query.direction != "head" or query.other_person:
            return {"message": "请指定回数、人物和动作，例如“第6回，徐荣追击了谁？”"}, ()
        times = self.by_chapter[query.chapter]
        if len(times) != 1:
            return {"message": "该回无法唯一对应一个模型时间片。"}, ()
        stage = times[0]
        timestamp = int(stage["model_time_id"])
        labels = {self.original_relations.get(relation, relation) for relation in query.relations}
        if query.person not in self.ids or len(labels) != 1 or not labels <= self.relations.keys():
            return {"message": "模型未收录该人物，或查询动作无法唯一映射。"}, ()
        label = next(iter(labels))
        response = self.worker.request([self.ids[query.person], self.relations[label], timestamp])
        if response.get("outside_history"):
            return {"message": "该回超出历史补全的观测时间范围。"}, ()
        ranking = response.get("ranking")
        if (not isinstance(ranking, list) or len(ranking) != COMPLETION_TOP_K
                or any(type(value) is not int or value not in self.names for value in ranking)
                or len(set(ranking)) != len(ranking)):
            return {"message": "历史补全暂不可用，仍可查看事实与原文。"}, ()
        candidates, matches = [], []
        for index, identifier in enumerate(ranking, 1):
            entity = self.names[identifier]
            support = [fact for fact in self.evidence.get((query.person, entity, stage["time_id"], query.chapter), ())
                       if fact.relation in query.relations]
            candidates.append({"rank": index, "entity": entity, "evidence_ids": [fact.fact_id for fact in support]})
            matches.extend(support)
        return {"model": "RE-GCN", "seed": COMPLETION_SEED, "head": query.person, "model_relation": label,
                "time": stage["time_text"], "candidates": candidates}, tuple(matches)


def start_completion_worker():
    return LocalWorker("SANGUO_COMPLETION", "completion_live_worker.py",
                       COMPLETION_STARTUP_SECONDS, COMPLETION_QUERY_SECONDS,
                       {"DGLBACKEND": "pytorch", "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": COMPLETION_THREADS,
                        "MKL_NUM_THREADS": COMPLETION_THREADS, "OPENBLAS_NUM_THREADS": COMPLETION_THREADS})
