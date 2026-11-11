"""Read a version-checked Leiden index and return the original cited facts."""

import json
from pathlib import Path

from config import QA_MAX_RESULTS
from fact_qa import Answer, evidence_fingerprint


class CommunityIndex:
    def __init__(self, groups, facts):
        self.facts = {fact.fact_id: fact for fact in facts}
        self.groups = {}
        for group in groups:
            members = set(group["members"])
            if not members or members & self.groups.keys() or not set(group["core"]) <= members:
                raise ValueError("社区成员重复或核心实体无效")
            expected = [fact.fact_id for fact in facts if fact.head in members and fact.tail in members]
            if group["fact_ids"] != expected:
                raise ValueError("社区的证据清单与图谱事实不一致")
            for entity in members:
                self.groups[entity] = group
        graph_entities = {name for fact in facts for name in (fact.head, fact.tail)}
        if set(self.groups) != graph_entities:
            raise ValueError("社区索引没有覆盖全部图谱实体")

    @classmethod
    def load(cls, path: Path, facts):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["algorithm"] != "Leiden" or data["evidence_sha256"] != evidence_fingerprint(facts):
            raise ValueError("社区索引和当前事实版本不一致")
        return cls(data["communities"], facts)

    def search(self, entity: str, aliases=None) -> Answer:
        names = aliases.get(entity, (entity,)) if aliases else (entity,)
        if len(names) > 1:
            return Answer(entity, f"“{entity}”可能指{'、'.join(names)}，请使用完整名称。")
        group = self.groups.get(names[0])
        if not group:
            return Answer(entity, "现有资料不足，未找到该实体的社区。")
        candidates = [self.facts[key] for key in group["fact_ids"]]
        # Rank facts about the queried entity before other community evidence.
        candidates.sort(key=lambda fact: names[0] not in (fact.head, fact.tail))
        answer = (f"{names[0]}所在的社区有 {len(group['members'])} 个实体。"
                  f"连接较多的实体包括{'、'.join(group['core'])}。"
                  f"社区内有 {len(candidates)} 条带出处的关系记录。")
        return Answer(entity, answer, tuple(candidates[:QA_MAX_RESULTS]))
