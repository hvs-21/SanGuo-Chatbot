"""Optional local Neo4j evidence store using parameterized Cypher and Query API."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from http.client import HTTPException
import json
import os
import re
import subprocess
import sys
from urllib.parse import urlsplit

from config import NEO4J_MAX_RESPONSE_BYTES, NEO4J_TIMEOUT_SECONDS, NEO4J_VERIFY_TIMEOUT_SECONDS, QA_FACTS_PATH, QA_MAX_RESULTS
from fact_qa import EvidenceFact, FactQuery, evidence_fingerprint, load_evidence_facts
from graph_repository import SanguoDataError
from neo4j_transport import post_http, post_wsl


class Neo4jError(RuntimeError):
    """Neo4j is unavailable or returned an invalid response."""


ENTITY_CONSTRAINT = """
CREATE CONSTRAINT sanguo_entity_catalog_name IF NOT EXISTS
FOR (node:SanguoEntity) REQUIRE (node.catalog, node.name) IS UNIQUE
"""
IMPORT_FACTS = """
UNWIND $rows AS row
MERGE (head:SanguoEntity {catalog: $catalog, name: row.head})
MERGE (tail:SanguoEntity {catalog: $catalog, name: row.tail})
MERGE (head)-[fact:SANGUO_FACT {catalog: $catalog, fact_id: row.fact_id}]->(tail)
SET fact.relation = row.relation, fact.time_id = row.time_id, fact.time_text = row.time_text,
    fact.source_chapter = row.source_chapter, fact.source_chunk = row.source_chunk,
    fact.source_text = row.source_text, fact.ordinal = row.ordinal
RETURN count(fact) AS imported
"""
FACT_PATTERN = """
MATCH (head:SanguoEntity {catalog: $catalog})
      -[fact:SANGUO_FACT {catalog: $catalog}]->
      (tail:SanguoEntity {catalog: $catalog})
"""
FACT_PROJECTION = """
RETURN fact.fact_id AS fact_id, head.name AS head, fact.relation AS relation, tail.name AS tail,
       fact.time_id AS time_id, fact.time_text AS time_text, fact.source_chapter AS source_chapter,
       fact.source_chunk AS source_chunk, fact.source_text AS source_text
ORDER BY fact.ordinal
"""
FIND_FACTS = FACT_PATTERN + """
WHERE (($direction = 'head' AND head.name = $person AND ($other = '' OR tail.name = $other))
    OR ($direction = 'tail' AND tail.name = $person AND ($other = '' OR head.name = $other))
    OR ($direction = 'either' AND (
        (head.name = $person AND ($other = '' OR tail.name = $other))
        OR (tail.name = $person AND ($other = '' OR head.name = $other)))))
  AND (size($relations) = 0 OR fact.relation IN $relations)
  AND fact.time_text CONTAINS $period
  AND ($chapter = '' OR fact.source_chapter = $chapter)
""" + FACT_PROJECTION + " LIMIT $limit"
FIND_PATH = """
MATCH (start:SanguoEntity {catalog: $catalog, name: $person})
      -[first:SANGUO_FACT {catalog: $catalog}]-
      (via:SanguoEntity {catalog: $catalog})
      -[second:SANGUO_FACT {catalog: $catalog}]-
      (finish:SanguoEntity {catalog: $catalog, name: $other})
WHERE start <> via AND via <> finish AND start <> finish
  AND first.time_text CONTAINS $period AND second.time_text CONTAINS $period
RETURN [first.fact_id, second.fact_id] AS evidence
ORDER BY first.ordinal, second.ordinal LIMIT 1
"""


class Neo4jRepository:
    def __init__(self, endpoint: str, database: str, user: str, password: str, facts, wsl_distro=""):
        url = urlsplit(endpoint)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"} or
                url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("SANGUO_NEO4J_URL 必须是本机 Neo4j HTTP 地址")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]{0,62}", database):
            raise ValueError("Neo4j 数据库名格式错误")
        self.host, self.port = url.hostname, url.port or 7474
        if wsl_distro and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", wsl_distro):
            raise ValueError("WSL 发行版名称格式错误")
        self.wsl_distro = wsl_distro if sys.platform == "win32" else ""
        self.path = f"/db/{database}/query/v2"
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if password:
            token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
            self.headers["Authorization"] = f"Basic {token}"
        self.facts = tuple(facts)
        self.by_id = {fact.fact_id: fact for fact in self.facts}
        self.rows = [dict(asdict(fact), ordinal=index) for index, fact in enumerate(self.facts)]
        if not self.rows or len({row["fact_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("图谱输入缺少事实或存在重复事实编号")
        self.catalog = "sanguo-" + evidence_fingerprint(self.facts)

    @classmethod
    def from_environment(cls, facts):
        endpoint = os.environ.get("SANGUO_NEO4J_URL", "").strip()
        if not endpoint:
            return None
        return cls(endpoint, os.environ.get("SANGUO_NEO4J_DATABASE", "neo4j"),
                   os.environ.get("SANGUO_NEO4J_USER", "neo4j"),
                   os.environ.get("SANGUO_NEO4J_PASSWORD", ""), facts,
                   os.environ.get("SANGUO_NEO4J_WSL_DISTRO", ""))

    def query(self, statement: str, parameters: dict, *, timeout=NEO4J_TIMEOUT_SECONDS) -> list[dict]:
        payload = json.dumps({"statement": statement, "parameters": parameters}, ensure_ascii=False).encode("utf-8")
        try:
            if self.wsl_distro:
                status, raw = post_wsl(self.wsl_distro, self.port, self.path, self.headers, payload, timeout)
            else:
                status, raw = post_http(self.host, self.port, self.path, self.headers, payload, timeout)
            if status != 202:
                raise Neo4jError(f"Neo4j HTTP {status}")
            if len(raw) > NEO4J_MAX_RESPONSE_BYTES:
                raise Neo4jError("Neo4j 响应过大")
            parsed = json.loads(raw)
            if parsed.get("errors"):
                raise Neo4jError(f"Cypher 查询失败：{parsed['errors'][0].get('code', 'unknown')}")
            data = parsed.get("data", {})
            fields = data.get("fields", [])
            return [dict(zip(fields, values, strict=True)) for values in data.get("values", [])]
        except (OSError, HTTPException, ValueError, TypeError, AttributeError, subprocess.SubprocessError) as exc:
            raise Neo4jError("无法读取 Neo4j 查询响应") from exc

    def count(self) -> int:
        rows = self.query(FACT_PATTERN + " RETURN count(fact) AS count", {"catalog": self.catalog})
        if len(rows) != 1 or type(rows[0].get("count")) is not int:
            raise Neo4jError("Neo4j 未返回有效的事实计数")
        return rows[0]["count"]

    @staticmethod
    def decode_facts(rows) -> tuple[EvidenceFact, ...]:
        try:
            return tuple(EvidenceFact(**row) for row in rows)
        except TypeError as exc:
            raise Neo4jError("Neo4j 缺少证据字段") from exc

    def find(self, query: FactQuery) -> tuple[EvidenceFact, ...]:
        period = query.period.removesuffix("阶段") or query.period
        if query.path_question:
            rows = self.query(FIND_PATH, {"catalog": self.catalog, "person": query.person,
                                         "other": query.other_person, "period": period})
            if not rows:
                return ()
            try:
                first, second = rows[0]["evidence"]
                return self.by_id[first], self.by_id[second]
            except (KeyError, TypeError, ValueError) as exc:
                raise Neo4jError("Neo4j 路径的事实引用无效") from exc
        rows = self.query(FIND_FACTS, {"catalog": self.catalog, "person": query.person,
                                      "other": query.other_person, "direction": query.direction,
                                      "relations": query.relations, "period": period, "chapter": query.chapter,
                                      "limit": QA_MAX_RESULTS})
        return self.decode_facts(rows)

    def verify(self):
        rows = self.query(FACT_PATTERN + FACT_PROJECTION, {"catalog": self.catalog},
                          timeout=NEO4J_VERIFY_TIMEOUT_SECONDS)
        if self.decode_facts(rows) != self.facts:
            raise Neo4jError("Neo4j 数据和本地证据版本不一致")

    def import_facts(self):
        self.query(ENTITY_CONSTRAINT, {})
        result = self.query(IMPORT_FACTS, {"catalog": self.catalog, "rows": self.rows})
        if result != [{"imported": len(self.rows)}]:
            raise Neo4jError("Neo4j 导入条数不符")
        self.verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("import", "verify"))
    args = parser.parse_args()
    try:
        graph = Neo4jRepository.from_environment(load_evidence_facts(QA_FACTS_PATH))
        if graph is None:
            raise ValueError("请设置 SANGUO_NEO4J_URL")
        if args.action == "import":
            graph.import_facts()
        else:
            graph.verify()
        print(f"PASS: Neo4j {graph.count()} cited facts ({graph.catalog})")
    except (Neo4jError, SanguoDataError, OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
