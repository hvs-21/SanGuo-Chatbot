"""Check the sealed question set against the live browser API and save each verdict."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from acceptance_check import request
from config import BOOK_PATH, QA_FACTS_PATH, WEB_PORT, WORKSPACE_ROOT
from fact_qa import load_evidence_facts


MATERIALS = WORKSPACE_ROOT / "结题材料"
QUESTION_PATH = MATERIALS / "全文问答封存题.json"
BASELINE_PATH = MATERIALS / "全文问答开发前基线.json"
RESULT_PATH = MATERIALS / "全文问答验收结果.json"


def evidence_errors(case, sources, facts, book_lines):
    valid_facts = set()
    valid_spans = []
    for source in sources:
        if source["id"] in facts:
            fact = facts[source["id"]]
            if (source["chapter"], source["chunk"], source["quote"]) != (
                fact.chapter_label, fact.source_chunk, fact.source_text
            ):
                return ["图谱引用与事实记录不一致"]
            valid_facts.add(fact.fact_id)
        elif source["id"] == "Gutenberg 23950":
            chapter = re.fullmatch(r"第(\d+)回", source["chapter"])
            location = re.fullmatch(r"第(\d+)–(\d+)行", source["chunk"])
            if not chapter or not location:
                return ["引文缺少可核对的章节或行号"]
            start, end = map(int, location.groups())
            if not 1 <= start <= end <= len(book_lines) or source["quote"] != "".join(
                line.strip() for line in book_lines[start - 1:end]
            ):
                return ["引文与所标注的原文不一致"]
            valid_spans.append((int(chapter[1]), start, end))
    if ids := set(case.get("facts", ())):
        if not (ids <= valid_facts if case.get("all_facts") else ids & valid_facts):
            return ["缺少该题的已核对事实"]
    if spans := case.get("passages"):
        if not any(chapter == expected_chapter and start <= expected_end and expected_start <= end
                   for chapter, start, end in valid_spans
                   for expected_chapter, expected_start, expected_end in spans):
            return ["缺少该题的原文证据范围"]
    return []


def grade(case, response, facts, book_lines):
    sources = response["sources"]
    expected = {"answer": "direct"}.get(case["kind"], case["kind"])
    actual = response.get("evidence_status", "direct" if sources else "insufficient")
    problems = []
    if actual != expected:
        problems.append(f"预期 {expected}，实际 {actual}")
    missing = [term for term in case.get("terms", ()) if term not in response["answer"]]
    if missing:
        problems.append("结论缺少 " + "、".join(missing))
    if expected == "insufficient" and (sources or "不足" not in response["answer"]):
        problems.append("范围外问题应明确表示资料不足")
    if expected == "clue" and "不足" not in response["answer"]:
        problems.append("线索应明确标注结论无法确认")
    problems.extend(evidence_errors(case, sources, facts, book_lines))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    raw_questions = QUESTION_PATH.read_bytes()
    question_hash = sha256(raw_questions).hexdigest()
    if question_hash != QUESTION_PATH.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]:
        parser.error("封存题目摘要不符")
    cases = json.loads(raw_questions)
    if Counter(case["kind"] for case in cases) != {"answer": 20, "clue": 5, "insufficient": 3, "correction": 2}:
        parser.error("封存题数不符")
    if request(args.port, "/api/status")["qwen_enabled"]:
        parser.error("请在无密钥模式执行验收")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    book = BOOK_PATH.read_bytes()
    if baseline["question_sha256"] != question_hash or baseline["corpus_sha256"] != sha256(book).hexdigest():
        parser.error("当前题单或底本与旧版基线不一致")
    lines = book.decode("utf-8-sig").splitlines()
    facts = {fact.fact_id: fact for fact in load_evidence_facts(QA_FACTS_PATH)}
    previous = {row["id"]: row["response"] for row in baseline["results"]}
    records = []
    for case in cases:
        response = request(args.port, "/api/query", {"query": case["query"], "mode": "qa", "use_qwen": False})
        errors = grade(case, response, facts, lines)
        before = grade(case, previous[case["id"]], facts, lines)
        records.append({**case, "response": response, "passed": not errors, "failures": errors,
                        "baseline_passed": not before, "baseline_failures": before})
    scores = Counter(row["kind"] for row in records if row["passed"])
    gates_passed = scores["answer"] >= 16 and sum(scores[key] for key in ("clue", "insufficient", "correction")) == 10
    result = {"question_sha256": question_hash, "corpus_sha256": sha256(book).hexdigest(),
              "scores": dict(scores), "gates_passed": gates_passed, "results": records}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"可回答 {scores['answer']}/20；线索 {scores['clue']}/5；"
          f"范围外 {scores['insufficient']}/3；纠正 {scores['correction']}/2")
    print(f"逐题记录 {args.output}")
    if not gates_passed:
        parser.exit(1, "未达到验收门槛\n")


if __name__ == "__main__":
    main()
