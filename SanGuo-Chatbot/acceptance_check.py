"""Exercise the started local service with the project's representative questions."""

import argparse
from contextlib import closing
from http.client import HTTPConnection
import json

from config import PROJECT_ROOT, WEB_HOST, WEB_PORT
from fact_qa import parse_question


def request(port, path, payload=None):
    with closing(HTTPConnection(WEB_HOST, port, timeout=10)) as client:
        if payload is None:
            client.request("GET", path)
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            client.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = client.getresponse()
        if response.status != 200:
            raise ValueError(f"{path} returned HTTP {response.status}")
        return json.load(response)


def verify_questions(port, backend):
    cases = json.loads((PROJECT_ROOT / "acceptance_cases.json").read_text(encoding="utf-8"))
    failures = []
    for case in cases:
        result = request(port, "/api/query", {"mode": case["mode"], "query": case["query"], "use_qwen": False})
        sources = result["sources"]
        expected = case["source"]
        cited = [source for source in sources if source["id"] == expected]
        if case["answer"] not in result["answer"]:
            failures.append(f"{case['query']}: answer differed")
        if expected and not any(source["chapter"] and source["chunk"] and source["quote"] for source in cited):
            failures.append(f"{case['query']}: expected citation {expected}")
        if expected is None and sources:
            failures.append(f"{case['query']}: unexpected evidence")
        if case.get("chapter") and any(source["chapter"] != case["chapter"] for source in sources):
            failures.append(f"{case['query']}: mismatched chapter")
        if case["mode"] == "qa" and expected and result["retrieval_backend"] != backend:
            failures.append(f"{case['query']}: wrong evidence backend")
    if failures:
        raise ValueError("\n".join(failures))
    return len(cases)


def verify_predictions(port):
    data = request(port, "/api/predictions")
    if not data["available"] or not data.get("review_available") or data.get("data_version") != "v2_fulltext_review":
        raise ValueError("Missing v2 predictions or corresponding validation answers")
    outcomes = set()
    for case in data["cases"]:
        answers = {answer["entity_id"] for answer in case["answers"]}
        outcomes.add(any(candidate["entity_id"] in answers for ranking in case["predictions"].values() for candidate in ranking))
    if outcomes != {True, False}:
        raise ValueError("Expected both a hit and a miss in the validation demo")
    return len(data["cases"])


def verify_bert_questions(port, backend):
    cases = json.loads((PROJECT_ROOT / "bert_acceptance_cases.json").read_text(encoding="utf-8"))
    failures = []
    for case in cases:
        query = case["query"]
        if parse_question(query) is not None:
            failures.append(f"{query}: also recognized by the rule parser")
        result = request(port, "/api/query", {"mode": "qa", "query": query})
        sources = result["sources"]
        if (result["query_parser"] != case["parser"] or case["answer"] not in result["answer"]
                or [source["id"] for source in sources] != case["sources"]
                or any(not (source["chapter"] and source["chunk"] and source["quote"]) for source in sources)
                or (sources and result["retrieval_backend"] != backend)):
            failures.append(f"{query}: unexpected response {result['query_parser']}, {result['answer']}")
    if failures:
        raise ValueError("\n".join(failures))
    return len(cases)


def verify_completion(port, backend):
    def ask(query, mode="qa"):
        return request(port, "/api/query", {"mode": mode, "query": query})

    positive = ask("第6回，徐荣追击了谁？")
    ranks = positive["completion"]["candidates"]
    if ([row["entity"] for row in ranks] != ["孙坚", "夏侯惇", "董卓", "曹操", "曹仁"]
            or ranks[3]["evidence_ids"] != ["v2-B005"] or any(row["evidence_ids"] for row in ranks if row["rank"] != 4)
            or positive["retrieval_backend"] != "review" or "曹操" not in positive["answer"]
            or len(positive["sources"]) != 1 or "卻是徐榮從上流渡河來追" not in positive["sources"][0]["quote"]
            or positive["sources"][0]["chapter"] != "第6回" or "994" not in positive["sources"][0]["chunk"]):
        raise ValueError("Expected one directly supported candidate in the live ranking")
    unknown = ask("第5回，张飞斩杀了谁？", "completion")
    if ("资料不足" not in unknown["answer"] or unknown["sources"] or len(unknown["completion"]["candidates"]) != 5
            or any(row["evidence_ids"] for row in unknown["completion"]["candidates"])):
        raise ValueError("Unsupported suggestions were not kept separate from facts")
    known = ask("第5回，云长斩杀了谁？")
    if (known["completion"] is not None or known["retrieval_backend"] != backend
            or [source["id"] for source in known["sources"]] != ["F00167"]):
        raise ValueError("Existing chapter-scoped facts should bypass completion")
    unsupported = ask("第120回，关羽斩杀了谁？", "completion")
    if unsupported["sources"] or "超出" not in unsupported["completion"]["message"]:
        raise ValueError("Expected the held-out period to remain outside completion")
    ambiguous = ask("第1回，曹操杀死了谁？", "completion")
    if "无法唯一" not in ambiguous["completion"]["message"] or "candidates" in ambiguous["completion"]:
        raise ValueError("Expected ambiguity when one chapter contains several model periods")
    return 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--backend", choices=("local", "neo4j"), default="local")
    parser.add_argument("--bert", action="store_true", help="also verify paraphrases using the local BERT process")
    parser.add_argument("--completion", action="store_true", help="also verify live RE-GCN and its evidence checks")
    args = parser.parse_args()
    try:
        status = request(args.port, "/api/status")
        if (status["book_chapters"] != 120 or not status["community_enabled"]
                or status["graph_enabled"] != (args.backend == "neo4j")):
            raise ValueError("Expected the full book, community index and requested evidence backend")
        count = verify_questions(args.port, args.backend)
        predictions = verify_predictions(args.port)
        semantic = ""
        if args.bert:
            if not status["bert_enabled"]:
                raise ValueError("Expected the local BERT process to be available")
            semantic = f", {verify_bert_questions(args.port, args.backend)} BERT checks"
        if args.completion:
            if not status["completion_enabled"]:
                raise ValueError("Expected the local RE-GCN process to be available")
            semantic += f", {verify_completion(args.port, args.backend)} completion checks"
        print(f"PASS: {count} representative queries, {predictions} v2 prediction cases{semantic}, backend={args.backend}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Acceptance failed: {exc}\n")


if __name__ == "__main__":
    main()
