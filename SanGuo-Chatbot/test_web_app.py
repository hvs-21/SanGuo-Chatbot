"""Exercise the browser API locally without a real Qwen key or network call."""

import json
import unittest
from contextlib import closing
from http.client import HTTPConnection
from threading import Thread
from unittest.mock import patch
from pathlib import Path
import tempfile

from config import BOOK_INDEX_PATH, BOOK_PATH, QA_ENTITIES_PATH, QA_FACTS_PATH, QWEN_DEFAULT_ENDPOINT, WEB_HOST, QwenSettings
from corpus_search import BookCorpus
from fact_qa import load_aliases, load_evidence_facts
from neo4j_repository import Neo4jError
from qwen_client import QwenError
from web_app import DemoServer


KNOWN_QUESTION = "刘备和关羽在桃园结义阶段是什么关系？"
TEST_QWEN = QwenSettings("test-only-key", QWEN_DEFAULT_ENDPOINT, "test-model")


class WebDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        facts = load_evidence_facts(QA_FACTS_PATH)
        cls.server = DemoServer(facts, None, port=0, corpus=BookCorpus.load(BOOK_PATH, BOOK_INDEX_PATH))
        cls.addClassCleanup(cls.server.server_close)
        thread = Thread(target=cls.server.serve_forever, daemon=True)
        thread.start()
        cls.addClassCleanup(thread.join, 5)
        cls.addClassCleanup(cls.server.shutdown)

    def request(self, method, path, payload=None, origin=None):
        with closing(HTTPConnection(WEB_HOST, self.server.server_port, timeout=5)) as client:
            body = None
            headers = {}
            if payload is not None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"
            if origin:
                headers["Origin"] = origin
            client.request(method, path, body, headers)
            response = client.getresponse()
            return response.status, response.read().decode("utf-8")

    def test_page_and_assets_are_served(self):
        for path, marker in (("/", "循原文，问三国"), ("/app.js", "/api/query"), ("/styles.css", ".workbench")):
            with self.subTest(path=path):
                status, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn(marker, body)

    def test_fact_answer_has_a_citation(self):
        status, body = self.request("POST", "/api/query", {"query": KNOWN_QUESTION})
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("结义兄弟", data["answer"])
        self.assertEqual(data["sources"][0]["chunk"], "chapter_001_chunk_05")

    def test_graph_error_falls_back_to_cited_local_answer(self):
        from unittest.mock import Mock
        graph = Mock()
        graph.find.side_effect = Neo4jError("unavailable")
        with patch.object(self.server, "graph", graph):
            status, body = self.request("POST", "/api/query", {"query": KNOWN_QUESTION})
        result = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(result["retrieval_backend"], "local")
        self.assertIn("图谱暂不可用", result["notice"])
        self.assertEqual(result["sources"][0]["id"], "F00015")

    def test_passage_search_requires_each_keyword(self):
        status, body = self.request("POST", "/api/query", {"query": "刘备 既结为兄弟", "mode": "search"})
        self.assertEqual(status, 200)
        sources = json.loads(body)["sources"]
        self.assertTrue(sources)
        self.assertTrue(all("刘备" in item["quote"] and "既结为兄弟" in item["quote"] for item in sources))

    def test_fulltext_search_cites_source_lines_and_respects_chapter(self):
        status, body = self.request("POST", "/api/query", {"query": "第98回 王雙 魏延", "mode": "corpus"})
        result = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(result["method"], "全文检索")
        self.assertTrue(result["sources"])
        for source in result["sources"]:
            self.assertEqual(source["chapter"], "第98回")
            self.assertEqual(source["id"], "Gutenberg 23950")
            self.assertIn("王雙", source["quote"])
            self.assertIn("魏延", source["quote"])
        status, body = self.request("POST", "/api/query", {"query": "第1回 王雙 魏延", "mode": "corpus"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["sources"], [])

    def test_latest_office_uses_appointment_evidence(self):
        cases = (
            ("张飞最后的官职是什么", "第81回", ("车骑将军", "领司隶校尉", "兼阆中牧", "爵位为西乡侯")),
            ("请问翼德生前最后担任了什么职务？", "第81回", ("车骑将军",)),
            ("皇甫嵩最后的官职是什么", "第2回", ("车骑将军", "领冀州牧")),
            ("张飞最后的官职薪水是多少", None, ("资料不足",)),
        )
        with patch.object(self.server, "aliases", load_aliases(QA_ENTITIES_PATH)):
            for question, chapter, expected in cases:
                with self.subTest(question=question):
                    status, body = self.request("POST", "/api/query", {"query": question})
                    result = json.loads(body)
                    self.assertEqual(status, 200)
                    for phrase in expected:
                        self.assertIn(phrase, result["answer"])
                    self.assertEqual([source["chapter"] for source in result["sources"]], [chapter] if chapter else [])

    def test_fact_question_automatically_searches_indexed_passages(self):
        aliases = load_aliases(QA_ENTITIES_PATH)
        with patch.object(self.server, "aliases", aliases):
            status, body = self.request("POST", "/api/query", {"query": "草船借箭时，诸葛亮请鲁肃借多少条船？"})
            self.assertEqual(status, 200)
            result = json.loads(body)
            self.assertEqual(result["evidence_status"], "direct")
            self.assertIn("二十只船", result["answer"])
            self.assertIn("二十隻船", result["sources"][0]["quote"])
            status, body = self.request("POST", "/api/query", {"query": "第95回，空城计时孔明在城楼上做什么？"})
            self.assertEqual(status, 200)
            result = json.loads(body)
            self.assertEqual(result["evidence_status"], "clue")
            self.assertEqual({source["chapter"] for source in result["sources"]}, {"第95回"})
            self.assertTrue(any("焚香操琴" in source["quote"] for source in result["sources"]))

    def test_fact_question_accepts_chapter_after_person(self):
        status, body = self.request("POST", "/api/query", {"query": "蔡邕在第4回担任什么官职？"})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertIn("侍中", result["answer"])
        self.assertEqual([source["id"] for source in result["sources"]], ["F00132"])

    def test_person_relationships_fall_back_to_reviewed_book_events(self):
        cases = (
            ("赵云和关羽是什么关系", "第73回", "五虎大将军", "封關羽、張飛、趙雲"),
            ("曹操和赵云是什么关系", "第41回", "分属敌对阵营", "只要捉活的"),
            ("子龙和孟德有什么关系？", "第41回", "分属敌对阵营", "只要捉活的"),
            ("马超和黄忠是什么关系？", "第73回", "五虎大将军", "馬超、黃忠"),
        )
        with patch.object(self.server, "aliases", load_aliases(QA_ENTITIES_PATH)):
            for question, chapter, answer, quote in cases:
                with self.subTest(question=question):
                    status, body = self.request("POST", "/api/query", {"query": question})
                    result = json.loads(body)
                    self.assertEqual(status, 200)
                    self.assertEqual(result["evidence_status"], "direct")
                    self.assertIn(answer, result["answer"])
                    self.assertEqual([source["chapter"] for source in result["sources"]], [chapter])
                    self.assertIn(quote, result["sources"][0]["quote"])

    def test_person_relationship_can_cite_a_change_over_time(self):
        with patch.object(self.server, "aliases", load_aliases(QA_ENTITIES_PATH)):
            status, body = self.request("POST", "/api/query", {"query": "关羽和孙权是什么关系"})
        result = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(result["evidence_status"], "direct")
        self.assertIn("求亲，关羽拒绝", result["answer"])
        self.assertIn("孙权下令处死关羽父子", result["answer"])
        self.assertEqual([source["chapter"] for source in result["sources"]], ["第73回", "第77回"])

    def test_no_evidence_does_not_call_qwen(self):
        with patch.object(self.server, "settings", TEST_QWEN), patch("web_app.rewrite_answer") as rewrite:
            status, body = self.request("POST", "/api/query", {"query": "曹操用什么手机？", "use_qwen": True})
        self.assertEqual(status, 200)
        self.assertIn("资料不足", json.loads(body)["answer"])
        self.assertEqual(json.loads(body)["sources"], [])
        rewrite.assert_not_called()

    def test_completion_candidates_without_evidence_are_excluded_from_generation(self):
        from unittest.mock import Mock
        from completion_repository import CompletionRepository

        worker = Mock()
        repository = CompletionRepository(self.server.facts, worker)
        worker.request.return_value = {"ranking": [repository.ids[name] for name in
                                                  ("孙坚", "夏侯惇", "董卓", "曹操", "曹仁")]}
        with (patch.object(self.server, "completion", repository), patch.object(self.server, "settings", TEST_QWEN),
              patch("web_app.rewrite_answer", return_value="原文记载徐荣追击曹操。") as rewrite):
            status, body = self.request("POST", "/api/query", {"query": "第6回，徐荣追击了谁？", "use_qwen": True})
        self.assertEqual(status, 200)
        self.assertEqual([fact.fact_id for fact in rewrite.call_args.args[0].facts], ["v2-B005"])
        self.assertEqual(len(json.loads(body)["completion"]["candidates"]), 5)

    def test_qwen_failure_keeps_the_local_answer(self):
        with patch.object(self.server, "settings", TEST_QWEN), patch("web_app.rewrite_answer", side_effect=QwenError):
            status, body = self.request("POST", "/api/query", {"query": KNOWN_QUESTION, "use_qwen": True})
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["method"], "事实问答")
        self.assertIn("已显示本地回答", data["notice"])
        self.assertEqual(data["sources"][0]["id"], "F00015")

    def test_qwen_adapter_sends_evidence_and_reads_cited_response(self):
        content = json.dumps({"answer": "刘备与关羽结为兄弟。", "fact_ids": ["F00015"]})
        response = {"choices": [{"message": {"content": content}}]}
        with patch.object(self.server, "settings", TEST_QWEN), patch("qwen_client.build_opener") as opener:
            upstream = opener.return_value.open
            upstream.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode("utf-8")
            status, body = self.request("POST", "/api/query", {"query": KNOWN_QUESTION, "use_qwen": True})
        data = json.loads(body)
        request_body = json.loads(upstream.call_args.args[0].data)
        evidence = json.loads(request_body["messages"][1]["content"])["evidence"]
        self.assertEqual([item["id"] for item in evidence], ["F00015"])
        self.assertEqual(status, 200)
        self.assertEqual(data["method"], "千问整理")
        self.assertEqual(data["answer"], "刘备与关羽结为兄弟。")

    def test_qwen_can_answer_from_retrieved_passages_with_verifiable_quotes(self):
        question = "第95回，空城计时孔明在城楼上做什么？"
        quote = "憑欄而坐，焚香操琴"
        with (patch.object(self.server, "aliases", load_aliases(QA_ENTITIES_PATH)),
              patch.object(self.server, "settings", TEST_QWEN), patch("qwen_client.build_opener") as opener):
            _, body = self.request("POST", "/api/query", {"query": question})
            original = json.loads(body)
            opener.assert_not_called()
            passage_id = next(f"P{index}" for index, source in enumerate(original["sources"], 1)
                              if quote in source["quote"])
            upstream = opener.return_value.open
            cases = (
                ({"answer": "孔明在城楼上焚香弹琴。", "evidence": [{"id": passage_id, "quote": quote}]}, "direct"),
                ({"answer": "孔明在城楼上吹笛。", "evidence": [{"id": passage_id, "quote": "孔明吹笛"}]}, "clue"),
                ({"answer": None, "evidence": []}, "clue"),
            )
            for payload, expected in cases:
                with self.subTest(expected=expected, payload=payload):
                    response = {"choices": [{"message": {"content": json.dumps(payload)}}]}
                    upstream.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
                    status, body = self.request("POST", "/api/query", {"query": question, "use_qwen": True})
                    result = json.loads(body)
                    self.assertEqual(status, 200)
                    self.assertEqual(result["evidence_status"], expected)
                    if expected == "direct":
                        self.assertEqual(result["method"], "千问原文问答")
                        self.assertEqual(result["answer"], payload["answer"])
                        self.assertIn(quote, result["sources"][0]["quote"])
                    else:
                        self.assertEqual(result["answer"], original["answer"])
                        self.assertEqual(result["sources"], original["sources"])

            sent = json.loads(upstream.call_args.args[0].data)
            evidence = json.loads(sent["messages"][1]["content"])["evidence"]
            self.assertTrue(any(item["id"] == passage_id and quote in item["quote"] for item in evidence))

    def test_cross_origin_requests_are_rejected(self):
        status, _ = self.request("POST", "/api/query", {"query": KNOWN_QUESTION}, origin="https://example.invalid")
        self.assertEqual(status, 403)

    def test_prediction_endpoint_has_both_real_models(self):
        with patch("web_app.rewrite_answer") as rewrite:
            status, body = self.request("GET", "/api/predictions")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(data["available"])
        self.assertEqual(data["split"], "valid")
        self.assertEqual(set(data["models"]), {"tirgn", "regcn"})
        self.assertGreater(len(data["cases"]), 0)
        rewrite.assert_not_called()

    def test_missing_predictions_do_not_break_fact_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("web_app.PREDICTIONS_PATH", Path(directory) / "missing.json"):
                status, body = self.request("GET", "/api/predictions")
                self.assertEqual(status, 200)
                self.assertFalse(json.loads(body)["available"])
                status, body = self.request("POST", "/api/query", {"query": KNOWN_QUESTION})
                self.assertEqual(status, 200)
                self.assertIn("结义兄弟", json.loads(body)["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
