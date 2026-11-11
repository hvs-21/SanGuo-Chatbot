"""Local HTTP demo for the existing facts and their source passages."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from contextlib import ExitStack
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import (
    BOOK_INDEX_PATH,
    BOOK_PATH,
    COMMUNITY_INDEX_PATH,
    PREDICTIONS_PATH,
    QA_ENTITIES_PATH,
    QA_FACTS_PATH,
    WEB_ASSET_DIR,
    WEB_HOST,
    WEB_MAX_QUERY_CHARS,
    WEB_MAX_REQUEST_BYTES,
    WEB_PORT,
    QwenSettings,
    load_qwen_settings,
)
from corpus_search import BookCorpus
from book_qa import BookAnswer, resolve_from_book
from community_search import CommunityIndex
from completion_repository import CompletionRepository, start_completion_worker
from evidence_search import search_passages
from fact_qa import EvidenceFact, answer_question, load_aliases, load_evidence_facts
from graph_repository import SanguoDataError
from query_graph import configure_console_encoding
from qwen_client import QwenError, answer_from_passages, rewrite_answer
from prediction_repository import load_predictions
from neo4j_repository import Neo4jError, Neo4jRepository
from semantic_parser import BertWorker, SemanticParser


ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def format_source(fact: EvidenceFact) -> dict[str, str]:
    return {
        "id": fact.fact_id,
        "fact": f"{fact.head} —{fact.relation}→ {fact.tail}",
        "time": fact.time_text,
        "chapter": fact.chapter_label,
        "chunk": fact.source_chunk,
        "quote": fact.source_text,
    }


def book_response(query: str, book: BookAnswer, use_qwen: bool, settings: QwenSettings | None,
                  notice: str, suggestions: dict | None) -> dict:
    answer, passages, status = book.answer, book.passages, book.kind
    method = "原文线索" if status == "clue" else "原文问答"
    if status == "clue" and use_qwen and settings is not None:
        try:
            if generated := answer_from_passages(query, passages, settings):
                answer, passages = generated
                method, status = "千问原文问答", "direct"
                notice = "千问根据下方原文归纳，请对照引文。"
        except QwenError:
            notice = "千问暂未返回可核对的回答，已显示本地原文线索。"
    return {"answer": answer, "method": method, "notice": notice, "retrieval_backend": "corpus",
            "query_parser": "corpus", "completion": suggestions, "evidence_status": status,
            "sources": [passage.citation() for passage in passages]}


def query_response(query: str, mode: str, use_qwen: bool, facts, settings, corpus: BookCorpus | None = None,
                   aliases=None, graph: Neo4jRepository | None = None, communities: CommunityIndex | None = None,
                   semantic: SemanticParser | None = None, completion: CompletionRepository | None = None) -> dict:
    """Resolve a demo request, keeping the underlying sources on every path."""
    notice = ""
    backend = "local"
    query_parser = ""
    suggestions = None
    if mode == "community":
        result = communities.search(query, aliases) if communities else None
        return {"answer": result.answer if result else "社区索引暂不可用。",
                "method": "社区检索", "notice": "按已收录的关系边划分社区，出处列于下方。" if result and result.facts else "",
                "sources": [format_source(fact) for fact in result.facts] if result else []}
    if mode == "corpus":
        passages = corpus.search(query) if corpus is not None else ()
        if corpus is None:
            answer = "全文索引暂不可用。可以切换到证据片段检索。"
        elif passages:
            answer = "找到以下《三国演义》原文，引用保留底本字形。"
        else:
            answer = "未找到同时包含这些关键词的原文段落。"
        return {"answer": answer, "method": "全文检索", "notice": "", "sources": [p.citation() for p in passages]}
    if mode == "search":
        matches = search_passages(query, facts)
        answer = "找到以下原文片段。" if matches else "现有片段中没有同时包含这些关键词的记录。"
        method = "原文检索"
    else:
        semantic_parse = semantic.parse if semantic else None
        try:
            result = answer_question(query, facts, aliases=aliases, lookup=graph.find if graph else None,
                                     semantic_parse=semantic_parse)
            if graph and result.facts and not result.evidence_origin:
                backend = "neo4j"
        except Neo4jError:
            result = answer_question(query, facts, aliases=aliases, semantic_parse=semantic_parse)
            notice = "图谱暂不可用，已从本地证据表查询。"
        if result.evidence_origin:
            backend = result.evidence_origin
        chapter_query = result.fact_query and result.fact_query.chapter
        if mode == "completion" or (completion and chapter_query and not result.facts):
            if completion:
                suggestions, supported = completion.suggest(result.fact_query)
            else:
                suggestions, supported = {"message": "历史补全未启用，仍可使用事实问答。"}, ()
            if supported and not result.facts:
                summary = "原文复核支持：" + "、".join(dict.fromkeys(
                    f"{fact.head} —{fact.relation}→ {fact.tail}" for fact in supported
                )) + "。"
                result = replace(result, facts=supported, answer=summary)
                backend = "review"
            elif supported and mode == "completion":
                result = replace(result, facts=tuple(dict.fromkeys((*result.facts, *supported))))
        scoped = result.fact_query and (result.fact_query.period or result.fact_query.chapter or result.fact_query.time_question)
        if (mode == "qa" and corpus and not result.facts and result.answer.startswith("现有资料不足")
                and (book := resolve_from_book(query, corpus, aliases, scoped=bool(scoped)))):
            return book_response(query, book, use_qwen, settings, notice, suggestions)
        matches, answer, method = result.facts, result.answer, "事实问答"
        if mode == "completion":
            method = "历史补全"
        query_parser = result.query_parser
        if use_qwen and matches:
            if settings is None:
                notice = "千问未配置，已显示本地回答。"
            else:
                try:
                    answer = rewrite_answer(result, settings)
                    method = "千问整理"
                    notice = "模型整理的表述请与下方来源核对。"
                except QwenError:
                    notice = "千问暂不可用，已显示本地回答。"

    return {
        "answer": answer,
        "method": method,
        "notice": notice,
        "retrieval_backend": backend,
        "query_parser": query_parser,
        "completion": suggestions,
        "evidence_status": "direct" if matches else "insufficient",
        "sources": [format_source(fact) for fact in matches],
    }


def validate_query(payload: object) -> tuple[str, str, bool]:
    if not isinstance(payload, dict):
        raise ValueError("请求必须是 JSON 对象。")
    query = payload.get("query")
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= WEB_MAX_QUERY_CHARS:
        raise ValueError(f"请输入 1～{WEB_MAX_QUERY_CHARS} 字的问题或关键词。")
    mode, use_qwen = payload.get("mode", "qa"), payload.get("use_qwen", False)
    if mode not in ("qa", "search", "corpus", "community", "completion") or not isinstance(use_qwen, bool):
        raise ValueError("请求模式不正确。")
    return query.strip(), mode, use_qwen


class DemoServer(ThreadingHTTPServer):
    def __init__(self, facts: list[EvidenceFact], settings: QwenSettings | None, port: int, corpus: BookCorpus | None = None,
                 aliases=None, graph: Neo4jRepository | None = None, communities: CommunityIndex | None = None,
                 semantic: SemanticParser | None = None, completion: CompletionRepository | None = None):
        self.facts = tuple(facts)
        self.settings = settings
        self.corpus = corpus
        self.aliases = aliases
        self.graph = graph
        self.communities = communities
        self.semantic = semantic
        self.completion = completion
        super().__init__((WEB_HOST, port), DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def send_body(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_body(body, "application/json; charset=utf-8", status)

    def is_local_request(self) -> bool:
        hosts = {f"{WEB_HOST}:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in hosts and (origin is None or origin == f"http://{host}")

    def do_GET(self) -> None:
        if not self.is_local_request():
            self.send_json({"error": "此服务只接受本地请求。"}, 403)
            return

        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            self.send_json({"fact_count": len(self.server.facts), "qwen_enabled": self.server.settings is not None,
                            "book_chapters": self.server.corpus.chapter_count if self.server.corpus else 0,
                            "graph_enabled": self.server.graph is not None,
                            "bert_enabled": bool(self.server.semantic and self.server.semantic.worker.available),
                            "completion_enabled": bool(self.server.completion and self.server.completion.worker.available),
                            "community_enabled": self.server.communities is not None})
            return
        if path == "/api/predictions":
            self.send_json(load_predictions(PREDICTIONS_PATH))
            return
        asset = ASSETS.get(path)
        if asset is None:
            self.send_json({"error": "页面不存在。"}, 404)
            return
        name, content_type = asset
        try:
            content = (WEB_ASSET_DIR / name).read_bytes()
        except OSError:
            self.send_json({"error": "无法加载页面文件。"}, 500)
            return
        self.send_body(content, content_type)

    def read_query(self) -> tuple[str, str, bool]:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("请发送 JSON 请求。")
        length = int(self.headers.get("Content-Length", "0"))
        if not 1 <= length <= WEB_MAX_REQUEST_BYTES:
            raise ValueError("请求正文为空或过长。")
        self.connection.settimeout(5)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求正文不是有效的 UTF-8 JSON。") from exc
        return validate_query(payload)

    def do_POST(self) -> None:
        if not self.is_local_request():
            self.send_json({"error": "此服务只接受本地请求。"}, 403)
            return
        if self.path != "/api/query":
            self.send_json({"error": "接口不存在。"}, 404)
            return
        try:
            query, mode, use_qwen = self.read_query()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
            return
        except TimeoutError:
            self.send_json({"error": "读取请求超时。"}, 408)
            return
        self.send_json(query_response(query, mode, use_qwen, self.server.facts, self.server.settings,
                                      self.server.corpus, self.server.aliases, self.server.graph,
                                      self.server.communities, self.server.semantic, self.server.completion))


def main() -> int:
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="在本机打开三国问答演示。")
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--bert", action="store_true", help="启用可选的本地 BERT 语义解析")
    parser.add_argument("--completion", action="store_true", help="启用可选的本地 RE-GCN 历史补全")
    parser.add_argument("--open-browser", action="store_true", help="启动后打开本机演示网页")
    args = parser.parse_args()
    try:
        settings = load_qwen_settings()
    except ValueError:
        print("千问配置不正确，继续使用本地问答。", file=sys.stderr)
        settings = None

    try:
        facts = load_evidence_facts(QA_FACTS_PATH)
        try:
            corpus = BookCorpus.load(BOOK_PATH, BOOK_INDEX_PATH)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"无法读取全文索引：{exc}", file=sys.stderr)
            corpus = None
        try:
            aliases = load_aliases(QA_ENTITIES_PATH)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"无法读取别名表：{exc}", file=sys.stderr)
            aliases = None
        graph = None
        try:
            graph = Neo4jRepository.from_environment(facts)
            if graph:
                graph.verify()
        except (Neo4jError, ValueError, KeyError, TypeError) as exc:
            print(f"Neo4j 查询暂不可用：{exc}", file=sys.stderr)
            graph = None
        try:
            communities = CommunityIndex.load(COMMUNITY_INDEX_PATH, facts)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"无法读取社区索引：{exc}", file=sys.stderr)
            communities = None
        with ExitStack() as stack:
            semantic = None
            if args.bert and aliases:
                try:
                    worker = BertWorker()
                    stack.callback(worker.close)
                    semantic = SemanticParser(aliases, worker)
                except (OSError, ValueError) as exc:
                    print(f"BERT 未能启动，继续使用规则解析：{exc}", file=sys.stderr)
            completion = None
            if args.completion:
                try:
                    worker = start_completion_worker()
                    stack.callback(worker.close)
                    completion = CompletionRepository(facts, worker)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f"历史补全未能启动，继续使用本地问答：{exc}", file=sys.stderr)
            server = stack.enter_context(DemoServer(
                facts, settings, args.port, corpus, aliases, graph, communities, semantic, completion
            ))
            url = f"http://{WEB_HOST}:{server.server_port}/"
            print(f"Local: {url}", flush=True)
            if args.open_browser:
                try:
                    opened = webbrowser.open(url)
                except (OSError, webbrowser.Error):
                    opened = False
                if not opened:
                    print("请将上面的地址复制到浏览器。", file=sys.stderr)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (SanguoDataError, OSError, OverflowError) as exc:
        print(f"无法启动演示：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
