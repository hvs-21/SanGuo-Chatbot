from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# 可通过环境变量覆盖数据目录，便于以后迁移项目。
DEFAULT_SANGUO_DATA_DIR = WORKSPACE_ROOT / "tirgn" / "TiRGN-main" / "data" / "SANGUO"
SANGUO_DATA_DIR = Path(
    os.environ.get("SANGUO_DATA_DIR", str(DEFAULT_SANGUO_DATA_DIR))
).expanduser()

TEXT_ENCODING = "utf-8-sig"

ENTITY_MAPPING_FILE = "entity_mapping.csv"
RELATION_MAPPING_FILE = "relation_mapping.csv"
TIME_MAPPING_FILE = "time_mapping.csv"

DATA_SPLITS = ("train", "valid", "test")

DEFAULT_QA_FACTS_PATH = (
    WORKSPACE_ROOT / "sanguo_data_handoff" / "quadruples_master_v17_question_review.csv"
)
QA_FACTS_PATH = Path(os.environ.get("SANGUO_FACTS_PATH", str(DEFAULT_QA_FACTS_PATH))).expanduser()
QA_MAX_RESULTS = 5
QA_EVENT_RELATIONS = {"结义": "结义兄弟", "结为兄弟": "结义兄弟"}
QA_ENTITIES_PATH = WORKSPACE_ROOT / "sanguo_data_handoff" / "entities_master_v10_question_review.csv"
COMMUNITY_INDEX_PATH = PROJECT_ROOT / "community_index.json"
COMMUNITY_RANDOM_SEED = 42
COMPLETION_RUN = WORKSPACE_ROOT / "sanguo_experiments/runs/completion_v2_3seeds_r2"
COMPLETION_RELEASE_SHA256 = "66db8c3d7f3b86fb8e2ff2d9a70bf12abe63c37ac73b09d2dc23c3a5a494a857"
COMPLETION_REVIEW_SHA256 = "3e8b61ddf5640c2091f43185cc91af89a3aa93a4931fbbda4002736e88ad2be6"
COMPLETION_SEED = 42
COMPLETION_TOP_K = 5
COMPLETION_THREADS = "2"
COMPLETION_STARTUP_SECONDS = 35
COMPLETION_QUERY_SECONDS = 4
QA_DIRECTED_RELATIONS = {
    "斩杀": ("斩杀",), "斩": ("斩杀",), "杀害": ("杀害",), "杀死": ("斩杀", "杀害"), "杀": ("斩杀", "杀害"),
    "击败": ("击败",), "追击": ("追击",), "追赶": ("追击",), "攻打": ("攻打",),
    "救援": ("救援",), "救": ("救援",), "投靠": ("投靠",), "擒获": ("擒获",),
    "到达": ("到达",), "驻守": ("驻守",), "任命": ("任命",), "任职": ("任职",),
    "担任": ("任职",), "受封": ("受封",), "辅佐": ("辅佐",), "推荐": ("推荐",),
    "劝谏": ("劝谏",), "护送": ("护送",), "废黜": ("废黜",), "拥立": ("拥立",),
    "对战": ("对战",), "下令处死": ("下令处死",), "派遣": ("派遣",), "释放": ("释放",),
    "劝降": ("劝降",), "托孤": ("托孤",), "屯田": ("屯田",),
}
NEO4J_TIMEOUT_SECONDS = 3
NEO4J_VERIFY_TIMEOUT_SECONDS = 12
NEO4J_MAX_RESPONSE_BYTES = 2_000_000

BERT_RUNTIME_DIR = WORKSPACE_ROOT / ".runtime" / "bert"
BERT_MODEL_DIR = BERT_RUNTIME_DIR / "model"
BERT_MODEL_ID = "BAAI/bge-small-zh-v1.5"
BERT_MODEL_REVISION = "8399f11f8da998fe932df2684586c92024219d05"
BERT_MODEL_FILES = {
    "config.json": "3853a7979202c348751b753e36f579c41d8da7d36af617d3d907e1fc9b441f2a",
    "model.safetensors": "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
    "tokenizer_config.json": "e6f3b96db926a37d4039995fbf5ad17de158dfb8f6343d607e4dbaad18d75f5a",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}
BERT_THREADS = 2
BERT_MAX_TOKENS = 96
BERT_STARTUP_TIMEOUT_SECONDS = 45
BERT_QUERY_TIMEOUT_SECONDS = 4
BERT_MIN_SIMILARITY = 0.74
BERT_MIN_MARGIN = 0.025
BERT_PROTOTYPES = {
    "pair": ("甲和乙是什么关系", "甲与乙有何交情", "甲和乙有过什么来往", "甲同乙有什么交集"),
    "path": ("甲与乙有什么间接关联", "甲和乙中间经过谁联系起来", "甲跟乙通过谁产生联系"),
    "person": ("介绍一下甲的事迹", "想看看甲的故事", "甲都有哪些事迹", "甲有哪些相关记载"),
    "time": ("甲做了这件事是什么时候", "甲对乙采取行动发生在哪一回", "乙被甲击败是在什么时候"),
    "actor": ("甲是被谁击败的", "对甲做这件事的人是谁", "甲遭谁攻打过"),
    "object": ("甲击败过哪些人", "甲对哪些人做了这件事", "甲曾经攻打过谁"),
    "unsupported": ("甲和乙谁更厉害", "甲和乙为什么有来往", "甲和乙是不是同一个人", "甲和乙有没有亲属关系",
                    "甲为什么这么做", "甲用过什么手机", "甲有什么性格特点", "甲一生有几个儿子"),
}

WEB_HOST = "127.0.0.1"
WEB_PORT = 8765
WEB_ASSET_DIR = PROJECT_ROOT / "dist"
WEB_MAX_QUERY_CHARS = 200
WEB_MAX_REQUEST_BYTES = 8192
WEB_MAX_SEARCH_TERMS = 8
BOOK_PATH = WORKSPACE_ROOT / "sanguo_experiments" / "gutenberg_23950.txt"
BOOK_INDEX_PATH = WORKSPACE_ROOT / "sanguo_experiments/data_revisions/v2_fulltext_review/chapter_index.json"
BOOK_PERSON_NAMES = {
    "吴懿": ("吳懿",), "鲁肃": ("子敬",), "张松": (),
    "刘备": ("刘玄德", "刘皇叔"), "关羽": ("关云长", "关公"), "张飞": ("张翼德", "张车骑"),
    "曹操": ("曹孟德", "曹公"), "诸葛亮": ("诸葛孔明", "孔明", "卧龙", "伏龙"),
    "周瑜": ("周公瑾", "公瑾"), "庞统": ("庞士元", "士元", "凤雏"),
    "赵云": ("赵子龙", "子龙"), "吕布": ("吕奉先", "奉先"), "孙策": ("孙伯符", "伯符", "小霸王"),
    "孙权": ("孙仲谋", "仲谋", "吴侯"), "袁绍": ("袁本初", "本初"),
    "刘禅": ("刘阿斗", "阿斗"), "司马懿": ("司马仲达", "仲达"), "马超": ("马孟起", "孟起"),
}
QA_REVIEWED_DEATHS = {
    "关羽": ("孙权", 77, 13671, 13672, "遂命推出。於是關公父子皆遇害"),
}
# Reviewed events with a chapter, line range and text anchor in the pinned edition.
BOOK_RELATION_EVENTS = (
    (("关羽", "张飞", "赵云", "马超", "黄忠"), 73, 12990, 12994,
     "封關羽、張飛、趙雲、馬超、黃忠為五虎大將軍",
     "两人都是刘备麾下的将领，也同列五虎大将军。"),
    (("曹操", "赵云"), 41, 7442, 7445, "真虎將也！吾當生致之",
     "长坂坡一战，两人分属敌对阵营。曹操见赵云勇猛，称他为“真虎将”，并下令活捉他。"),
    (("孙权", "关羽"), 73, 13044, 13047, "吾虎女安肯犬子乎",
     "孙权派诸葛瑾替儿子向关羽的女儿求亲，关羽拒绝。"),
    (("孙权", "关羽"), *QA_REVIEWED_DEATHS["关羽"][1:],
     "双方后来转为敌对，孙权下令处死关羽父子。"),
)
WEB_CORPUS_MAX_LINES = 12
PILOT_PREDICTIONS_PATH = WORKSPACE_ROOT / "sanguo_experiments/runs/pilot_seed42/validation_predictions.json"
FORMAL_PREDICTIONS_PATH = WORKSPACE_ROOT / "sanguo_experiments/runs/formal_v2_3seeds_r1/validation_predictions.json"
DEFAULT_PREDICTIONS_PATH = FORMAL_PREDICTIONS_PATH if FORMAL_PREDICTIONS_PATH.is_file() else PILOT_PREDICTIONS_PATH
PREDICTIONS_PATH = Path(os.environ.get("SANGUO_PREDICTIONS_PATH", str(DEFAULT_PREDICTIONS_PATH))).expanduser()

QWEN_DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_DEFAULT_MODEL = "qwen-plus"
QWEN_TIMEOUT_SECONDS = 12
QWEN_MAX_TOKENS = 512


@dataclass(frozen=True)
class QwenSettings:
    api_key: str = field(repr=False)
    endpoint: str
    model: str


def load_qwen_settings() -> QwenSettings | None:
    """Load optional Qwen credentials on the server, never from the browser."""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None

    endpoint = os.environ.get("SANGUO_QWEN_ENDPOINT", QWEN_DEFAULT_ENDPOINT).strip()
    url = urlsplit(endpoint)
    hostname = url.hostname or ""
    is_provider = hostname in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"} or (
        hostname.endswith(".maas.aliyuncs.com")
    )
    if (
        url.scheme != "https"
        or not is_provider
        or url.username is not None
        or url.port not in (None, 443)
        or url.path != "/compatible-mode/v1/chat/completions"
        or url.query
        or url.fragment
    ):
        raise ValueError("SANGUO_QWEN_ENDPOINT 必须是百炼的 HTTPS 对话接口。")

    model = os.environ.get("SANGUO_QWEN_MODEL", QWEN_DEFAULT_MODEL).strip()
    if not model:
        raise ValueError("SANGUO_QWEN_MODEL 不能为空。")
    return QwenSettings(api_key=api_key, endpoint=endpoint, model=model)

