"""Fetch and verify the fixed local BERT assets from the publisher on ModelScope."""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import BERT_MODEL_DIR, BERT_MODEL_FILES, BERT_MODEL_ID, BERT_MODEL_REVISION


REPO = f"https://modelscope.cn/api/v1/models/{BERT_MODEL_ID}/repo"
WEIGHT_SIZE = 95_827_648
PART_SIZE = 4 * 1024 * 1024


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, target) -> None:
    url = f"{REPO}?{urlencode({'Revision': BERT_MODEL_REVISION, 'FilePath': name})}"
    if name != "model.safetensors":
        with urlopen(url, timeout=20) as response:
            target.write(response.read(PART_SIZE + 1))
        return
    for start in range(0, WEIGHT_SIZE, PART_SIZE):
        end = min(start + PART_SIZE, WEIGHT_SIZE) - 1
        request = Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=30) as response:
            part = response.read(PART_SIZE + 1)
            if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{WEIGHT_SIZE}"
                    or len(part) != end - start + 1):
                raise ValueError(f"下载范围校验失败（{start}–{end}）")
        target.write(part)


def prepare() -> None:
    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in BERT_MODEL_FILES.items():
        path = BERT_MODEL_DIR / name
        if path.is_file() and checksum(path) == expected:
            continue
        with NamedTemporaryFile(dir=BERT_MODEL_DIR, delete=False) as target:
            temporary = Path(target.name)
        try:
            with temporary.open("wb") as target:
                download(name, target)
            if checksum(temporary) != expected:
                raise ValueError(f"{name} 的 SHA256 校验失败")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"已校验 {name}", flush=True)


if __name__ == "__main__":
    try:
        prepare()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"无法准备本地 BERT 文件：{exc}") from exc
