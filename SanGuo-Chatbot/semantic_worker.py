"""Score masked Chinese questions against intent examples using local BERT weights."""

import json
import sys

from config import BERT_MAX_TOKENS, BERT_MODEL_DIR, BERT_MODEL_FILES, BERT_PROTOTYPES, BERT_THREADS
from prepare_bert_model import checksum


class IntentEncoder:
    def __init__(self):
        for name, expected in BERT_MODEL_FILES.items():
            if checksum(BERT_MODEL_DIR / name) != expected:
                raise ValueError(f"{name} 校验失败")

        import torch
        from transformers import BertModel, BertTokenizerFast

        torch.set_num_threads(BERT_THREADS)
        self.torch = torch
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
        self.model = BertModel.from_pretrained(BERT_MODEL_DIR, local_files_only=True, use_safetensors=True).eval()
        self.labels = [label for label, phrases in BERT_PROTOTYPES.items() for _ in phrases]
        self.examples = self.encode([phrase for phrases in BERT_PROTOTYPES.values() for phrase in phrases])

    def encode(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=BERT_MAX_TOKENS, return_tensors="pt")
        with self.torch.inference_mode():
            cls_vectors = self.model(**batch).last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(cls_vectors, dim=1)

    def similarities(self, question):
        vector = self.encode([question])[0]
        scores = {}
        for label, score in zip(self.labels, (self.examples @ vector).tolist()):
            scores[label] = max(scores.get(label, -1.0), score)
        return scores


def send(payload):
    print(json.dumps(payload, allow_nan=False), flush=True)


def main():
    encoder = IntentEncoder()
    send({"ready": True})
    for line in sys.stdin:
        question = json.loads(line)
        if not isinstance(question, str) or not 0 < len(question) <= BERT_MAX_TOKENS - 2:
            send({})
            continue
        send(encoder.similarities(question))


if __name__ == "__main__":
    main()
