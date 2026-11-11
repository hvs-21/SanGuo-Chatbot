"""Serve live RE-GCN tail ranks from observed facts and a locked weight."""

from contextlib import redirect_stdout
import json
import sys

from completion_repository import locked_protocol, verified_bytes
from config import COMPLETION_RUN, COMPLETION_SEED, COMPLETION_TOP_K


def load_model():
    protocol, digest = locked_protocol()
    for name, sha in protocol["frozen_files"].items():
        if name.startswith(("code/", "models/regcn/")) or name == "observed.txt":
            verified_bytes(COMPLETION_RUN / name, sha)
    sys.path.insert(0, str(COMPLETION_RUN / "code"))
    from completion_worker import CompletionContext, NeuralCompletion, read_rows

    context = CompletionContext(read_rows(COMPLETION_RUN, "observed"), protocol["entities"], protocol["relations"])
    neural = NeuralCompletion(COMPLETION_RUN / "models/regcn", "regcn", context, COMPLETION_SEED)
    raw = verified_bytes(COMPLETION_RUN / f"regcn/{COMPLETION_SEED}/best.pt", digest)
    from io import BytesIO
    saved = neural.torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if saved["seed"] != COMPLETION_SEED:
        raise ValueError("Checkpoint seed differs")
    neural.model.load_state_dict(saved["state_dict"])
    neural.model.eval()
    return neural


def predict(neural, payload):
    import numpy as np

    if not isinstance(payload, list) or len(payload) != 3 or any(type(i) is not int for i in payload):
        return {}
    head, relation, timestamp = payload
    if timestamp not in neural.context.by_time:
        return {"outside_history": True}
    if not (0 <= head < neural.context.entities and 0 <= relation < neural.context.relations):
        return {}
    # The decoder only needs head and relation for the first tail score row.
    batch = np.array([[head, relation, 0, timestamp]], dtype=np.int64)
    order, _ = neural.ranking(timestamp, batch)
    return {"ranking": order[0, :COMPLETION_TOP_K].tolist()}


def main():
    with redirect_stdout(sys.stderr):
        neural = load_model()
    print('{"ready": true}', flush=True)
    for line in sys.stdin:
        payload = json.loads(line)
        with redirect_stdout(sys.stderr):
            result = predict(neural, payload)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
