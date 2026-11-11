"""Compare saved candidates with validation labels after inference has finished."""

import csv
import hashlib
import io


def attach_answers(data: dict, data_dir) -> None:
    """Attach all recorded answers; never filter or reorder model candidates."""
    raw = (data_dir / "valid.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != data["input_sha256"]["valid.txt"]:
        raise ValueError("Validation data changed since prediction export")
    mapping = (data_dir / "entity_mapping.csv").read_bytes()
    expected_mapping = data["input_sha256"].get("entity_mapping.csv")
    if expected_mapping and hashlib.sha256(mapping).hexdigest() != expected_mapping:
        raise ValueError("Entity mapping changed since prediction export")
    names = {int(row["model_entity_id"]): row["entity_name"]
             for row in csv.DictReader(io.StringIO(mapping.decode("utf-8-sig"), newline=""))}
    answers = {}
    for line in raw.decode("utf-8-sig").splitlines():
        head, relation, tail, time = map(int, line.split())
        answers.setdefault(f"{head}:{relation}:{time}", set()).add(tail)
    # Build all annotations before modifying the response, so errors cannot leave partial labels.
    annotations = {}
    for case in data["cases"]:
        annotations[case["id"]] = [
            {"entity_id": entity, "entity": names[entity]}
            for entity in sorted(answers[case["id"]])
        ]
    for case in data["cases"]:
        case["answers"] = annotations[case["id"]]


def summarize_predictions(data: dict) -> dict:
    """Report descriptive case-level coverage, not the experiment's filtered metrics."""
    result = {}
    for model in data["models"]:
        top_one, top_k, covered_answers, total_answers = 0, 0, 0, 0
        first_choices = {}
        for case in data["cases"]:
            answers = {answer["entity_id"] for answer in case["answers"]}
            candidates = case["predictions"][model]
            predicted = {item["entity_id"] for item in candidates}
            top_one += candidates[0]["entity_id"] in answers
            top_k += bool(answers & predicted)
            covered_answers += len(answers & predicted)
            total_answers += len(answers)
            name = candidates[0]["entity"]
            first_choices[name] = first_choices.get(name, 0) + 1
        result[model] = {"cases": len(data["cases"]), "top_one_cases": top_one,
                         "top_k_cases": top_k, "covered_answers": covered_answers,
                         "total_answers": total_answers, "first_choices": first_choices}
    return result
