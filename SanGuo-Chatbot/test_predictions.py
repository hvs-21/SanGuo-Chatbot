"""Independent standard-library checks for actual exported predictions and failures."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from config import FORMAL_PREDICTIONS_PATH, PILOT_PREDICTIONS_PATH as PREDICTIONS_PATH, SANGUO_DATA_DIR
from prediction_repository import load_predictions, validate_predictions
from prediction_review import attach_answers, summarize_predictions
from unittest.mock import patch


class PredictionTests(unittest.TestCase):
    @unittest.skipUnless(FORMAL_PREDICTIONS_PATH.is_file(), "formal results have not been generated")
    def test_formal_cases_use_the_matching_revision_and_checkpoints(self):
        run = FORMAL_PREDICTIONS_PATH.parent
        data = load_predictions(FORMAL_PREDICTIONS_PATH)
        self.assertTrue(data["review_available"])
        self.assertEqual(data["data_version"], "v2_fulltext_review")
        targets = [tuple(map(int, line.split())) for line in (run / "prediction_data/valid.txt").read_text().splitlines()]
        cases = {(case["head_id"], case["relation_id"], case["time_id"]): case for case in data["cases"]}
        self.assertEqual(set(cases), {(h, r, t) for h, r, _, t in targets})
        for h, r, answer, t in targets:
            self.assertIn(answer, [item["entity_id"] for item in cases[h, r, t]["answers"]])
        for name, metadata in data["models"].items():
            path = run / f"seed{metadata['seed']}" / f"{name}.pt"
            if path.is_file():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["checkpoint_sha256"])

    def test_answers_preserve_rankings_and_include_multiple_labels(self):
        data = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
        before = copy.deepcopy(data)
        attach_answers(data, SANGUO_DATA_DIR)
        self.assertTrue(any(len(case["answers"]) > 1 for case in data["cases"]))
        for old, case in zip(before["cases"], data["cases"]):
            self.assertEqual(old["predictions"], case["predictions"])
        summary = summarize_predictions(data)
        for stats in summary.values():
            # The validation file has 31 rows but includes a duplicate quadruple.
            rows = (SANGUO_DATA_DIR / "valid.txt").read_text().splitlines()
            self.assertEqual(stats["total_answers"], len({tuple(map(int, row.split())) for row in rows}))
            self.assertLessEqual(stats["top_one_cases"], stats["top_k_cases"])

    def test_changed_labels_are_not_shown_as_the_export_answers(self):
        data = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
        data["input_sha256"]["valid.txt"] = "wrong-hash"
        with self.assertRaises(ValueError):
            attach_answers(data, SANGUO_DATA_DIR)
        self.assertTrue(all("answers" not in case for case in data["cases"]))
        with patch("prediction_repository.attach_answers", side_effect=ValueError):
            result = load_predictions(PREDICTIONS_PATH)
        self.assertTrue(result["available"])
        self.assertFalse(result["review_available"])

    def test_real_export_matches_validation_and_weights(self):
        data = load_predictions(PREDICTIONS_PATH)
        self.assertTrue(data["available"], data.get("message"))
        valid = [list(map(int, row.split())) for row in (SANGUO_DATA_DIR / "valid.txt").read_text().splitlines()]
        expected = {f"{h}:{r}:{t}" for h, r, _, t in valid}
        self.assertEqual({case["id"] for case in data["cases"]}, expected)
        for filename, digest in data["input_sha256"].items():
            self.assertNotEqual(filename, "test.txt")
            self.assertEqual(hashlib.sha256((SANGUO_DATA_DIR / filename).read_bytes()).hexdigest(), digest)
        for model, metadata in data["models"].items():
            checkpoint = PREDICTIONS_PATH.parent / f"{model}.pt"
            if checkpoint.is_file():
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), metadata["checkpoint_sha256"])
        with (SANGUO_DATA_DIR / "entity_mapping.csv").open(encoding="utf-8-sig", newline="") as stream:
            entities = {int(row["model_entity_id"]): row["entity_name"] for row in csv.DictReader(stream)}
        for case in data["cases"]:
            for candidates in case["predictions"].values():
                for candidate in candidates:
                    self.assertEqual(candidate["entity"], entities[candidate["entity_id"]])

    def test_missing_or_corrupt_export_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            self.assertFalse(load_predictions(path)["available"])
            for content in ("not JSON", "null", "[]", '{}', '{"schema_version": 9}'):
                path.write_text(content, encoding="utf-8")
                self.assertFalse(load_predictions(path)["available"])

    def test_rejects_test_split_and_invalid_rankings(self):
        original = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
        altered = copy.deepcopy(original)
        altered["split"] = "test"
        with self.assertRaises(ValueError):
            validate_predictions(altered)
        for key, value in (("rank", 9), ("score", float("nan")), ("entity", "")):
            altered = copy.deepcopy(original)
            altered["cases"][0]["predictions"]["tirgn"][0][key] = value
            with self.assertRaises(ValueError):
                validate_predictions(altered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
