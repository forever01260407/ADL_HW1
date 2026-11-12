from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common import load_test_records


class DataJoiningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.test_records = [
            {"id": "second", "language": "en", "utterance": "What is the price?"},
            {"id": "first", "language": "zh", "utterance": "Please explain the service."},
        ]
        # The context order intentionally differs from the test order.
        self.contexts = {
            "first": [{"role": "assistant", "content": "First dialogue."}],
            "second": [{"role": "user", "content": "Second dialogue."}],
        }

    def load(self, contexts: dict, test_records: list | None = None) -> list:
        context_path = self.directory / "context.json"
        test_path = self.directory / "test.json"
        context_path.write_text(json.dumps(contexts), encoding="utf-8")
        test_path.write_text(
            json.dumps(self.test_records if test_records is None else test_records),
            encoding="utf-8",
        )
        return load_test_records(context_path, test_path)

    def test_exact_map_joins_by_id_and_preserves_test_order(self) -> None:
        joined = self.load(self.contexts)
        self.assertEqual([record["id"] for record in joined], ["second", "first"])
        self.assertEqual(joined[0]["context"], self.contexts["second"])
        self.assertEqual(joined[1]["context"], self.contexts["first"])

    def test_subset_ignores_extra_context_ids(self) -> None:
        contexts = {**self.contexts, "private-example": []}
        joined = self.load(contexts, self.test_records[:1])
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["id"], "second")
        self.assertEqual(joined[0]["context"], self.contexts["second"])

    def test_missing_context_is_still_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing context for test id: second"):
            self.load({"first": self.contexts["first"]})

    def test_empty_dialogue_context_is_valid(self) -> None:
        joined = self.load({**self.contexts, "second": []})
        self.assertEqual(joined[0]["context"], [])

    def test_duplicate_test_ids_are_still_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate test id: second"):
            self.load(self.contexts, [self.test_records[0], self.test_records[0]])


if __name__ == "__main__":
    unittest.main()
