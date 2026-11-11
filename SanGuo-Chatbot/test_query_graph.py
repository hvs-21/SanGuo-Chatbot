from __future__ import annotations

import unittest

from config import SANGUO_DATA_DIR
from graph_repository import EntityNotFoundError, SanguoGraphRepository


class SanguoGraphQueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = SanguoGraphRepository(SANGUO_DATA_DIR)
        cls.repository.load_mappings()

    def test_liubei_name_maps_to_id_8(self) -> None:
        self.assertEqual(self.repository.get_entity_id("刘备"), 8)

    def test_liubei_query_contains_taoyuan_fact(self) -> None:
        matches = self.repository.query_by_entity_name("刘备")
        expected = "（关羽，联盟，刘备，桃园结义与涿郡初战阶段）"

        self.assertGreater(len(matches), 0)
        self.assertIn(expected, [match.as_chinese_text() for match in matches])
        self.assertTrue(
            all(
                match.source.head_id == 8 or match.source.tail_id == 8
                for match in matches
            )
        )

    def test_unknown_entity_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(EntityNotFoundError, "没有"):
            self.repository.query_by_entity_name("不存在的人物")


if __name__ == "__main__":
    unittest.main(verbosity=2)

