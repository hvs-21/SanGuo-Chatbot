"""Run the local evidence QA checks with: python test_fact_qa.py."""

import unittest

from config import QA_ENTITIES_PATH, QA_FACTS_PATH
from fact_qa import answer_question, load_aliases, load_evidence_facts
from qa_pipeline import format_answer


class FactQuestionAnsweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = load_evidence_facts(QA_FACTS_PATH)
        cls.aliases = load_aliases(QA_ENTITIES_PATH)

    def test_relationship_answer_links_to_real_evidence(self):
        result = answer_question("刘备和关羽在桃园结义阶段是什么关系？", self.facts)
        fact = result.facts[0]
        self.assertEqual((fact.head, fact.relation, fact.tail), ("关羽", "结义兄弟", "刘备"))
        self.assertEqual(fact.fact_id, "F00015")
        output = format_answer(result)
        self.assertIn("既结为兄弟", output)
        self.assertIn("《三国演义》第1回（chapter_001_chunk_05）", output)

    def test_event_time_preserves_the_data_granularity(self):
        result = answer_question("刘备和关羽什么时候结义？", self.facts)
        self.assertIn("桃园结义与涿郡初战阶段", result.answer)
        self.assertEqual(result.facts[0].source_chapter, "chapter_001")

    def test_confirmation_uses_cited_death_and_distinguishes_an_unknown_event(self):
        def answer(question):
            return answer_question(question, self.facts, aliases=self.aliases)

        result = answer("曹操杀了关羽吗")
        self.assertTrue(result.answer.startswith("没有。"))
        self.assertIn("孙权下令处死关羽", result.answer)
        self.assertEqual(result.facts[0].chapter_label, "第77回")
        self.assertIn("遂命推出", result.facts[0].source_text)
        self.assertIn("孙权下令处死关羽", answer("谁杀了云长？").answer)
        self.assertTrue(answer("关羽杀了华雄吗").answer.startswith("是。"))
        self.assertIn("资料不足", answer("曹操攻打过孙悟空吗").answer)
        self.assertIn("资料不足", answer("曹操没有杀了关羽吗").answer)

    def test_person_lookup_can_filter_by_time(self):
        result = answer_question("查询刘备在桃园结义阶段的事迹", self.facts)
        self.assertTrue(result.facts)
        for fact in result.facts:
            self.assertIn("刘备", (fact.head, fact.tail))
            self.assertIn("桃园结义", fact.time_text)

    def test_insufficient_evidence_does_not_return_unrelated_facts(self):
        questions = (
            "曹操用什么手机？",
            "刘备和孙悟空是什么关系？",
            "刘备和关羽在赤壁之战阶段是什么关系？",
        )
        for question in questions:
            with self.subTest(question=question):
                result = answer_question(question, self.facts)
                self.assertIn("现有资料不足", result.answer)
                self.assertEqual(result.facts, ())
                self.assertNotIn("证据：", format_answer(result))

    def test_unambiguous_alias_keeps_event_direction(self):
        result = answer_question("云长何时斩杀华雄？", self.facts, aliases=self.aliases)
        self.assertEqual([fact.fact_id for fact in result.facts], ["F00167"])
        reverse = answer_question("华雄何时斩杀云长？", self.facts, aliases=self.aliases)
        self.assertEqual(reverse.facts, ())

    def test_ambiguous_alias_requires_a_full_name(self):
        result = answer_question("查询将军", self.facts, aliases={"将军": ("刘备", "曹操")})
        self.assertIn("请使用完整名称", result.answer)
        self.assertEqual(result.facts, ())

    def test_two_hop_answer_cites_each_edge_with_its_time(self):
        result = answer_question("张飞和华雄有什么间接关联？", self.facts)
        self.assertIn("经由关羽", result.answer)
        self.assertEqual([fact.fact_id for fact in result.facts], ["F00017", "F00167"])
        self.assertNotEqual(result.facts[0].time_id, result.facts[1].time_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
