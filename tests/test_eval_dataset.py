import dataclasses
import unittest

from reviewproof.eval.dataset import EvalQuestion, load_questions, sample_questions
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.loader import load_dataset
from tests.support import DatabaseTestCase, make_manifest, make_product


def eq(qid, category="Electronics", answerable=True):
    return EvalQuestion(qid=qid, product_id="A", category=category, split="dev", strategy="hybrid",
                        question_text="q", question_type="descriptive", answerable_silver=answerable, snippets=())


class LoadQuestionsTests(DatabaseTestCase):
    def test_load_questions_fields_and_strategy(self):
        q = Question(qid="q2", product_id="B", question_text="Does it fit?", question_type="yesno",
                     is_answerable=False, answers=({"answerText": "x"},), review_snippets=("Fits well.",))
        long_texts = ["word " * 2000 + str(i) for i in range(4)]
        manifest = make_manifest("A", "B")
        manifest["products"][1]["split"] = "test"
        load_dataset(self.conn, manifest,
                     {"A": make_product("A"), "B": dataclasses.replace(make_product("B", long_texts), category="Beauty")},
                     {"B": [q]})
        self.assertEqual(load_questions(self.conn, "dev", full_context_max_tokens=6000), [])
        [loaded] = load_questions(self.conn, "test", full_context_max_tokens=6000)
        self.assertEqual(loaded, EvalQuestion(qid="q2", product_id="B", category="Beauty", split="test",
                                              strategy="hybrid", question_text="Does it fit?", question_type="yesno",
                                              answerable_silver=False, snippets=("Fits well.",)))


class SampleQuestionsTests(unittest.TestCase):
    def setUp(self):
        self.questions = ([eq(f"e{i:02d}") for i in range(20)] + [eq(f"n{i:02d}", answerable=False) for i in range(5)]
                          + [eq(f"b{i:02d}", category="Beauty") for i in range(3)]
                          + [eq("bx", category="Beauty", answerable=False)])

    def test_sample_is_deterministic_and_sorted(self):
        a = sample_questions(self.questions, 10, seed=7)
        self.assertEqual(a, sample_questions(self.questions, 10, seed=7))
        self.assertEqual(len(a), 10)
        self.assertEqual([q.qid for q in a], sorted(q.qid for q in a))

    def test_every_stratum_represented(self):
        strata = {(q.category, q.answerable_silver) for q in sample_questions(self.questions, 6, seed=1)}
        self.assertEqual(strata, {("Electronics", True), ("Electronics", False), ("Beauty", True), ("Beauty", False)})

    def test_n_larger_than_pool_returns_all(self):
        self.assertEqual(len(sample_questions(self.questions, 1000, seed=1)), len(self.questions))


if __name__ == "__main__":
    unittest.main()
