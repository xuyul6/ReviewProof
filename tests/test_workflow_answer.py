import unittest

from reviewproof.retrieval.types import Candidate, ChunkRow
from reviewproof.workflow.answer import AnswerClaim, final_status, is_chinese, render_answer, verify_claims
from reviewproof.workflow.prompts import (ASSESS_SYSTEM, GENERATE_SYSTEM, PLAN_SYSTEM, assess_prompt,
                                          generate_prompt, plan_prompt)
from reviewproof.workflow.schemas import Claim


def cand(chunk_id, text, product_id="A"):
    return Candidate(ChunkRow(chunk_id, chunk_id.rsplit("-c", 1)[0], product_id, text))


CANDS_A = [cand("A-r00001-c00", "Too loud for me."), cand("A-r00002-c00", "Fits the 4th gen iPad.")]
FOREIGN = cand("B-r00000-c00", "Too loud for me.", product_id="B")


def render(**overrides):
    args = dict(question="Is it loud?", route="review_question", out_of_scope_reason="", service_status="ok",
                status="answered", verdict=None, claims=[], unknowns=[])
    return render_answer(**{**args, **overrides})


class VerifyClaimsTests(unittest.TestCase):
    def test_drops_unknown_and_foreign_ids_and_numbers_citations(self):
        claims = [Claim(text="Loud.", chunk_ids=["A-r00001-c00", "B-r00000-c00", "nope"]),
                  Claim(text="Only bad ids.", chunk_ids=["nope"]),
                  Claim(text="Again loud.", chunk_ids=["A-r00001-c00", "A-r00002-c00"])]
        kept, citations, dropped = verify_claims(claims, CANDS_A + [FOREIGN], "A")
        self.assertEqual([(c.text, c.citation_numbers) for c in kept], [("Loud.", (1,)), ("Again loud.", (1, 2))])
        self.assertEqual([(c.number, c.chunk_id, c.review_id) for c in citations],
                         [(1, "A-r00001-c00", "A-r00001"), (2, "A-r00002-c00", "A-r00002")])
        self.assertEqual(citations[0].text, "Too loud for me.")
        self.assertEqual(set(dropped), {"B-r00000-c00", "nope"})


class FinalStatusTests(unittest.TestCase):
    def test_rules(self):
        one = [AnswerClaim("x", (1,))]
        self.assertEqual(final_status("answered", one, []), "answered")
        self.assertEqual(final_status("answered", one, ["size"]), "partial")
        self.assertEqual(final_status("partial", one, []), "partial")
        self.assertEqual(final_status("partial", [], []), "unanswerable")
        self.assertEqual(final_status("answered", [], []), "unanswerable")
        self.assertEqual(final_status("unanswerable", one, []), "unanswerable")


class RenderTests(unittest.TestCase):
    def test_is_chinese(self):
        self.assertTrue(is_chinese("声音大吗？"))
        self.assertFalse(is_chinese("Is it loud?"))

    def test_service_failure_never_says_unanswerable(self):
        text = render(service_status="failed", status=None)
        self.assertEqual(text, "A service error stopped this run; this is not a finding about the reviews.")
        self.assertEqual(render(question="声音大吗？", service_status="failed", status=None),
                         "服务出错，这次没有完成检索或评估，不代表评论里没有答案。")

    def test_chinese_partial_with_verdict(self):
        text = render(question="声音大吗？", status="partial", verdict="mixed",
                      claims=[AnswerClaim("有人说很吵。", (1,))], unknowns=["夜间使用"])
        self.assertEqual(text, "结论：评论意见不一\n有人说很吵。 [1]\n评论没有提到：夜间使用")

    def test_english_answered_with_two_claims(self):
        text = render(verdict="yes", claims=[AnswerClaim("It is loud.", (1, 2)), AnswerClaim("Hard to adjust.", (3,))],
                      unknowns=["at night", "with earplugs"])
        self.assertEqual(text, "Verdict: Yes\nIt is loud. [1][2]\nHard to adjust. [3]\n"
                               "Not covered by the reviews: at night; with earplugs")

    def test_unanswerable(self):
        self.assertEqual(render(status="unanswerable"), "The reviews don't confirm this.")
        self.assertEqual(render(question="防水吗？", status="unanswerable", verdict="unknown"),
                         "结论：无法确认\n评论无法确认这个问题。")

    def test_out_of_scope(self):
        self.assertEqual(render(question="多少钱？", route="out_of_scope", out_of_scope_reason="价格不由评论决定",
                                status="unanswerable"),
                         "这个问题不在评论能回答的范围内：价格不由评论决定")
        self.assertEqual(render(route="out_of_scope", out_of_scope_reason="price", status="unanswerable"),
                         "This question is outside what reviews can answer: price")


class PromptTests(unittest.TestCase):
    def test_prompts_contain_question_and_chunks(self):
        p = assess_prompt("Is it loud?", CANDS_A)
        self.assertIn("[A-r00001-c00] Too loud for me.", p)
        self.assertIn("Is it loud?", p)
        g = generate_prompt("Is it loud?", True, "partial", CANDS_A, ["at night"])
        self.assertIn("[A-r00002-c00] Fits the 4th gen iPad.", g)
        self.assertIn("at night", g)
        self.assertIn("Is it loud?", plan_prompt("Is it loud?"))

    def test_system_prompts_state_the_rules(self):
        self.assertIn("out_of_scope", PLAN_SYSTEM)
        self.assertIn("English", PLAN_SYSTEM)
        for status in ("answered", "partial", "unanswerable"):
            self.assertIn(status, ASSESS_SYSTEM)
        self.assertIn("mixed", GENERATE_SYSTEM)


if __name__ == "__main__":
    unittest.main()
