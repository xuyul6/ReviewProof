import unittest

from reviewproof.eval.alignment import ReviewSpans, align_snippets, silver_qrels
from reviewproof.eval.dataset import load_questions
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.loader import load_dataset
from tests.support import DatabaseTestCase, make_manifest, make_product


def review(review_id, text, *spans):
    return ReviewSpans(review_id, text, tuple(spans) or ((f"{review_id}-c00", 0, len(text)),))


class AlignSnippetsTests(unittest.TestCase):
    def test_exact_snippet_hits_chunk(self):
        reviews = [review("A-r00000", "The fan is very loud at night."), review("A-r00001", "Battery lasts two days.")]
        self.assertEqual(align_snippets(["very loud at night"], reviews), {"A-r00000-c00"})

    def test_whitespace_insensitive_match(self):
        reviews = [review("A-r00000", "This one works great!!! works every time.")]
        self.assertEqual(align_snippets(["works great !! ! works every"], reviews), {"A-r00000-c00"})

    def test_html_entities_normalized(self):
        reviews = [review("A-r00000", 'They call it "quiet" but it is not.')]
        self.assertEqual(align_snippets(["call it &quot;quiet&quot; but"], reviews), {"A-r00000-c00"})

    def test_snippet_inside_one_chunk_hits_only_that_chunk(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        r = review("A-r00000", text, ("A-r00000-c00", 0, 42), ("A-r00000-c01", 21, 63))
        self.assertEqual(align_snippets(["First sentence here"], [r]), {"A-r00000-c00"})
        self.assertEqual(align_snippets(["Second sentence"], [r]), {"A-r00000-c00", "A-r00000-c01"})  # overlap

    def test_snippet_across_chunk_boundary_hits_both(self):
        text = "Alpha beta gamma. Delta epsilon zeta."
        r = review("A-r00000", text, ("A-r00000-c00", 0, 17), ("A-r00000-c01", 18, 37))
        self.assertEqual(align_snippets(["gamma. Delta epsilon"], [r]), {"A-r00000-c00", "A-r00000-c01"})

    def test_unmatched_and_too_short_snippets(self):
        reviews = [review("A-r00000", "ok it is ok really ok")]
        self.assertEqual(align_snippets(["waterproof in heavy rain"], reviews), frozenset())
        self.assertEqual(align_snippets(["ok"], reviews), frozenset())  # under 10 non-space characters


class SilverQrelsTests(DatabaseTestCase):
    def test_only_same_product_chunks(self):
        q = Question(qid="q1", product_id="A", question_text="Is it loud?", question_type="yesno",
                     is_answerable=True, answers=(), review_snippets=("Too loud for me.",))
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me."])}, {"A": [q]})
        questions = load_questions(self.conn, "dev", full_context_max_tokens=6000)
        self.assertEqual(silver_qrels(self.conn, questions), {"q1": frozenset({"A-r00001-c00"})})


if __name__ == "__main__":
    unittest.main()
