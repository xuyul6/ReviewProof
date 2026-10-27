import unittest

from reviewproof.ingest.amazonqa import Review
from reviewproof.ingest.chunking import MAX_CHARS, chunk_review, estimate_tokens


def review(text):
    return Review(
        review_id="A-r00000", product_id="A", review_index=0, raw_text=text, text=text,
        helpful_yes=0, helpful_total=0, source_line=1, source_position=0,
    )


class ChunkingTests(unittest.TestCase):
    def test_short_review_is_one_chunk(self):
        c = chunk_review(review("Works great."))
        self.assertEqual([(x.chunk_id, x.start_offset, x.end_offset) for x in c], [("A-r00000-c00", 0, 12)])
        self.assertEqual((c[0].review_id, c[0].product_id, c[0].chunk_index), ("A-r00000", "A", 0))

    def test_long_review_chunks_are_bounded_contiguous_and_overlapping(self):
        text = " ".join(f"Sentence number {i} talks about the battery life." for i in range(40))
        c = chunk_review(review(text))
        self.assertGreater(len(c), 1)
        for x in c:
            self.assertLessEqual(len(x.text), MAX_CHARS)
            self.assertEqual(x.text, text[x.start_offset:x.end_offset])
        self.assertEqual((c[0].start_offset, c[-1].end_offset), (0, len(text)))
        for a, b in zip(c, c[1:]):
            self.assertLess(b.start_offset, a.end_offset)  # one-sentence overlap
        self.assertEqual([x.chunk_index for x in c], list(range(len(c))))

    def test_no_chunk_is_only_the_overlap_sentence(self):
        # s1 + s2 fit in 800 chars; s2 + s3 do not, so overlapping on s2 would yield a chunk of just s2.
        s1, s2, s3 = "a" * 299 + ".", "b" * 299 + ".", "c" * 699 + "."
        text = " ".join((s1, s2, s3))
        c = chunk_review(review(text))
        self.assertEqual([x.text for x in c], [f"{s1} {s2}", s3])

    def test_leading_punctuation_is_covered(self):
        text = "...and then it broke. " + "Filler sentence about the case. " * 40
        c = chunk_review(review(text.strip()))
        self.assertEqual(c[0].start_offset, 0)

    def test_giant_sentence_without_punctuation_is_hard_split(self):
        text = ("word " * 1000).strip()  # about 5,000 chars, no sentence end
        c = chunk_review(review(text))
        self.assertTrue(all(len(x.text) <= MAX_CHARS for x in c))
        self.assertEqual((c[0].start_offset, c[-1].end_offset), (0, len(text)))

    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens("abcd" * 3), 3)
        self.assertEqual(estimate_tokens("abcde"), 2)


if __name__ == "__main__":
    unittest.main()
