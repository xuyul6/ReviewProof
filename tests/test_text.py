import unittest

from reviewproof.text import dedupe_key, normalize_text


class TextTests(unittest.TestCase):
    def test_normalize_unescapes_and_collapses(self):
        self.assertEqual(normalize_text("  It&#39;s\n\t&quot;ok&quot;  "), "It's \"ok\"")
        self.assertEqual(dedupe_key("Great  Product"), "great product")


if __name__ == "__main__":
    unittest.main()
