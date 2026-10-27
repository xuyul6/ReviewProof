import unittest

from tests.support import check_test_database_url


class TestDatabaseGuardTests(unittest.TestCase):
    def test_refuses_a_database_not_named_test(self):
        # DatabaseTestCase truncates every table; one dropped suffix would wipe the real dataset.
        with self.assertRaisesRegex(RuntimeError, "_test"):
            check_test_database_url("postgresql://reviewproof:reviewproof@localhost:54329/reviewproof")

    def test_accepts_a_test_database(self):
        check_test_database_url("postgresql://reviewproof:reviewproof@localhost:54329/reviewproof_test")


if __name__ == "__main__":
    unittest.main()
