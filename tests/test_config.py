import unittest

from reviewproof.config import load_settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = load_settings({})
        self.assertEqual(s.database_url, "postgresql://reviewproof:reviewproof@localhost:54329/reviewproof")
        self.assertEqual((s.embedding_model, s.embedding_dim, s.embedding_batch_size), ("BAAI/bge-m3", 1024, 32))

    def test_env_overrides(self):
        s = load_settings({"REVIEWPROOF_DATABASE_URL": "postgresql://x/y", "REVIEWPROOF_EMBEDDING_BATCH_SIZE": "8"})
        self.assertEqual((s.database_url, s.embedding_batch_size), ("postgresql://x/y", 8))

    def test_embedding_device_is_auto_unless_set(self):
        self.assertIsNone(load_settings({}).embedding_device)
        self.assertIsNone(load_settings({"REVIEWPROOF_EMBEDDING_DEVICE": " "}).embedding_device)
        self.assertEqual(load_settings({"REVIEWPROOF_EMBEDDING_DEVICE": "mps"}).embedding_device, "mps")

    def test_bad_int_names_the_variable(self):
        with self.assertRaisesRegex(ValueError, "REVIEWPROOF_EMBEDDING_DIM"):
            load_settings({"REVIEWPROOF_EMBEDDING_DIM": "abc"})


if __name__ == "__main__":
    unittest.main()
