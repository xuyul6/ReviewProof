import unittest

from reviewproof.db import ALL_TABLES, EMBEDDING_DIM, apply_schema, embedding_column_dim
from tests.support import DatabaseTestCase


class SchemaTests(DatabaseTestCase):
    def test_apply_schema_is_idempotent_and_creates_tables(self):
        apply_schema(self.conn)  # setUp already applied it once
        names = {r[0] for r in self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")}
        self.assertTrue(set(ALL_TABLES) <= names)

    def test_embedding_column_dim_matches_schema(self):
        self.assertEqual(embedding_column_dim(self.conn), EMBEDDING_DIM)

    def test_vector_extension_and_dimension(self):
        self.assertEqual(
            self.conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()[0], "vector")
        typ = self.conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'").fetchone()[0]
        self.assertEqual(typ, f"vector({EMBEDDING_DIM})")


if __name__ == "__main__":
    unittest.main()
