import unittest
from pathlib import Path

import yaml  # installed with transformers (sentence-transformers)

ROOT = Path(__file__).resolve().parents[1]


def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


class DockerConfigTests(unittest.TestCase):
    def test_dockerignore_keeps_secrets_and_data_out(self):
        # The image must never carry the API key in .env or the 1.5 GB of AmazonQA data.
        entries = {line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
        for entry in (".env", ".env.*", "data", ".venv", "frontend/node_modules"):
            self.assertIn(entry, entries)

    def test_api_waits_for_ingest_and_db(self):
        services = compose()["services"]
        self.assertEqual(services["api"]["depends_on"]["ingest"]["condition"], "service_completed_successfully")
        self.assertEqual(services["api"]["depends_on"]["db"]["condition"], "service_healthy")
        self.assertEqual(services["ingest"]["depends_on"]["db"]["condition"], "service_healthy")

    def test_host_ports_bind_to_localhost(self):
        services = compose()["services"]
        for name in ("db", "api"):
            for port in services[name]["ports"]:
                self.assertTrue(port.startswith("127.0.0.1:"), f"{name}: {port}")

    def test_api_key_comes_from_optional_env_file(self):
        # A fresh clone has no .env: the stack must still start, and questions end as service errors.
        self.assertEqual(compose()["services"]["api"]["env_file"], [{"path": ".env", "required": False}])
        self.assertNotIn("sk-ant", (ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
