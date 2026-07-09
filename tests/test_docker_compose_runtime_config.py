import unittest
from pathlib import Path


class DockerComposeRuntimeConfigTests(unittest.TestCase):
    def test_echomind_service_targets_compose_chromadb(self):
        compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")

        start = compose_text.index("  echomind:")
        end = compose_text.index("  # ── Nginx", start)
        echomind_block = compose_text[start:end]

        self.assertIn("- CHROMA_HOST=chromadb", echomind_block)
        self.assertIn("- CHROMA_PORT=8000", echomind_block)


if __name__ == "__main__":
    unittest.main()
