import unittest
from pathlib import Path


class SkillDockerConfigTests(unittest.TestCase):
    def test_production_compose_persists_uploaded_skills_and_exposes_git_catalog(self):
        compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
        start = compose_text.index("  echomind:")
        end = compose_text.index("  # ── Nginx", start)
        echomind_block = compose_text[start:end]

        self.assertIn("./skills/catalog:/app/skills/catalog:ro", echomind_block)
        self.assertIn("./data/skills:/app/data/skills", echomind_block)


if __name__ == "__main__":
    unittest.main()
