import unittest
from pathlib import Path


class RagasEvaluationContainerTests(unittest.TestCase):
    def test_dockerfile_has_a_separate_evaluation_stage(self):
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

        self.assertIn("FROM dependencies AS evaluation", dockerfile)
        self.assertIn("COPY evaluation/requirements-ragas.txt /tmp/requirements-ragas.txt", dockerfile)
        evaluation_stage = dockerfile.split("FROM dependencies AS evaluation", 1)[1].split(
            "FROM base AS production", 1
        )[0]
        self.assertIn("apt-get install -y --no-install-recommends git", evaluation_stage)

    def test_compose_evaluation_profile_uses_internal_app_address_and_isolated_mounts(self):
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")
        start = compose.index("  ragas-eval:")
        end = compose.index("  # ── Nginx", start)
        block = compose[start:end]

        self.assertIn('profiles: ["eval"]', block)
        self.assertIn("EVAL_APP_BASE_URL=http://echomind:8000", block)
        self.assertIn("./evaluation/datasets:/app/evaluation/datasets:ro", block)
        self.assertIn("./data/eval:/app/data/eval", block)
        self.assertIn("env_file:", block)
        self.assertIn("- .env", block)
        self.assertIn("- EVAL_RECALL_K=12", block)
        self.assertIn("- EVAL_REQUIRE_RERANK=true", block)
        self.assertIn("- EVAL_REQUIRED_RERANK_PROVIDER=tei", block)
        self.assertIn("- EVAL_MAX_CONCURRENCY=3", block)
        self.assertNotIn("ANTHROPIC_API_KEY=${", block)
        self.assertNotIn("EVAL_EMBEDDING_API_KEY=${", block)
        self.assertIn('entrypoint: ["python", "-m", "evaluation.ragas_runner"]', block)
        self.assertNotIn('command: ["python", "-m", "evaluation.ragas_runner"]', block)
        self.assertNotIn("ports:", block)

    def test_dockerignore_excludes_runtime_credentials(self):
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

        self.assertIn(".env", dockerignore.splitlines())

    def test_evaluation_requirements_pin_ragas_langchain_to_the_same_compatibility_family(self):
        requirements = Path("evaluation/requirements-ragas.txt").read_text(encoding="utf-8")

        self.assertIn("langchain>=0.3,<0.4", requirements)
        self.assertIn("langchain-core>=0.3,<0.4", requirements)
        self.assertIn("langchain-community>=0.3,<0.4", requirements)
        self.assertIn("langchain-openai>=0.3,<0.4", requirements)


if __name__ == "__main__":
    unittest.main()
