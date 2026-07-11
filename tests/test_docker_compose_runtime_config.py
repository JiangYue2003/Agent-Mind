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

    def test_echomind_uses_host_local_gpu_reranker_by_default(self):
        compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("  reranker:", compose_text)
        reranker_start = compose_text.index("  reranker:")
        reranker_end = compose_text.index("  # ── EchoMind", reranker_start)
        reranker_block = compose_text[reranker_start:reranker_end]

        echomind_start = compose_text.index("  echomind:")
        echomind_end = compose_text.index("  # ── Nginx", echomind_start)
        echomind_block = compose_text[echomind_start:echomind_end]

        self.assertIn("BAAI/bge-reranker-v2-m3", reranker_block)
        self.assertIn("gpus: all", reranker_block)
        self.assertIn('profiles: ["tei"]', reranker_block)
        self.assertIn('- --max-concurrent-requests\n        - "4"', reranker_block)
        self.assertIn('- --max-client-batch-size\n        - "15"', reranker_block)
        self.assertIn("reranker-model-cache:/data", reranker_block)
        self.assertNotIn("ports:", reranker_block)
        self.assertIn("- RERANK_PROVIDER=tei", echomind_block)
        self.assertIn("RERANK_URL=${RERANK_URL:-http://host.docker.internal:18080/rerank}", echomind_block)
        self.assertIn("- RAG_HYBRID_RECALL_K=20", echomind_block)
        self.assertIn("- RAG_RERANK_CANDIDATE_K=20", echomind_block)
        self.assertIn("- RAG_ANSWER_TOP_K=5", echomind_block)
        self.assertIn("- RAG_RERANK_SCORE_THRESHOLD=0.05", echomind_block)
        self.assertNotIn("reranker:\n        condition", echomind_block)

    def test_embedding_v4_settings_are_documented_without_credentials(self):
        env_example = Path(".env.example").read_text(encoding="utf-8")

        self.assertIn("DASHSCOPE_API_KEY=your_dashscope_api_key_here", env_example)
        self.assertIn("DASHSCOPE_HTTP_BASE_URL=https://your-workspace-host/api/v1", env_example)
        self.assertIn("DASHSCOPE_WORKSPACE=ws-your_workspace", env_example)
        self.assertIn("EMBEDDING_MODEL=text-embedding-v4", env_example)
        self.assertIn("EMBEDDING_DIMENSION=1024", env_example)

    def test_evaluation_defaults_use_recall_20_and_reranked_top_5(self):
        env_example = Path(".env.example").read_text(encoding="utf-8")

        self.assertIn("EVAL_TOP_K=5", env_example)
        self.assertIn("EVAL_RECALL_K=20", env_example)
        self.assertIn("RAG_HYBRID_RECALL_K=20", env_example)
        self.assertIn("RAG_RERANK_CANDIDATE_K=20", env_example)
        self.assertIn("RAG_RERANK_SCORE_THRESHOLD=0.05", env_example)


if __name__ == "__main__":
    unittest.main()
