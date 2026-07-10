import os
from typing import Any, List, Optional

try:
    from dashscope import TextEmbedding
except ImportError:
    TextEmbedding = None


class EmbeddingUnavailable(RuntimeError):
    """Raised when the configured remote embedding provider cannot supply vectors."""


class DashScopeEmbeddingClient:
    DEFAULT_MODEL = "text-embedding-v4"
    DEFAULT_DIMENSION = 1024
    MAX_DOCUMENT_BATCH_SIZE = 10

    def __init__(
        self,
        *,
        api_key: str,
        workspace: str,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        text_embedding: Any = None,
    ):
        self._api_key = api_key.strip()
        self._workspace = workspace.strip()
        self._model = model.strip()
        self._dimension = int(dimension)
        self._text_embedding = text_embedding if text_embedding is not None else TextEmbedding
        if not self._api_key or not self._workspace or not self._model or self._dimension <= 0:
            raise EmbeddingUnavailable("DashScope embedding configuration is incomplete")
        if self._text_embedding is None:
            raise EmbeddingUnavailable("dashscope SDK is not installed")

    @classmethod
    def from_env(cls, *, text_embedding: Any = None) -> "DashScopeEmbeddingClient":
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_HTTP_BASE_URL", "").strip()
        workspace = os.getenv("DASHSCOPE_WORKSPACE", "")
        model = os.getenv("EMBEDDING_MODEL", cls.DEFAULT_MODEL)
        dimension = os.getenv("EMBEDDING_DIMENSION", str(cls.DEFAULT_DIMENSION))
        if not base_url:
            raise EmbeddingUnavailable("DASHSCOPE_HTTP_BASE_URL is required")
        # DashScope SDK reads this documented variable when issuing requests.
        # Keep the application-facing name explicit about the workspace endpoint.
        os.environ["DASHSCOPE_BASE_HTTP_API_URL"] = base_url
        try:
            client = cls(
                api_key=api_key,
                workspace=workspace,
                model=model,
                dimension=int(dimension),
                text_embedding=text_embedding,
            )
            if client.model != cls.DEFAULT_MODEL or client.dimension != cls.DEFAULT_DIMENSION:
                raise EmbeddingUnavailable("Only text-embedding-v4 with 1024 dimensions is supported")
            return client
        except ValueError as ex:
            raise EmbeddingUnavailable("EMBEDDING_DIMENSION must be an integer") from ex

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.MAX_DOCUMENT_BATCH_SIZE):
            vectors.extend(self._embed(
                texts[start:start + self.MAX_DOCUMENT_BATCH_SIZE],
                text_type="document",
            ))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text], text_type="query")
        return vectors[0]

    def _embed(self, texts: List[str], *, text_type: str) -> List[List[float]]:
        if not texts:
            return []
        response = self._text_embedding.call(
            api_key=self._api_key,
            workspace=self._workspace,
            model=self._model,
            input=texts,
            text_type=text_type,
            dimension=self._dimension,
        )
        if getattr(response, "status_code", None) != 200:
            raise EmbeddingUnavailable(getattr(response, "message", "DashScope embedding request failed"))

        output = getattr(response, "output", None) or {}
        embeddings = output.get("embeddings", []) if isinstance(output, dict) else getattr(output, "embeddings", [])
        vectors = []
        for item in embeddings:
            vector = item.get("embedding", []) if isinstance(item, dict) else getattr(item, "embedding", [])
            if len(vector) != self._dimension:
                raise EmbeddingUnavailable("DashScope returned an unexpected embedding dimension")
            vectors.append(list(vector))
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable("DashScope returned an unexpected embedding count")
        return vectors
