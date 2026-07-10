import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_DEVICE = "cuda:0"


class Reranker(Protocol):
    def compute_score(self, pairs: list[list[str]]) -> Any:
        """Return one relevance score for each query-document pair."""


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    texts: list[str]
    return_text: bool = False
    truncate: bool = True


def load_flag_reranker() -> Reranker:
    """Load one GPU-resident reranker and fail instead of silently using CPU."""
    import torch
    from FlagEmbedding import FlagReranker

    device = os.getenv("RERANK_DEVICE", DEFAULT_DEVICE)
    if not device.startswith("cuda"):
        raise RuntimeError("RERANK_DEVICE must target a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU reranker fallback")

    return FlagReranker(
        os.getenv("RERANK_MODEL_ID", DEFAULT_MODEL_ID),
        devices=[device],
        use_fp16=True,
        query_max_length=256,
        passage_max_length=512,
    )


def create_app(reranker_factory: Callable[[], Reranker] | None = None) -> FastAPI:
    device = os.getenv("RERANK_DEVICE", DEFAULT_DEVICE)
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        factory = reranker_factory or load_flag_reranker
        app.state.reranker = factory()
        app.state.device = device
        yield
        app.state.reranker = None

    app = FastAPI(title="EchoMind Local Reranker", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "backend": "FlagEmbedding",
            "device": app.state.device,
        }

    @app.post("/rerank")
    def rerank(request: RerankRequest) -> list[dict[str, Any]]:
        if not request.texts:
            return []

        pairs = [[request.query, text] for text in request.texts]
        try:
            with inference_lock:
                raw_scores = app.state.reranker.compute_score(pairs, normalize=True)
            if len(request.texts) == 1 and not isinstance(raw_scores, (list, tuple)):
                scores = [float(raw_scores)]
            else:
                scores = [float(score) for score in raw_scores]
        except Exception as exc:
            logger.exception("Local reranker inference failed")
            raise HTTPException(status_code=500, detail="Rerank inference failed") from exc

        if len(scores) != len(request.texts):
            raise HTTPException(status_code=500, detail="Reranker returned an unexpected score count")

        results = [
            {"index": index, "score": score, "text": text}
            for index, (text, score) in enumerate(zip(request.texts, scores))
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        if not request.return_text:
            for item in results:
                item.pop("text")
        return results

    return app


app = create_app()
