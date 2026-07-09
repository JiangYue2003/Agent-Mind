# Local BGE Reranker v2-m3 Service Design

## Status

Approved architecture pending implementation planning and final spec review.

## Context

EchoMind's knowledge search follows this path:

1. `KnowledgeBase` retrieves child chunks with vector search and BM25.
2. It fuses the two result sets with RRF.
3. `_rerank_candidates()` currently calls DashScope's `qwen3-rerank` only when
   `DASHSCOPE_API_KEY` is available; otherwise it returns the RRF order.
4. The selected chunks are enriched with their parent context before being
   passed into the chat pipeline.

The local service must replace only step 3. It must not change the bounded
workflow, MCP tool contract, chunking, recall logic, or prompt assembly.

## Goal

Run `BAAI/bge-reranker-v2-m3` as a GPU-backed, independently deployable
Docker Compose service so EchoMind can rerank retrieved knowledge chunks on
the local network during each request.

## Non-goals

- Do not replace ChromaDB's embedding model or the vector store.
- Do not add an autonomous agent loop or alter the existing workflow state
  machine.
- Do not expose the reranker to the public host network.
- Do not remove the existing DashScope configuration in this change. It
  remains a supported alternate provider.
- Do not tune score thresholds without evaluation evidence.

## Considered Approaches

### 1. Hugging Face Text Embeddings Inference (selected)

Run Text Embeddings Inference (TEI) with its native reranking endpoint. It
provides a small, dedicated HTTP surface and server-side dynamic batching. The
main application only needs a provider adapter that translates candidate
chunks to TEI's request and maps the indexed scores back to candidates.

This is the selected approach because it isolates CUDA and model dependencies
from the application image, preserves the existing Compose operational model,
and directly supports cross-encoder reranking.

### 2. Custom FastAPI service using FlagEmbedding

A custom Python service would provide fine-grained control of model loading,
batching, and future BGE-specific features. It also makes EchoMind responsible
for maintaining CUDA, PyTorch, request queuing, and a separate API contract.
It is not justified while TEI meets the required interface.

### 3. External reranking API

This preserves the current operational simplicity but does not meet the local
GPU inference or reduced network dependency requirements.

## Architecture

```text
vector recall + BM25 -> RRF -> KnowledgeBase rerank adapter
                                       |
                                       | POST http://reranker:80/rerank
                                       v
                reranker Compose service (TEI + GPU + bge-reranker-v2-m3)
                                       |
                                       v
                        indexed scores -> existing candidates -> Top K
```

`reranker` joins the existing `echomind-network`. It has no `ports` mapping,
so only services in that Compose network can reach it by the DNS name
`reranker`. The model cache persists in the `reranker-model-cache` named
volume mounted at `/data`.

The service requests the NVIDIA GPU through current Docker Compose GPU support
and uses the official CUDA TEI image. It starts with these deliberately
conservative settings for an RTX 5070 with 12 GB of VRAM:

| Setting | Initial value | Reason |
| --- | --- | --- |
| Model | `BAAI/bge-reranker-v2-m3` | Multilingual cross-encoder, suitable for Chinese knowledge search |
| Data type | `float16` | Lowers model memory use and enables GPU inference |
| Max concurrent requests | `8` | Applies backpressure instead of unbounded queuing |
| Max batch tokens | `8192` | Safe starting point for the current short child chunks |
| Max client batch size | `32` | Covers the default recall set of 20 chunks |
| Auto truncate | enabled | Bounds requests that exceed the model limit |

These are starting limits, not throughput claims. They will be tuned only
after a local load test confirms GPU memory use and p95 latency.

## Request and Response Contract

The application calls TEI directly:

```json
{
  "query": "退款到账需要多久",
  "texts": ["候选片段 A", "候选片段 B"],
  "return_text": false,
  "truncate": true
}
```

TEI returns a relevance-sorted JSON array. Each item contains the original
candidate `index` and a relevance `score`. The adapter validates every index,
copies the associated candidate, writes both `rerank_score` and `score`, and
returns the first requested number of valid candidates.

`raw_scores` remains disabled. TEI then returns sigmoid scores in the 0-to-1
range, which is easier to expose in existing response data. Score scale is
model-specific and must not be treated as interchangeable with Qwen scores.

## Application Configuration

The provider becomes explicit:

```text
RERANK_PROVIDER=tei
RERANK_URL=http://reranker:80/rerank
RAG_RERANK_TIMEOUT_SECONDS=3
RAG_RERANK_SCORE_THRESHOLD=0
```

`RERANK_PROVIDER` selects either `tei` or `dashscope`; existing DashScope
environment variables continue to configure the latter. The Compose service
sets `RERANK_PROVIDER=tei` and the internal `RERANK_URL` for the application.
No API key is needed for TEI.

The threshold starts at `0`. Existing child chunks are normally about 180
characters, and reranking should first be judged by ordering rather than by
dropping results with a threshold copied from a different model. An evaluation
run may later establish a non-zero local threshold.

## Failure, Availability, and Startup Behavior

- `reranker` has an HTTP health check. EchoMind depends on that health check,
  so a first boot waits until the model download and GPU initialization finish.
- On a normal request, the adapter uses a three-second HTTP timeout.
- Network errors, non-2xx responses, malformed result arrays, invalid
  candidate indexes, or a response with no usable candidates all produce the
  current RRF-order fallback. The chat request continues.
- A failure is logged with the provider and reason, without logging query or
  document content.
- The existing `knowledge.rerank` trace stage remains. Its metadata gains the
  provider, input candidate count, returned candidate count, and whether the
  RRF fallback was used.

## Prerequisites

- NVIDIA driver that supports the RTX 5070.
- Docker Desktop using its WSL2 GPU integration, or a Linux Docker host with
  NVIDIA Container Toolkit configured.
- Network access to Hugging Face during the first model download. Afterwards
  the mounted cache allows restart without a download.

The implementation must verify actual GPU initialization. If the chosen TEI
CUDA image cannot run on the RTX 5070's CUDA architecture, it must update to a
TEI CUDA image that supports the installed driver before proceeding; CPU
fallback is not an acceptable deployment mode for this feature.

## Verification Plan

1. Add focused unit tests for the TEI request body, indexed-score mapping,
   malformed responses, timeout fallback, and provider selection.
2. Extend the Compose configuration test to assert the internal reranker URL
   and service dependency.
3. Run the existing knowledge retrieval and trace tests to protect the
   `rerank_score` contract.
4. Validate the generated Compose file with `docker compose config`.
5. Start only the reranker, confirm its health endpoint, and issue a real
   `/rerank` request from the Compose network.
6. Start the full stack and call `/knowledge/search`; confirm responses are
   locally reranked and the trace contains `knowledge.rerank` metadata.
7. Temporarily stop the reranker and confirm a search still succeeds in RRF
   order.

## Acceptance Criteria

- `docker compose up` starts a healthy GPU-backed reranker and mounts a
  persistent model cache.
- EchoMind calls `http://reranker:80/rerank`, not a public endpoint, when
  `RERANK_PROVIDER=tei`.
- A successful search exposes scores that originate from the TEI response.
- Reranker unavailability does not fail `/chat` or `/knowledge/search`.
- All focused and affected existing tests pass.
- No unrelated user worktree changes are modified.
