# DashScope text-embedding-v4 Full Migration Design

## Status

Architecture approved. Pending spec review before implementation planning.

## Context

EchoMind currently has three independent vector-producing paths:

1. `mcp/knowledge_base.py` stores parent and child knowledge chunks in
   ChromaDB without explicit embeddings. ChromaDB therefore uses its default
   local embedding function for document writes and `query_texts` searches.
2. `memory/conversation_memory.py` stores episodic memory and user profiles
   in ChromaDB with the same implicit embedding behavior.
3. `core/intent_recognizer.py` can call DashScope `TextEmbedding.call`, but
   only for intent template matching. If that call fails, it currently uses a
   local character n-gram vector.

The existing default Chroma embedding model is not an acceptable retrieval
baseline for a Chinese-first customer-service knowledge base. All new vector
indexing and semantic queries must instead use Alibaba DashScope
`text-embedding-v4`.

## Goal

Make DashScope `text-embedding-v4` the only embedding provider for all active
semantic paths:

- knowledge-base parent and child chunks;
- episodic conversation memory and user profile memory;
- intent template embedding and user-message semantic matching.

The user will manually upload knowledge-base documents after the migration.
No old vectors may participate in active retrieval.

## Non-goals

- Do not migrate or preserve old vector values across model spaces.
- Do not add multimodal document ingestion; this is a text-only embedding
  migration.
- Do not write API keys into source files, Compose files, tests, logs, or
  committed `.env` files.
- Do not delete old Chroma collections automatically.
- Do not change the reranker, workflow state machine, LLM orchestration, or
  document chunking rules.

## Selected Strategy

Create new, versioned Chroma collections and make all active reads and writes
use externally generated DashScope vectors.

| Active purpose | New collection |
| --- | --- |
| Knowledge parents | `knowledge_base_parent_v4` |
| Knowledge children | `knowledge_base_child_v4` |
| Episodic memory | `episodic_v4` |
| User profiles | `user_profile_v4` |

The old collections remain physically intact but are never opened by the
active code path. This prevents Chroma dimension mismatches, avoids an
automatic destructive migration, and leaves the user in control of legacy
data cleanup.

The initial active collections are empty. The user repopulates the knowledge
base by using the existing document upload path after deployment. Historical
episodic/profile memory becomes inactive and new memory accumulates in the
v4 collections from the cutover point.

## Embedding Client

A single shared DashScope embedding client will be introduced for synchronous
knowledge indexing/search, asynchronous memory operations, and intent
recognition.

It will use the installed DashScope Python SDK and configure the workspace
endpoint through environment variables. It exposes two narrow operations:

- `embed_documents(texts)`: batch document chunks with `text_type=document`;
- `embed_query(text)`: one search query with `text_type=query`.

The client uses `text-embedding-v4` with a fixed 1024-dimensional dense
vector. The dimension is configurable only through one environment setting
and is validated before vectors are stored or queried. Chroma collections use
explicit `embeddings=` on writes and `query_embeddings=` on searches, so
Chroma's default embedding function is never invoked.

## Configuration

The deployment provides configuration only through environment variables:

```text
DASHSCOPE_API_KEY=<secret, supplied outside git>
DASHSCOPE_HTTP_BASE_URL=https://<workspace-host>/api/v1
DASHSCOPE_WORKSPACE=<workspace-id>
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1024
```

`DASHSCOPE_HTTP_BASE_URL` and `DASHSCOPE_WORKSPACE` are required because the
credential is scoped to a workspace-specific Model Studio endpoint. The
application must fail its embedding preflight clearly if these values are
missing or DashScope rejects the request.

Before production cutover, the API key must be rotated because it was exposed
outside the secret store. The replacement key is placed only in the local
runtime `.env` file or deployment secret manager.

## Read and Write Flows

### Knowledge import

1. Existing document chunking produces parent and child chunks unchanged.
2. The client batches parent and child texts to DashScope as documents.
3. Chroma receives IDs, documents, metadata, and explicit v4 embeddings.
4. A partial batch failure fails the import and reports the failed batch; it
   must not silently insert a local Chroma embedding.

### Knowledge search

1. Vector recall embeds the user query as a DashScope query vector.
2. The child collection searches with `query_embeddings` and returns vector
   candidates.
3. BM25 continues to execute locally and is fused with vector candidates by
   the existing RRF logic.
4. If DashScope is unavailable, vector recall is skipped and the search uses
   BM25 candidates only; it never substitutes all-MiniLM or a hashed vector.
5. Existing local TEI reranking and parent-context attachment remain
   unchanged.

### Memory and intent

Episodic/profile writes use explicit document embeddings in their new v4
collections, while episodic search uses explicit query embeddings. Intent
template matching uses the same client. If the remote client fails, intent
recognition skips the embedding similarity branch and continues through its
existing deterministic/LLM logic instead of using the local n-gram vector.

## Failure and Observability

- All remote embedding errors are logged without query text, document text,
  or credentials.
- Knowledge-search traces record embedding provider, model, vector dimension,
  request size, and whether vector recall was unavailable and BM25-only
  fallback was used.
- Ingestion reports a failure instead of mixing default Chroma embeddings with
  v4 embeddings.
- Runtime startup performs one small v4 preflight request. It must log a
  concise configuration failure but allow the service to start in
  BM25-only/non-embedding mode so operational endpoints remain diagnosable.

## Testing and Acceptance

1. Unit-test document and query requests against a fake DashScope client,
   including text type, fixed dimension, workspace/base-URL configuration, and
   batch behavior.
2. Unit-test explicit `embeddings` writes and `query_embeddings` reads for
   knowledge and memory collections; assert no `query_texts` vector search is
   used on active v4 collections.
3. Unit-test that a remote embedding failure produces BM25-only retrieval and
   never a local vector fallback.
4. Unit-test intent recognition to ensure DashScope failure does not call the
   local n-gram embedding fallback when the v4 provider is configured.
5. Run affected knowledge, memory, intent, trace, and Compose tests.
6. From the running application container, call DashScope v4 using the
   workspace endpoint and report only status code, model, vector dimension,
   and usage metadata.
7. Upload a small Chinese knowledge document, query it through `/search`, and
   confirm a v4 vector recall trace followed by local reranking.

## Acceptance Criteria

- Every active Chroma collection has a v4-specific name and accepts only
  1024-dimensional DashScope vectors.
- Knowledge import and vector search use explicit external vectors.
- Old Chroma collections are never read by active code.
- DashScope failure cannot cause all-MiniLM or local n-gram vectors to enter
  an active v4 collection or retrieval calculation.
- The user can upload Chinese knowledge documents after cutover and retrieve
  them through the existing RAG path.
- The repository contains no API credentials.
