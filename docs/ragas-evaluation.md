# EchoMind RAGAS Evaluation

## Scope

`ragas-eval` evaluates the deployed EchoMind service, not a locally imported
`KnowledgeBase`. Each user turn makes two independent calls through
`http://echomind:8000`: `/search` collects retrieval diagnostics, while `/chat`
produces the answer through the real workflow. The `/search` response is never
injected into `/chat`; the chat request still runs slot collection, planning,
state transition, and only then its own knowledge-search action when the plan
requires it.

The evaluator uses the existing Anthropic-compatible configuration. In this
deployment that is the DeepSeek endpoint exposed through:

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`

`AnswerRelevancy` additionally uses an OpenAI-compatible embedding endpoint:

- `EVAL_EMBEDDING_API_KEY`
- `EVAL_EMBEDDING_BASE_URL`
- `EVAL_EMBEDDING_MODEL=text-embedding-v4`

The evaluation service loads these runtime-only values from `.env`, rather
than interpolating ambient shell variables. This keeps it aligned with the
DeepSeek Anthropic-compatible settings used by EchoMind and prevents a shell
proxy setting from changing only the evaluation container. Credentials are
excluded from Docker build context and never written into reports.

## Dataset

`evaluation/datasets/customer_service_v1.jsonl` contains 80 cases grounded in
`docs/customer-service-kb`:

- 50 direct questions
- 15 paraphrases
- 10 easy-to-confuse questions
- 5 questions that the knowledge base cannot answer

Every answerable case has a reference answer and an exact evidence sentence
from its source document. The evaluator checks `/knowledge/chunks` before
running and fails if any answerable source title is absent from deployed
ChromaDB.

`evaluation/datasets/customer_service_workflow_v1.jsonl` adds strict multi-turn
workflow cases. Its first turn must request the missing order ID; the next turn
only supplies that ID in the same `user_id` and `conv_id`. The latter turn
therefore exposes lost workflow state instead of masking it by repeating the
original request.

## Run

Start the host-native reranker first, then start or recreate EchoMind with the
normal Compose workflow. An existing TEI container must be stopped during this
migration:

```powershell
.\tools\start_local_reranker.ps1
docker compose --profile tei stop reranker
docker compose up -d --build echomind
docker compose exec echomind curl -sS http://host.docker.internal:18080/health
```

### Run from the host

The Ragas runner can run from a host Conda environment. It calls the Docker
application through its published port, so it uses `http://127.0.0.1:8000`,
not the Compose-only `http://echomind:8000`. Datasets and reports are regular
Windows paths; no Docker volume mounts are involved.

Use Python 3.12 for the `ragas` environment, matching the evaluated Docker
image. Python 3.14 may resolve newer transitive packages than the pinned
container build.

```powershell
cd F:\AI-agent\EchoMind
conda activate ragas
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r evaluation/requirements-ragas.txt
```

Confirm the deployed application and the local reranker are ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:18080/health
docker compose exec echomind printenv RERANK_URL
```

The final command must print `http://host.docker.internal:18080/rerank`.
Then run the evaluator. `python -m dotenv run` loads the existing `.env` into
the local process without printing API keys.

```powershell
python -m dotenv run -- python -m evaluation.ragas_runner `
  --base-url http://127.0.0.1:8000 `
  --dataset evaluation/datasets/customer_service_v1.jsonl `
  --workflow-dataset evaluation/datasets/customer_service_workflow_v1.jsonl `
  --output-dir data/eval `
  --top-k 5 `
  --recall-k 20 `
  --require-rerank true `
  --required-rerank-provider tei `
  --max-concurrency 3
```

The report is written directly to `data\eval\ragas-<timestamp>.json`. To
resume scoring a saved report without calling `/search` or `/chat` again:

```powershell
python -m dotenv run -- python -m evaluation.ragas_runner `
  --resume data/eval/<run_id>.json `
  --max-concurrency 3
```

### Run from Docker

Alternatively, build and run the on-demand evaluation service:

```powershell
docker compose --profile eval build ragas-eval
docker compose --profile eval run --rm ragas-eval
```

Optional environment variables are `EVAL_RUN_ID`, `EVAL_TOP_K`,
`EVAL_RECALL_K`, `EVAL_REQUIRE_RERANK`, `EVAL_REQUIRED_RERANK_PROVIDER`,
`EVAL_MAX_CONCURRENCY`, `EVAL_DATASET_PATH`, `EVAL_WORKFLOW_DATASET`,
`EVAL_RETRIEVAL_GOLD_PATH`, and `EVAL_OUTPUT_DIR`. The defaults recall 20
candidates before the local GPU reranker produces Top-K 5, require the
TEI-compatible `tei` protocol marker
for answerable knowledge cases, run at most three RAGAS
metric tasks at once, and use the customer-service dataset plus workflow cases.

Query rewrites each perform hybrid recall, then their child-chunk candidates are merged
and deduplicated by `parent_id + child_chunk_index` before one global TEI-compatible rerank. This prevents one user request
from overwhelming the reranker with one rerank request per rewrite.
The host-native `local_reranker` service scores up to 20 highest RRF-scored
child candidates in one GPU batch before reranking to Top-K 5. Children below
the `0.05` rerank-score threshold are discarded; parent deduplication happens
only when final context is assembled for `/chat` and does not backfill lower-ranked children.
Start it before the
Compose application and verify its `/health` endpoint from the `echomind`
container. RAGAS scoring concurrency remains limited to three because that
limit protects the remote evaluation-model API, not the reranker.

The strict rerank gate is intentionally applied only to answerable knowledge
cases. Workflow clarification and unanswerable cases still call `/search` for
diagnosis, but cannot be required to have a non-empty candidate set. A strict
failure means the report is not a valid retrieval-backed evaluation; inspect
the app's reranker logs before rerunning.

Each run writes `data/eval/<run_id>.json` immediately after collecting the
deployed search contexts and chat responses. If scoring fails later, the same
file is retained with `status: scoring_failed` and the error message; completed
runs use `status: completed` and include `ragas_rows` for knowledge answers
plus `workflow_rows` for slot-collection and continuation checks. Reports do
not contain API keys.

To re-score a saved report without calling `/search` or `/chat` again, run:

```powershell
docker compose --profile eval run --rm ragas-eval --resume /app/data/eval/<run_id>.json
```

## Metrics

Each evaluation turn calls `/search` once with `recall_k=20`, `top_k=5`, and
`include_debug=true`. It then computes deterministic retrieval metrics without
additional model or retrieval calls:

- `retrieval.raw` evaluates the merged child-chunk RRF candidate list at
  `Recall@20`, `Hit@20`, `MRR@20`, and `nDCG@20`.
- `retrieval.reranked` evaluates the same metrics after the local GPU reranker
  returns its final Top-5 child chunks.
- Unanswerable cases do not enter those averages; their raw and reranked noise
  rates show how often either layer returns any candidate.

The versioned `customer_service_retrieval_gold_v2.json` maps every answerable
knowledge case to an active child-chunk key and records that chunk's SHA-256
content hash. The run fails before collection if the deployed
`/knowledge/chunks` no longer contains a required key or its content changed,
rather than silently evaluating against a stale KB revision. Both retrieval
layers use child chunks as their evaluation unit; the parent block is only a
prompt-assembly concern for `/chat`.

`/search`, `/chat`, and both evaluation runners use final Top-K 5. `/chat`
deduplicates those five child chunks by parent only while assembling the prompt;
its answer metrics remain independent:

- `FactualCorrectness` compares the answer with its reference answer.
- `AnswerRelevancy` checks whether the answer addresses the user request.
- `Faithfulness` runs only when `/chat` reports `knowledge_used=true`; it uses
  the case's gold evidence text, so it measures grounding without treating an
  independent `/search` response as the exact chat prompt.

`ContextPrecision` and `ContextRecall` are intentionally not averaged. The
preserved `/search` result is diagnostic data only. Every raw record includes
its `search.rerank_applied` and `search.rerank_providers` fields, so a report
can prove that its diagnostic search was actually reranked rather than merely
assigned an RRF fallback score.

The five unanswerable cases remain in the raw report for manual or custom
refusal-policy review; they are not averaged with answerable RAGAS scores.

Use the built-in summary to inspect both answer and retrieval aggregates:

```powershell
python -m tools.summarize_ragas_report data/eval/<run_id>.json
```

`AnswerRelevancy` uses the remote embedding endpoint only for the evaluation
container. The application container does not receive its embedding credential.

## Runtime Cost

RAGAS does not only compare strings. `FactualCorrectness` decomposes and checks
claims, `Faithfulness` validates claims against evidence when chat used the
knowledge base, and `AnswerRelevancy` calls both the judge model and the remote
embedding service. The runner therefore writes a checkpoint after every
completed record and uses `EVAL_MAX_CONCURRENCY=3` by default. It is safe to
raise this value gradually only after confirming the DeepSeek endpoint and
embedding endpoint do not rate-limit the run.

## Retrieval-Only Evaluation

`customer_service_search_v1.jsonl` is separate from the answer and workflow
datasets. It contains 70 static knowledge-retrieval cases and 10 OOD cases;
it never calls `/chat`, so workflow slot collection, order lookup, and answer
generation cannot change the result.

After a parent/child chunking change, replace the active knowledge base through
`POST /knowledge/replace` before running either evaluator. The request must
contain every current customer-service document with its filename stem as
`title`; this resets the active parent collection, child collection, and BM25
index together. The v2 Gold files correspond to 300-character parent chunks
and 100-character child chunks with 60/20-character overlap.

```powershell
cd F:\AI-agent\EchoMind
conda activate ragas
python -m evaluation.retrieval_runner `
  --base-url http://127.0.0.1:8000 `
  --dataset evaluation/datasets/customer_service_search_v1.jsonl `
  --gold evaluation/datasets/customer_service_search_gold_v2.json `
  --output-dir data/eval `
  --top-k 5 `
  --recall-k 20 `
  --require-rerank true `
  --required-rerank-provider tei
```

The report contains raw `Recall@20`, `Hit@20`, `MRR@20`, and `nDCG@20`, then
the same metrics for reranked Top-5 child chunks. The 10 OOD cases are excluded from those
averages and contribute only raw and reranked candidate-noise rates. The runner
validates every gold child chunk and its content hash before issuing searches.
