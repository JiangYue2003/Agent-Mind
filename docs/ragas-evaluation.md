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

Then build and run the on-demand evaluation service:

```powershell
docker compose --profile eval build ragas-eval
docker compose --profile eval run --rm ragas-eval
```

Optional environment variables are `EVAL_RUN_ID`, `EVAL_TOP_K`,
`EVAL_RECALL_K`, `EVAL_REQUIRE_RERANK`, `EVAL_REQUIRED_RERANK_PROVIDER`,
`EVAL_MAX_CONCURRENCY`, `EVAL_DATASET_PATH`, `EVAL_WORKFLOW_DATASET`, and
`EVAL_OUTPUT_DIR`. The defaults recall 12 candidates before the local GPU
reranker produces Top-K 3, require the TEI-compatible `tei` protocol marker
for answerable knowledge cases, run at most three RAGAS
metric tasks at once, and use the customer-service dataset plus workflow cases.

Query rewrites each perform hybrid recall, then their candidates are merged
and deduplicated before one global TEI-compatible rerank. This prevents one user request
from overwhelming the reranker with one rerank request per rewrite.
The host-native `local_reranker` service scores the 12 highest RRF-scored
candidates in one GPU batch before reranking to Top-K 3. Start it before the
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

The current run focuses on answer quality rather than retrieval ranking:

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
