# EchoMind Python Chat Streaming Design

## Scope

Upgrade the Python chat path from one-shot JSON responses to streamed chat
updates that expose both user-facing workflow stages and incremental answer
text. The change applies only to the Python backend in this repository and the
shared Vue frontend panel when the selected backend is `python`.

The existing synchronous contract remains available:

```text
POST /chat -> complete JSON response
```

The new contract is additive:

```text
POST /chat/stream -> text/event-stream -> stage events + answer deltas
```

This design does not change the bounded workflow model, routing logic, or the
business-tool surface. It changes transport and response assembly only.

## Goals

The streamed experience must let the frontend show:

- a quick "request accepted" transition;
- a small, stable set of user-facing stage updates;
- incremental assistant text while the model is generating; and
- a smoother typewriter effect than raw network chunk timing.

The design must preserve:

- the current `/chat` behavior for fallback and compatibility;
- the current `message`, `user_id`, `conv_id`, and `operation_id` request
  contract;
- the current workflow semantics for clarify / knowledge / live-record /
  action flows; and
- conversation-memory correctness, meaning assistant memory is written only
  after a successful final response is fully assembled.

## Transport Choice

Use `POST /chat/stream` with an SSE-formatted response body served through
FastAPI `StreamingResponse`.

This is the right fit for the current project shape:

- the chat request already needs a POST body, so native `EventSource` is a poor
  fit;
- the interaction is still server-to-client only, so WebSocket adds complexity
  without current product value; and
- SSE event framing gives cleaner semantics than ad hoc chunked text or NDJSON
  for stage changes, answer deltas, completion, and errors.

The frontend will use `fetch()` plus `ReadableStream` parsing rather than the
browser `EventSource` API. This keeps the POST request shape and still consumes
SSE framing cleanly.

## API Contract

### Request

`POST /chat/stream` accepts the same JSON body as `POST /chat`:

```json
{
  "message": "我想申请退款",
  "user_id": "u1001",
  "conv_id": "optional-conversation-id",
  "operation_id": "optional-operation-id"
}
```

### Response

Response headers:

- `Content-Type: text/event-stream; charset=utf-8`
- `Cache-Control: no-cache`
- `Connection: keep-alive`

Each event is emitted as standard SSE framing:

```text
event: stage
data: {"type":"stage.started","stage":"planning","label":"正在规划处理流程","seq":3}

```

The backend exposes only a small stable event vocabulary:

- `run.started`
- `stage.started`
- `stage.completed`
- `answer.delta`
- `answer.completed`
- `run.completed`
- `error`

Suggested `stage` identifiers are user-facing and intentionally coarser than
internal trace stages:

- `understanding`
- `planning`
- `retrieving`
- `answering`

Representative payloads:

```json
{"type":"run.started","conversation_id":"...","request_id":"..."}
{"type":"stage.started","stage":"understanding","label":"正在理解问题","seq":1}
{"type":"stage.completed","stage":"understanding","seq":1}
{"type":"answer.delta","delta":"您好，"}
{"type":"answer.completed","response":"完整最终回答","intent":"billing","agent_type":"billing","knowledge_used":true,"escalated":false}
{"type":"run.completed","conversation_id":"...","latency_ms":842.6,"trace_id":"..."}
{"type":"error","message":"服务暂时不可用，请稍后重试。","retryable":false}
```

`answer.completed` includes the normalized final answer metadata so the
frontend does not need a second summary request.

## Backend Architecture

`api/main.py` remains the control point. The design introduces a shared chat
pipeline that can drive either:

- the current buffered `/chat` JSON response path; or
- the new `/chat/stream` SSE path.

The refactor boundary is:

```text
request -> shared pipeline -> events + final result
                       |-> /chat       -> buffer -> ChatResponse
                       |-> /chat/stream -> forward -> SSE
```

The shared pipeline must not duplicate workflow logic. It should reuse the
existing sequence:

```text
memory.read
-> intent.recognize
-> workflow.decide / slot_check / planner / state_transition
-> optional tool / knowledge / handoff execution
-> orchestrator answer generation
-> memory.write
```

### Stage Mapping

Internal steps are grouped into four user-visible phases:

- `understanding`
  - memory read
  - intent recognition
- `planning`
  - workflow decision
  - slot assessment
  - planner
  - state transition
- `retrieving`
  - knowledge search or live tool execution when actually used
- `answering`
  - assistant answer generation

If a request goes directly to clarify, the pipeline still emits
`understanding`, `planning`, then a short `answering` phase for the clarify
prompt. If no retrieval happens, the `retrieving` phase is omitted.

### Streaming The Final Answer

The current orchestrator path uses a non-streaming Anthropic call. The design
adds a streaming answer path only for `/chat/stream`.

For the final answer stage:

- use the async Anthropic message streaming API;
- forward text deltas as `answer.delta` events;
- accumulate the full text server-side while forwarding deltas; and
- finalize metadata after the stream completes.

The streamed path must preserve current prompt construction, agent selection,
skill injection, and bounded workflow rules. The model call changes from
buffered completion to streamed completion; the orchestration contract does not
become an open-ended agent loop.

### Memory And Finalization Rules

Assistant memory is written only after the final full answer has been
assembled successfully. This avoids storing partial text when the stream is
interrupted.

Finalization order:

1. emit all answer deltas;
2. finish model stream and build the final answer string;
3. write user and assistant messages to memory;
4. schedule profile update;
5. emit `answer.completed`;
6. emit `run.completed`.

If failure happens before step 3, no assistant memory is persisted for that
turn. If the user message has not yet been written, the pipeline keeps current
behavior and writes neither side.

### Error Handling

There are three error classes:

1. pre-answer failure  
   The backend emits `error` and terminates the stream. No assistant message is
   persisted.
2. mid-answer stream failure  
   The backend emits `error` and terminates the stream. Partial UI text may
   remain visible, but no assistant message is persisted.
3. client disconnect  
   The backend stops streaming and cancels remaining work as early as practical
   without forcing extra writes.

The error payload must be concise and user-safe. Raw internal trace details are
kept in logs and trace storage, not in SSE payloads.

## Frontend Architecture

The frontend currently submits one buffered chat request and appends the
assistant message only after completion. For Python backend mode, it will
switch to a streamed request path.

The Java backend path is unchanged and keeps using buffered `/chat`.

### Frontend Data Flow

For Python backend:

```text
submit -> insert user message
      -> insert assistant placeholder
      -> POST /chat/stream
      -> parse SSE frames
      -> update stage status + buffer incoming text
      -> smooth-render visible text
      -> finalize metadata
```

The assistant placeholder stores at least:

- visible text;
- raw buffered incoming text;
- current stage label;
- completion/error state; and
- response metadata once available.

### Smooth Typewriter Effect

The UI effect is not fake full-response playback. It is real streamed text plus
frontend smoothing.

The frontend keeps two strings:

- `incomingText`: exact text received from `answer.delta`
- `displayText`: text currently rendered

A lightweight scheduler moves text from `incomingText` into `displayText` at a
stable cadence, for example every 20-40 ms with a bounded character budget per
tick. This hides irregular network chunking and produces a more natural typing
effect.

When `answer.completed` arrives, the renderer flushes any remaining buffered
text immediately so the UI does not lag behind the actual completed response.

### SSE Parsing

Use `fetch()` and consume `response.body` as a stream. The parser reads UTF-8
chunks, reconstructs SSE frames split by blank lines, extracts `event:` and
`data:` fields, then routes parsed JSON payloads by type.

The parser must tolerate:

- chunk boundaries inside UTF-8 characters;
- chunk boundaries inside an SSE frame;
- multiple SSE events in a single network chunk; and
- trailing whitespace / empty lines.

## UI Semantics

The UI should keep the agent interaction simple:

- show only one current stage label at a time;
- replace low-level terms with user-facing language;
- keep metadata such as `intent`, `agent_type`, `knowledge_used`, and
  `escalated` as a compact footer after completion; and
- mark interrupted streams clearly instead of silently freezing.

Recommended stage labels:

- `understanding` -> `正在理解问题`
- `planning` -> `正在规划处理流程`
- `retrieving` -> `正在查询相关信息`
- `answering` -> `正在生成回答`

## Backward Compatibility And Rollout

Compatibility rules:

- existing `POST /chat` remains unchanged;
- existing Java frontend path remains unchanged;
- existing Python monitoring and health endpoints remain unchanged.

Rollout strategy:

1. add backend `/chat/stream` behind the current Python API base path;
2. keep `/chat` as fallback;
3. update frontend so only `settings.backend === 'python'` uses stream mode;
4. on stream setup failure, fall back to current buffered request for that
   message.

This keeps the new transport low-risk and reversible.

## Test Plan

### Backend

Targeted unit tests should cover:

1. SSE event encoding and framing.
2. Stage emission order for:
   - clarify-only flow,
   - knowledge flow,
   - live-tool flow.
3. Streamed answer aggregation from model deltas.
4. No assistant memory write on pre-answer or mid-answer failure.
5. `answer.completed` / `run.completed` payload correctness.
6. Existing `/chat` tests remain green.

### Frontend

Targeted tests should cover:

1. SSE parser behavior across fragmented chunks.
2. Typewriter buffer draining logic.
3. Correct placeholder lifecycle from submit to completion.
4. Python streamed path and Java buffered path selection.
5. Fallback to buffered `/chat` on stream startup failure.

### Manual Verification

Manual verification should confirm:

- stage labels appear in the expected order;
- answer text grows incrementally rather than waiting for completion;
- the final text matches the buffered response for the same prompt class;
- interrupted streams are surfaced clearly; and
- conversation ID, trace ID, and completion metadata still reconcile.

## Non-Goals

- Changing the Java backend.
- Replacing the bounded workflow with WebSocket-based bidirectional control.
- Streaming every internal trace event to the UI.
- Adding tool approval UX, mid-flight cancellation, or resumable execution.
- Reworking the entire frontend chat layout beyond the streamed-message
  behavior needed here.
