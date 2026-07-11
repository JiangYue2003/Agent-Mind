# EchoMind Hot-Reload Skill Design

## Scope

Add engineering support for declarative Agent Skills. A Skill can change an
Agent's instructions, response constraints, and access to existing MCP tools
without restarting the FastAPI process. It cannot upload or execute arbitrary
Python code.

The feature extends the current `/chat` path instead of replacing it:

```text
/chat -> intent recognition -> workflow decision and plan -> SkillRuntime
      -> selected Skill instructions -> existing AgentOrchestrator -> response
```

`MCPToolManager` remains the only execution boundary for business tools. A
Skill grants a subset of already registered tool names; it does not register a
new handler or bypass the tool schema, timeout, circuit breaker, or workflow
slot checks.

## Package Contract

Every package is a ZIP archive or a directory containing exactly these
user-authored files:

```text
skill.json
prompt.md
```

`skill.json` is the source of truth and contains:

- `id`, `name`, and semantic `version`;
- `description` for diagnostics and catalog display;
- `targets.agent_roles`, `targets.intent_categories`, and
  `targets.goals` for deterministic activation;
- `allowed_tools`, constrained to the active MCP tool catalog;
- `prompt_file`, `priority`, and bounded response/tool constraints.

`prompt.md` contains the instruction fragment appended to the selected child
Agent's base system prompt. The runtime treats it as data, not source code.
Skill IDs, archive paths, file sizes, JSON shape, duplicate versions, tool
bindings, and prompt size are validated before a package can become active.

## Sources And Lifecycle

There are two trusted sources:

```text
skills/catalog/<skill-id>/<version>/       Git-managed, read-only to the app
data/skills/published/<skill-id>/<version>/ Admin-published packages
data/skills/drafts/<draft-id>/              Uploaded, inactive packages
```

The catalog keeps an immutable `SkillSnapshot`: a validated mapping from Skill
ID to one active version and parsed manifest. Every `/chat` request captures a
snapshot before selection, so a reload cannot change the Skill set halfway
through a response. A refresh builds a complete replacement snapshot and
swaps it only after all packages validate. If a new package is invalid, the
last valid snapshot stays active and the failure is observable.

`SkillRuntime` runs a lightweight polling task in FastAPI lifespan. It compares
the sources' file fingerprints at a short configurable interval, refreshes on
change, and stops with the application. Publication also calls refresh
synchronously, so successful uploads take effect in that request rather than
waiting for the next poll.

## Activation And Agent Integration

After the existing workflow plan chooses the primary/supporting Agent roles,
`SkillRuntime` selects packages whose manifest targets match the current:

- Agent role;
- recognized intent category;
- workflow goal; and
- planned tool set, when the package declares one.

The selection is declarative and uses no new customer-service keyword rules.
The resulting instructions and allowed tool names are passed in the existing
orchestrator context. `BaseAgent` composes them with its fixed base system
prompt for that invocation. Supporting Agents receive only Skills targeted at
their own role. The static Agent pool, bounded workflow, evidence flow, and
MCP executor remain intact.

## Upload, Review, And Development Mode

Administrative APIs use `X-Skill-Admin-Token`; application startup fails an
admin request closed when `SKILL_ADMIN_TOKEN` is not configured. The APIs are:

- `POST /admin/skills/upload`: validate and create a draft;
- `POST /admin/skills/{draft_id}/publish`: publish a validated draft;
- `GET /admin/skills`: inspect drafts, sources, active versions, and errors;
- `POST /admin/skills/reload`: request a catalog refresh.

The review rule is configuration, not a Git-hosting change:

- Production defaults to `SKILL_REVIEW_REQUIRED=true`. Uploads remain drafts
  until an administrator publishes them.
- When `APP_ENV=development`, the default is `SKILL_REVIEW_REQUIRED=false`.
  A valid upload is published and reloaded in the upload request. This is the
  requested local "plug in and use" path.
- A developer can also add or modify a valid directory under `skills/catalog`;
  the polling runtime discovers it with no API call or process restart.

Development mode removes the application-level review step only. It does not
allow executable package code, disable manifest validation, or remove the
administrative token.

## Failure Handling And Observability

The runtime exposes catalog generation, active package versions, source,
last refresh time, and validation failures. It records trace metadata for the
snapshot generation and selected Skill IDs, but never writes complete prompt
content into traces. Upload errors are explicit and leave active packages
unchanged. If the poller fails, it logs the fault and retries at the next
interval without stopping `/chat`.

## Test Plan

Targeted unit and API tests will verify:

1. Manifest, archive, path traversal, prompt-size, duplicate, and tool-binding
   validation.
2. Atomic snapshot swap, invalid-refresh rollback, and request snapshot
   stability.
3. Production upload remains a draft; development upload auto-publishes and is
   visible immediately.
4. File-system catalog changes reload without recreating the runtime.
5. Role/intent/goal/tool matching injects only the intended Skills into primary
   and supporting Agents.
6. Existing `/chat`, MCP tool behavior, and workflow tests remain green.

## Non-Goals

- Arbitrary Python/plugin execution from a Skill archive.
- Automatic discovery and execution of unknown MCP tools.
- Changing GitHub/GitLab branch-protection or pull-request settings.
- Replacing the current bounded workflow with an open-ended Agent loop.
