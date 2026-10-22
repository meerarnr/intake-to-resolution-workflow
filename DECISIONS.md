# Design Decisions

## Phase 1
- **Legal-term scan takes `ticket_text` as an engine argument.** `TicketSchema` has no raw text, and keyword checks must not depend on LLM output. Rejected: scanning `extracted_entities` (LLM-controlled, could omit the term).
- **Known resolution = `extracted_entities["resolution_code"]` checked against codes from `known_resolutions.json`.** Keeps the schema to the specified fields. Codes are passed into the engine so it stays I/O-free.
- **Keyword match is whole-word, case-insensitive.** Plain substring would flag "sue" in "issue". Config also lists inflections ("sued", "suing").
- **VIP_TIER, LEGAL_TERMS and LOW_CONFIDENCE are separate triggers recorded alongside Rules 1-3.** Rule 3 fires only when no other trigger did. Forced-human outcomes allow only `REQUIRES_HUMAN_APPROVAL`; otherwise Tier 2 reassignment is also allowed.
- **Required approver: MANAGER for Rule 1 or 2, SUPPORT_AGENT for other human paths, NONE for auto-resolve.** Spec says only "human escalation" for Rule 1; manager is my assumption, easy to change.
- **Audit immutability enforced via SQLAlchemy mapper events plus a `do_orm_execute` hook for bulk UPDATE/DELETE.** SQLite triggers rejected for now: spec asks for the ORM layer, and the hash chain covers raw-SQL tampering (detected, not prevented). Tail truncation is not detectable without an external anchor.
- **Audit timestamp and payload stored as strings exactly as hashed.** Avoids datetime/JSON round-trip drift breaking verification. `entry_hash` excludes the autoincrement id; ordering and `prev_hash` link rows.
- **`append_audit` flushes, never commits.** Audit row shares the caller's transaction with the state change it records.
- **Refund stored as str on `Ticket`, `Decimal` in schemas.** SQLite has no exact decimal type. Refund limited to 2 decimal places.

## Phase 2
- **LangGraph pause = `interrupt()` in `human_wait`, resume = `Command(resume=...)` on the same `thread_id` (`ticket-<id>`) with `SqliteSaver`.** Node bodies re-run from the top on resume, so `human_wait` does no writes before `interrupt()`; the pause is audited and the status set in `decide`.
- **Only the mock provider was implemented in this phase** (real providers added in Phase 5). The `LLMProvider` interface was designed so they plug in without touching the workflow.
- **Mock fixtures are keyed by exact ticket text; `attempt` index selects the response** so retry behavior is deterministic. Missing fixture = failed call, which takes the normal fail-safe path.
- **Customer tier uses the ticket's intake `customer_id`, not the LLM-extracted one.** The extracted value is overwritten and a mismatch is audited (`CUSTOMER_ID_MISMATCH`).
- **KB grounding gate is deterministic code**: the draft prompt gets only the section for the ticket's category; no section = stub draft, no LLM call, route to human. The prompt also instructs "use only this excerpt", but the code gate is what is enforced.
- **Any error (extraction, draft, KB gap) forces `REQUIRES_HUMAN_APPROVAL`, even if the engine would allow auto-resolve.** Approving such a ticket requires an `edited_draft` so a placeholder is never sent.
- **Non-auto actions (`REQUIRES_HUMAN_APPROVAL`, `REASSIGN_TIER_2`) both pause at `human_wait`.** There is no Tier 2 queue in scope. Approve = reply approved and ticket RESOLVED.
- **Reject always goes AWAITING -> REJECTED first**, then PROCESSING (override) or CLOSED (manual close). Gives REJECTED a real meaning and an audited trail.
- **Override covers classification fields only (category, urgency, sentiment, entities, refund), not a forced route.** The engine re-evaluates, so a human cannot push a ticket past policy by override. Confidence is set to 1.0 since the human vouches for it; rejected alternative: leave it, which would loop on LOW_CONFIDENCE forever.
- **Approver privilege is checked in code** (`MANAGER` > `SUPPORT_AGENT`), both in `resume_ticket` (before touching the graph) and in the node.
- **`structured_call` catches any `Exception` from the provider/validation** as a failed attempt: fail-safe beats surfacing a network error mid-graph. Errors are stored (truncated) in the audit row.
- **Node writes are not idempotent if a node crashes mid-way and is retried.** Acceptable for now; each node commits once at its end.

## Phase 3
- **Status codes: 404 missing ticket, 409 not awaiting action, 403 insufficient approver role, 422 bad input.** 409 is checked against both the DB status and the LangGraph pause state, before the graph is touched. Workflow `PermissionError` maps to 403, `ValidationError`/`ValueError` (e.g. fail-safe ticket approved without `edited_draft`) to 422.
- **`reviewer` and `reason` are required non-empty fields on both approve and reject** (`extra="forbid"` rejects unknown fields). They flow into `HumanDecision` and so into the audit row; the API writes no audit rows itself.
- **`approver_role` is a self-asserted request field, default `SUPPORT_AGENT`.** The project has no auth by spec, so the privilege check is real in code but not identity-backed. Rejected: inferring role from reviewer name (equally spoofable, more magic).
- **Reject takes `action: override|close` and the model validator enforces consistency** (override needs at least one field, close forbids one). Catches malformed requests at the edge as 422 instead of inside the graph.
- **One process-wide lock serializes submit/approve/reject.** Concurrent runs would read the same `prev_hash` and fork the audit chain. Costs throughput; rejected alternative: DB-level sequence/retry, too much machinery for SQLite.
- **Ticket detail reads `final_action`, policy decision and draft from the LangGraph checkpoint**, not DB columns. Avoids duplicating state and schema drift; the cost is that read endpoints touch the checkpointer.
- **`/tickets/{id}/audit` returns 404 for an unknown ticket instead of an empty list.** `/audit/verify` is global (the chain spans all tickets), and returns valid with `checked=0` when empty.

## Phase 4
- **Dashboard uses only HTTP (`httpx`) against `API_URL`; it reads `data/fixtures/` only to list samples.** No DB or workflow imports, so it can run against a remote API. Samples are submitted through `POST /tickets` like any ticket.
- **SLA label is display-only, from the API's `policy_decision.sla_deadline`.** The deadline is computed by the engine; the UI just compares it to now. Closed tickets use `updated_at` as the finish time (no resolved-at field exists). The overview fetches each ticket's detail (N+1 calls), fine at demo scale; rejected: adding a field to the list endpoint.
- **Timeline is the ticket's audit trail rendered in order**, not a separate node log, so it can never disagree with the audit log.
- **Approve sends `edited_draft` only when the text differs from the LLM draft.** Reject-override sends only the fields the reviewer changed (`(unchanged)` = omitted).
- **`approver_role` is a visible selector**, matching the API's self-asserted role; no auth by spec.
- **Streamlit `AppTest` drives the dashboard in tests with `httpx.Client` patched to the in-process API `TestClient`.** Real round trip, no server needed.

## Phase 5
- **Official `anthropic` and `openai` SDKs, lazily imported; Ollama via its REST `/api/chat` with `httpx`.** Mock mode needs none of them installed. Rejected: raw HTTP for the two hosted providers (reimplements retries, auth, typed errors).
- **No default model anywhere.** A real provider without `LLM_MODEL` raises at startup. Rejected: a baked-in default, which silently goes stale and violates "never hardcoded".
- **Providers return raw text; validation and the single retry stay in `structured_call`.** One code path for all providers, so fail-safe behavior is identical and tested once. Providers only strip a markdown fence and turn refusals/empty completions into `LLMProviderError`.
- **No sampling parameters, thinking config or forced tool use.** Newer models reject some of them; JSON-only prompting plus Pydantic validation works across providers. OpenAI and Ollama also request JSON mode.
- **Provider adapters are tested with fake clients only.** Never verified against live APIs here; treat first real run as unverified.

## Known limitations
- **Real providers untested live.** Prompts were written against the mock fixtures; real models may need prompt tuning, and extraction quality is unmeasured (no eval set).
- **No auth.** `reviewer` and `approver_role` are self-asserted; the privilege check is real but not identity-backed. Anyone who can reach the API can claim `MANAGER`.
- **Single approval level.** Refunds over $250 need one manager; there is no higher tier for very large refunds, and `REASSIGN_TIER_2` has no real Tier 2 queue (it pauses like any human action).
- **Tamper-evident, not tamper-proof.** Raw-SQL edits are detected, but truncating the tail of the log is not (no external anchor), and someone with DB write access can recompute the whole chain.
- **One process, one lock.** All workflow runs are serialized; SQLite and the in-process lock do not scale across workers.
- **Crash mid-node is not idempotent**, and there is no ticket retry/resume endpoint for a half-run ticket.
- **KB is plain markdown, one section per category.** No retrieval, so drafts can only be as good as that section; a missing section routes to a human.
- **Legal and resolution-code matching is keyword/exact.** Misspellings and paraphrases ("my solicitor") are missed.
- **Dashboard polls on demand** (manual refresh), makes N+1 API calls on the overview, and SLA "met" for closed tickets uses `updated_at`, not a real resolution time.
- **Sentiment/urgency come from the LLM**, so Rule 1 and SLA depend on its accuracy; only VIP tier is independent of it.

## Future enhancements
- Auth and RBAC (real reviewer identity, roles from a user store, not the request body); then drop the self-asserted role.
- Multi-tier approval: a second approver above a higher refund level (e.g. director over $5,000), configured in `policy_rules.json`.
- A Tier 2 queue with its own owner and SLA.
- Anchor the audit chain externally (periodic signed hash to object storage) to detect truncation; DB triggers to block raw UPDATE/DELETE.
- An eval set for extraction accuracy and prompt versions recorded in the audit log.
- Postgres plus a proper job queue, replacing the global lock; live updates in the dashboard.
- Semantic KB retrieval and a paraphrase-tolerant legal-term classifier (flag-only, so code still enforces the rule).
