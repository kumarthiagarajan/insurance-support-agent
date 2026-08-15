# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# .env must define ANTHROPIC_API_KEY (there is no .env.example checked in)

python db/seed_data.py     # (re)creates db/insurance.db from schema.sql, seeds it from db/Insurance_Support_Agent_Seed_Data.xlsx
python rag/ingest.py       # embeds FAQ docs into rag/chroma_store/ (ChromaDB)
python main.py             # interactive CLI chat loop
streamlit run app.py       # web UI (chat interface), same graph as main.py
python server.py           # FastAPI + static HTML/JS UI at http://127.0.0.1:8000

python tests/test_conversation_flow.py   # regression check, see below
```

There is no linter or build step, and no pytest dependency — `tests/test_conversation_flow.py`
is a standalone script (plain `assert`s, run directly with `python`). It makes real
`ANTHROPIC_API_KEY` calls through the full graph to check that multi-specialist and multi-turn
conversations don't crash the Supervisor's routing call — run it after touching
`agents/supervisor.py`, `agents/utils.py`, or `graph.py`. `db/insurance.db` and
`rag/chroma_store/` are gitignored, generated artifacts — regenerate them with the commands
above rather than editing them by hand. Re-running `db/seed_data.py` wipes and reseeds all
tables (`customers`, `policies`, `billing`, `claims`) from `db/Insurance_Support_Agent_Seed_Data.xlsx`,
which is checked into the repo and is the source of truth for seed data — edit the workbook
(one sheet per table, header row must match that table's column order in `schema.sql`) rather
than the generated `.db` file.

### Known gotcha: conversation must end on a user turn

Claude's API rejects any request whose message list ends on an assistant turn (treated as
disallowed "prefill"). Because specialists loop back to the Supervisor with their reply as the
last message, and multi-intent turns chain specialist → specialist, both
`agents/supervisor.py` (`_prompt`) and `agents/utils.py` (`run_grounded_agent`) append a
synthetic trailing `HumanMessage` before calling the LLM. If you change how either builds its
message list, keep this in place or the second-or-later LLM call in a turn will 400.

Try customer IDs `CUST001` or `CUST002` at the `main.py` prompt.

## Architecture

This is a LangGraph multi-agent system. A Supervisor node classifies each user message and
routes it to one specialist node at a time; every specialist loops back to the Supervisor
rather than ending the turn, so the graph iterates until the Supervisor is satisfied.

```
        supervisor  <--------------------------+
       /   |    |    \        \                |
   policy billing claims general  escalate      |
       \   |    |    /        /                |
        (loops back to supervisor) -------------+
```

- **`graph.py`** wires the `StateGraph`: `supervisor` is the entry point, conditional edges
  route to `policy` / `billing` / `claims` / `general` / `escalate` / `END`, and every
  specialist edges back to `supervisor`.
- **`state.py`** defines `AgentState`, a `TypedDict` threaded through every node: `messages`
  (append-only via `operator.add`), `customer_id`, `next` (routing decision), `handled` (list
  of specialists already consulted this turn — read by the Supervisor to avoid re-routing to
  one that already answered), and `iterations` (loop guard).
- **`agents/supervisor.py`** (`supervisor_node`) — the only node that calls
  `with_structured_output` (a `RouteDecision` Pydantic model) to pick the next node. Caps at
  `MAX_ITERATIONS = 4` per turn, then forces `escalate`, to prevent infinite supervisor loops.
- **`agents/{policy,billing,claims}_agent.py`** — each pulls that customer's rows from SQLite
  (`db/queries.py`) and passes them as literal context in the system prompt, instructed to
  answer strictly from that data (no invented coverage/amounts/dates). All follow the same
  shape: build a `*_context` string from DB rows → format `SYSTEM_TEMPLATE` → call
  `run_grounded_agent` (`agents/utils.py`), which prepends the system prompt to the message
  history and tags the reply with `name=` so `main.py` can attribute it in the transcript.
- **`agents/general_help_agent.py`** — same grounded pattern, but sources context from
  `rag/retriever.py` (ChromaDB similarity search) instead of SQLite, for account-agnostic
  questions (deductibles, grace periods, bundling, etc.).
- **`agents/escalation_agent.py`** — terminal node (edges straight to `END`, not back to
  `supervisor`). Builds a handoff summary from `state["handled"]` so a human doesn't need the
  customer to repeat context, and sets `next: "FINISH"`.
- **`rag/ingest.py`** — owns the FAQ corpus (`FAQ_DOCS`, currently a hardcoded list) and
  `get_chroma_collection()`, which both `ingest.py` and `retriever.py` use. Embeddings run
  locally via ChromaDB's bundled MiniLM model (`DefaultEmbeddingFunction`) — Anthropic has no
  embeddings endpoint, so `ANTHROPIC_API_KEY` is the only API key this project needs, and RAG
  ingestion/retrieval works offline.
- All LLM calls use `ChatAnthropic(model="claude-opus-5", thinking={"type": "disabled"})`,
  instantiated once per agent module at import time (`_llm` / `_llm | _router` module
  globals), not per-request.

### Adding a new specialist

Add `agents/<name>_agent.py` following the existing grounded-agent shape, register the node
and its back-edge to `supervisor` in `graph.py`, add it to `route_from_supervisor`'s edge map,
and add its description to the Supervisor's `SYSTEM_PROMPT` specialist list in
`agents/supervisor.py` so it's actually reachable.

## Scope note

Per `README.md`, this is an MVP scaffold, not a production system — no compliance/guardrail
gate on outbound responses, no PII redaction, no per-state regulatory rules, no audit
logging, no cross-session persistence, and no write-access flows (payments, filing a new
claim) yet. Keep this in mind when asked to harden or extend it.
