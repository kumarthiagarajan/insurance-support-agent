# Insurance Support Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Multi-agent insurance customer support system: a Supervisor agent (LangGraph) routes
each query to a Policy, Billing, Claims, or General Help specialist, or escalates to a
human. Specialists ground their answers in real customer data (SQLite) or a knowledge
base (ChromaDB RAG) instead of guessing. All agents run on Claude (Opus 5) via
`langchain-anthropic`; RAG retrieval embeds locally (ChromaDB's bundled MiniLM model),
since Anthropic has no embeddings endpoint — so the only API key needed is
`ANTHROPIC_API_KEY`.

## Architecture

```
        supervisor  <--------------------------+
       /   |    |    \        \                |
   policy billing claims general  escalate      |
       \   |    |    /        /                |
        (loops back to supervisor) -------------+
```

- **supervisor** (`agents/supervisor.py`) — classifies intent, routes to a specialist,
  or FINISHes the turn once everything the customer asked has been addressed.
- **policy / billing / claims** (`agents/*_agent.py`) — pull that customer's records
  from SQLite (`db/insurance.db`) and answer strictly from that data.
- **general** (`agents/general_help_agent.py`) — retrieves relevant FAQ chunks from
  ChromaDB and answers from those, not from the model's general knowledge.
- **escalate** (`agents/escalation_agent.py`) — builds a structured handoff summary
  (customer, specialists already consulted, reason) so a human doesn't need the
  customer to repeat themselves.

State (`state.py`) is a shared `AgentState` dict carrying the message history,
customer id, and which specialists have already run this turn — this is what lets a
single message like "why did my premium go up and what's my claim status" get routed
to two specialists in sequence and merged into one conversation, without either one
re-asking for the policy number.

## Setup

```bash
cd insurance-support-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
# optionally also set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL for
# tracing (see Observability below) -- the app runs fine without them

python db/seed_data.py     # creates + seeds db/insurance.db
python rag/ingest.py       # embeds FAQ docs into rag/chroma_store/
python main.py             # interactive CLI chat loop
streamlit run app.py       # or: Streamlit web UI at http://localhost:8501
python server.py           # or: FastAPI + static HTML/JS UI at http://127.0.0.1:8000
python gradio_app.py       # or: Gradio chat UI at http://127.0.0.1:7860
```

Try customer IDs `CUST001` or `CUST002` (seeded with sample policies, billing, and
claims). Example prompts:

- "What's my auto policy deductible?"
- "Is my payment overdue?"
- "What's the status of my claim?"
- "Why did my premium go up and what's happening with my claim?" (multi-intent)
- "I want to speak to a person" (escalation)

## Observability

Tracing is optional and off by default. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
and `LANGFUSE_BASE_URL` in `.env` (get keys from a free [Langfuse Cloud](https://langfuse.com/cloud)
project, or a self-hosted instance, under Settings > API Keys) and every conversation turn
across all four UIs is traced to Langfuse: one trace per turn, nested spans per specialist
call, grouped into Sessions by conversation, tagged by customer and by which UI produced it.
Emails, phone numbers, and card-ending references are redacted before export. See
`tracing.py`.

## Notes on this scaffold

This is an MVP, not a production system. Before going further, see the earlier plan
for what production-hardening adds on top of this: a compliance/guardrail gate on
outbound responses, PII redaction, per-state regulatory rules, audit logging, session
persistence across channels (Redis), and write-access flows (payments, filing a new
claim) gated separately from these read-only specialists.
