"""Regression checks for the Supervisor's routing call.

Every specialist loops back to the Supervisor with its reply appended as the
last message. If the Supervisor ever passes that raw history straight to
Claude again, the call fails because the conversation ends on an assistant
turn (see agents/supervisor.py, which works around this with a trailing
synthetic human instruction). A single-intent query already exercises this
path; the additional checks below cover multi-specialist and multi-turn
conversations too, plus the deterministic guard that stops a specialist's
own reply from escalating to a human within the same turn.

Most checks make real calls to the Anthropic API -- requires
ANTHROPIC_API_KEY. Run with:

    python tests/test_conversation_flow.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY before running this check.")

from langchain_core.messages import HumanMessage

import agents.supervisor as supervisor
from graph import build_graph


def fresh_state(customer_id: str) -> dict:
    return {
        "messages": [],
        "customer_id": customer_id,
        "next": "",
        "handled": [],
        "iterations": 0,
    }


def run_turn(app, state: dict, text: str) -> dict:
    state["messages"].append(HumanMessage(content=text))
    state["handled"] = []
    state["iterations"] = 0
    return app.invoke(state)


def test_single_specialist_round_trip():
    """Supervisor -> billing -> Supervisor (FINISH) must not error on the second call."""
    app = build_graph()
    state = run_turn(app, fresh_state("CUST001"), "Is my payment overdue?")
    assert any(m.type == "ai" for m in state["messages"]), "expected an AI reply"
    assert "billing" in state["handled"], state["handled"]


def test_multi_specialist_single_turn():
    """A multi-intent message should route through two specialists in one turn
    without the Supervisor crashing between them."""
    app = build_graph()
    state = run_turn(
        app,
        fresh_state("CUST001"),
        "Why did my premium go up and what's happening with my claim?",
    )
    assert len(state["handled"]) >= 2, f"expected 2+ specialists, got {state['handled']}"


def test_followup_turn_after_specialist_reply():
    """A second user turn, after a specialist already replied last turn, must
    not crash the Supervisor's routing call either."""
    app = build_graph()
    state = fresh_state("CUST001")
    state = run_turn(app, state, "Is my payment overdue?")
    state = run_turn(app, state, "What about my auto policy deductible?")
    assert "policy" in state["handled"], state["handled"]


def test_no_same_turn_escalate_after_specialist_reply():
    """Deterministic guard in supervisor_node: even if the router LLM decides
    "escalate" right after a specialist replied this turn, it must be forced
    to FINISH so the customer sees the specialist's reply and can confirm
    first. A fresh turn (handled reset) must still allow escalate through
    immediately, e.g. for an explicit "get me a human" request. No API call
    is made here -- the router is monkeypatched to isolate the guard itself
    from LLM variance.
    """

    class _FakeDecision:
        next = "escalate"
        reasoning = "test"

    class _FakeChain:
        def invoke(self, _input):
            return _FakeDecision()

    original_chain = supervisor._chain
    supervisor._chain = _FakeChain()
    try:
        mid_turn_state = {
            "messages": [HumanMessage(content="why did my premium go up?")],
            "customer_id": "CUST001",
            "next": "",
            "handled": ["billing"],
            "iterations": 1,
        }
        mid_turn_result = supervisor.supervisor_node(mid_turn_state)

        fresh_turn_state = {
            "messages": [HumanMessage(content="please connect me to a human")],
            "customer_id": "CUST001",
            "next": "",
            "handled": [],
            "iterations": 0,
        }
        fresh_turn_result = supervisor.supervisor_node(fresh_turn_state)
    finally:
        supervisor._chain = original_chain

    assert mid_turn_result["next"] == "FINISH", mid_turn_result
    assert fresh_turn_result["next"] == "escalate", fresh_turn_result


CHECKS = [
    test_single_specialist_round_trip,
    test_multi_specialist_single_turn,
    test_followup_turn_after_specialist_reply,
    test_no_same_turn_escalate_after_specialist_reply,
]


def main():
    failures = []
    for check in CHECKS:
        name = check.__name__
        try:
            check()
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")

    if failures:
        raise SystemExit(f"\n{len(failures)}/{len(CHECKS)} checks failed")
    print(f"\nAll {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
