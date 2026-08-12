"""Regression check for the Supervisor's routing call.

Every specialist loops back to the Supervisor with its reply appended as the
last message. If the Supervisor ever passes that raw history straight to
Claude again, the call fails because the conversation ends on an assistant
turn (see agents/supervisor.py, which works around this with a trailing
synthetic human instruction). A single-intent query already exercises this
path; the additional checks below cover multi-specialist and multi-turn
conversations too.

Makes real calls to the Anthropic API -- requires ANTHROPIC_API_KEY. Run with:

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


CHECKS = [
    test_single_specialist_round_trip,
    test_multi_specialist_single_turn,
    test_followup_turn_after_specialist_reply,
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
