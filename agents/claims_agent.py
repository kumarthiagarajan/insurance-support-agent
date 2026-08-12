from langchain_anthropic import ChatAnthropic

from agents.utils import run_grounded_agent
from db.queries import get_claims, get_customer
from state import AgentState

SYSTEM_TEMPLATE = """You are the Claims specialist on an insurance support team.
Answer using ONLY the claims data below -- never invent statuses, dates, or amounts that
aren't listed. If the customer wants to file a new claim, walk them through what information
you'd need (date of loss, description, policy number) and let them know a claim will be
opened; don't invent a claim ID. If existing data doesn't cover the question, say so plainly
and suggest escalation.

Customer: {customer_name}
Claims on file:
{claims_context}
"""

_llm = ChatAnthropic(model="claude-opus-5", thinking={"type": "disabled"})


def claims_agent_node(state: AgentState) -> dict:
    customer = get_customer(state["customer_id"])
    claims = get_claims(state["customer_id"])

    if not claims:
        claims_context = "No claims found on file."
    else:
        claims_context = "\n".join(
            f"- {c['claim_id']} (policy {c['policy_id']}, {c['claim_type']}): status: "
            f"{c['status']}, filed {c['filed_date']}, adjuster: {c['adjuster_name']}, "
            f"description: {c['description']}"
            for c in claims
        )

    system_prompt = SYSTEM_TEMPLATE.format(
        customer_name=customer["name"] if customer else "Unknown",
        claims_context=claims_context,
    )
    ai_message = run_grounded_agent(_llm, system_prompt, state["messages"], name="claims_agent")
    return {
        "messages": [ai_message],
        "handled": state.get("handled", []) + ["claims"],
    }
