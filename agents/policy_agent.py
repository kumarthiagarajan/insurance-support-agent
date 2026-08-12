from langchain_anthropic import ChatAnthropic

from agents.utils import run_grounded_agent
from db.queries import get_customer, get_policies
from state import AgentState

SYSTEM_TEMPLATE = """You are the Policy specialist on an insurance support team.
Answer using ONLY the policy data below -- never invent coverage details, limits, or dates
that aren't listed. If the data doesn't cover the question, say so plainly and suggest
escalation.

Customer: {customer_name}
Policies on file:
{policy_context}
"""

_llm = ChatAnthropic(model="claude-opus-5", thinking={"type": "disabled"})


def policy_agent_node(state: AgentState) -> dict:
    customer = get_customer(state["customer_id"])
    policies = get_policies(state["customer_id"])

    if not policies:
        policy_context = "No policies found on file."
    else:
        policy_context = "\n".join(
            f"- {p['policy_id']} ({p['policy_type']}, status: {p['status']}): "
            f"{p['coverage_summary']}; premium ${p['premium_monthly']}/mo; "
            f"renews {p['renewal_date']}"
            for p in policies
        )

    system_prompt = SYSTEM_TEMPLATE.format(
        customer_name=customer["name"] if customer else "Unknown",
        policy_context=policy_context,
    )
    ai_message = run_grounded_agent(_llm, system_prompt, state["messages"], name="policy_agent")
    return {
        "messages": [ai_message],
        "handled": state.get("handled", []) + ["policy"],
    }
