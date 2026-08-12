from langchain_anthropic import ChatAnthropic

from agents.utils import run_grounded_agent
from db.queries import get_billing, get_customer
from state import AgentState

SYSTEM_TEMPLATE = """You are the Billing specialist on an insurance support team.
Answer using ONLY the billing data below -- never invent amounts, dates, or payment methods
that aren't listed. If the data doesn't cover the question, say so plainly and suggest
escalation.

Customer: {customer_name}
Billing records on file:
{billing_context}
"""

_llm = ChatAnthropic(model="claude-opus-5", thinking={"type": "disabled"})


def billing_agent_node(state: AgentState) -> dict:
    customer = get_customer(state["customer_id"])
    invoices = get_billing(state["customer_id"])

    if not invoices:
        billing_context = "No billing records found."
    else:
        billing_context = "\n".join(
            f"- {inv['invoice_id']} (policy {inv['policy_id']}): ${inv['amount_due']} due "
            f"{inv['due_date']}, status: {inv['status']}, payment method: "
            f"{inv['payment_method']}"
            for inv in invoices
        )

    system_prompt = SYSTEM_TEMPLATE.format(
        customer_name=customer["name"] if customer else "Unknown",
        billing_context=billing_context,
    )
    ai_message = run_grounded_agent(_llm, system_prompt, state["messages"], name="billing_agent")
    return {
        "messages": [ai_message],
        "handled": state.get("handled", []) + ["billing"],
    }
