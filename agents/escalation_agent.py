from langchain_core.messages import AIMessage

from db.queries import get_customer
from state import AgentState


def escalation_node(state: AgentState) -> dict:
    customer = get_customer(state["customer_id"])
    name = customer["name"] if customer else state["customer_id"]
    handled = ", ".join(state.get("handled", [])) or "none"

    summary = (
        "I'm connecting you with a member of our support team who can help further. "
        "Here's a summary for them:\n\n"
        f"- Customer: {name} ({state['customer_id']})\n"
        f"- Specialists already consulted this session: {handled}\n"
        "- Reason for escalation: request requires human review\n\n"
        "A representative will have this context and won't need you to repeat yourself."
    )
    return {
        "messages": [AIMessage(content=summary, name="escalation_agent")],
        "next": "FINISH",
    }
