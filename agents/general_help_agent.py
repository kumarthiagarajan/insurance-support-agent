from langchain_anthropic import ChatAnthropic

from agents.utils import last_human_message, run_grounded_agent
from rag.retriever import retrieve_faq
from state import AgentState

SYSTEM_TEMPLATE = """You are the General Help specialist on an insurance support team.
Answer using ONLY the reference material below. If it doesn't cover the question, say you're
not sure and suggest escalation rather than guessing.

Reference material:
{context}
"""

_llm = ChatAnthropic(model="claude-opus-5", thinking={"type": "disabled"})


def general_help_node(state: AgentState) -> dict:
    query = last_human_message(state["messages"])
    docs = retrieve_faq(query)
    context = "\n\n".join(docs) if docs else "No relevant reference material found."

    system_prompt = SYSTEM_TEMPLATE.format(context=context)
    ai_message = run_grounded_agent(
        _llm, system_prompt, state["messages"], name="general_help_agent"
    )
    return {
        "messages": [ai_message],
        "handled": state.get("handled", []) + ["general"],
    }
