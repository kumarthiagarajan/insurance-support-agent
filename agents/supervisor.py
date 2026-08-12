from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from state import AgentState

SYSTEM_PROMPT = """You are the Supervisor of an insurance customer support system. Your job is \
to read the conversation and decide which specialist should handle it next.

Specialists available:
- policy: policy details, coverage limits, what's included/excluded, auto/home policy specifics
- billing: invoices, payment status, payment methods, amounts due, billing history
- claims: claim status, filing a new claim, adjuster info, claim timelines
- general: general insurance questions not tied to this customer's account (deductibles, \
grace periods, how bundling works, how to add a driver) -- answered via a knowledge base
- escalate: hand off to a human when the request is out of scope for all specialists above, \
the customer explicitly asks for a human, the customer is frustrated, or a specialist has \
already tried and gave up with no further question or option left to offer

Specialists already consulted this turn: {handled}

If the customer's latest message has more than one distinct part (e.g. a policy question AND \
a claims question), keep routing to whichever specialist covers each remaining part before \
responding FINISH -- do not stop early just because one part has been answered.

Once every part of the customer's request has been addressed by the specialists already \
consulted, check the most recent specialist reply: if it ends by asking the customer for \
confirmation or permission before taking a further step (e.g. "would you like me to escalate \
this?"), respond with FINISH so that question reaches the customer -- only route to escalate \
on the specialist's behalf after the customer has answered it, not before. Otherwise, if \
everything has been addressed, respond with FINISH. Do not re-route to a specialist that \
already answered unless the customer asked a new, different question in their latest message.
"""


class RouteDecision(BaseModel):
    next: Literal["policy", "billing", "claims", "general", "escalate", "FINISH"] = Field(
        description="Which specialist handles the next step, or FINISH if fully resolved."
    )
    reasoning: str = Field(description="One sentence explanation, for logging only.")


_llm = ChatAnthropic(model="claude-opus-5", thinking={"type": "disabled"})
_router = _llm.with_structured_output(RouteDecision)
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("messages"),
        ("human", "Based on the conversation so far, decide the next step."),
    ]
)
_chain = _prompt | _router

MAX_ITERATIONS = 4


def supervisor_node(state: AgentState) -> dict:
    iterations = state.get("iterations", 0)
    if iterations >= MAX_ITERATIONS:
        return {"next": "escalate", "iterations": iterations + 1}

    handled = state.get("handled", [])
    decision = _chain.invoke(
        {
            "messages": state["messages"],
            "handled": ", ".join(handled) or "none",
        }
    )
    next_node = decision.next

    # Deterministic guardrail: never let a specialist's own reply escalate the
    # customer to a human within the same turn -- the customer must see that
    # reply and explicitly confirm (a fresh turn, with `handled` reset) before
    # a specialist's recommendation to escalate can be acted on. Relying on
    # the LLM alone to hold off is unreliable, since it sometimes escalates
    # immediately regardless of prompt instructions.
    if next_node == "escalate" and handled:
        next_node = "FINISH"

    return {"next": next_node, "iterations": iterations + 1}
