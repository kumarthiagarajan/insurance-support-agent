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
already tried and couldn't fully resolve it

Specialists already consulted this turn: {handled}

If everything the customer asked for has been addressed by the specialists already consulted, \
respond with FINISH. Do not re-route to a specialist that already answered unless the customer \
asked a new, different question in their latest message.
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

    decision = _chain.invoke(
        {
            "messages": state["messages"],
            "handled": ", ".join(state.get("handled", [])) or "none",
        }
    )
    return {"next": decision.next, "iterations": iterations + 1}
