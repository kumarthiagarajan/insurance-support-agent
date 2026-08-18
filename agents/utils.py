from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Anthropic's API only sees role="assistant" for every prior specialist reply --
# the `name` tag LangChain carries locally never reaches the model (see
# langchain_anthropic._format_messages, which reads only role and content).
# Without a label, a specialist can't tell a colleague's turn (different data,
# different scope) from its own, and sometimes "corrects" a colleague's
# perfectly accurate reply because it can't ground it in its own context.
COLLEAGUE_NOTE = (
    "\n\nOther assistant turns in this conversation may be replies from a "
    "different specialist on this team, labeled with their name in brackets "
    "(e.g. \"[billing_agent]: ...\"). Treat those as already-correct answers "
    "from their own area -- do not contradict, second-guess, or apologize for "
    "them. Only speak to your own area of the conversation. That bracket "
    "labeling is only there to identify a colleague's past turn -- do not "
    "prefix your own reply with any bracketed name or label."
)


def _label_colleague_turns(messages: list, current_name: str) -> list:
    labeled = []
    for m in messages:
        if isinstance(m, AIMessage) and m.name and m.name != current_name:
            labeled.append(AIMessage(content=f"[{m.name}]: {m.content}"))
        else:
            labeled.append(m)
    return labeled


def run_grounded_agent(llm, system_prompt: str, messages: list, name: str) -> AIMessage:
    # When a prior specialist already replied this turn, `messages` ends on an
    # AIMessage. Claude rejects a request whose conversation ends on an
    # assistant turn ("prefill"), so anchor it back to a user turn.
    full_messages = [SystemMessage(content=system_prompt + COLLEAGUE_NOTE)]
    full_messages += _label_colleague_turns(messages, name)
    full_messages.append(
        HumanMessage(content="Please respond to the customer's request above.")
    )
    response = llm.invoke(full_messages)
    return AIMessage(content=response.content, name=name)


def last_human_message(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""
