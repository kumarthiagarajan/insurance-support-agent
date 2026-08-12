from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def run_grounded_agent(llm, system_prompt: str, messages: list, name: str) -> AIMessage:
    # When a prior specialist already replied this turn, `messages` ends on an
    # AIMessage. Claude rejects a request whose conversation ends on an
    # assistant turn ("prefill"), so anchor it back to a user turn.
    full_messages = [SystemMessage(content=system_prompt)] + list(messages)
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
