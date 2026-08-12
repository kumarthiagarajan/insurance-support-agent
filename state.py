import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    customer_id: str
    next: str
    handled: List[str]
    iterations: int
