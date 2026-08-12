from langgraph.graph import END, StateGraph

from agents.billing_agent import billing_agent_node
from agents.claims_agent import claims_agent_node
from agents.escalation_agent import escalation_node
from agents.general_help_agent import general_help_node
from agents.policy_agent import policy_agent_node
from agents.supervisor import supervisor_node
from state import AgentState


def route_from_supervisor(state: AgentState) -> str:
    return state["next"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("policy", policy_agent_node)
    graph.add_node("billing", billing_agent_node)
    graph.add_node("claims", claims_agent_node)
    graph.add_node("general", general_help_node)
    graph.add_node("escalate", escalation_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "policy": "policy",
            "billing": "billing",
            "claims": "claims",
            "general": "general",
            "escalate": "escalate",
            "FINISH": END,
        },
    )

    for node in ("policy", "billing", "claims", "general"):
        graph.add_edge(node, "supervisor")

    graph.add_edge("escalate", END)

    return graph.compile()
