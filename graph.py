from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import run_all_agents
from aggregate import aggregate_majority, aggregate_unanimity
from knowledge_base.prompt import build_knowledge_context
from models import Diagnosis

AggregationRule = Callable[[list[Diagnosis]], str]


class ClinicalGraphState(TypedDict):
    text: str
    active_agents: list[str]
    active_categories: list[str]
    method: str
    diagnoses: list[Diagnosis]
    result: str


RULES: dict[str, AggregationRule] = {
    "majority": aggregate_majority,
    "unanimity": aggregate_unanimity,
}


# Run the selected agents and store their diagnoses in state.
def execute_agents(state: ClinicalGraphState) -> dict[str, list[Diagnosis]]:
    knowledge_context = build_knowledge_context(state["active_categories"])
    diagnoses = run_all_agents(
        state["text"],
        state["active_agents"],
        knowledge_context,
    )
    return {"diagnoses": diagnoses}


# Apply the selected aggregation rule to the diagnoses in state.
def aggregate_diagnoses(state: ClinicalGraphState) -> dict[str, str]:
    rule = RULES.get(state["method"])
    if rule is None:
        raise ValueError(f"Unknown aggregation method: {state['method']}")
    return {"result": rule(state["diagnoses"])}


# Build the LangGraph workflow used by the public run function.
def _build_graph():
    workflow = StateGraph(ClinicalGraphState)
    workflow.add_node("execute_agents", execute_agents)
    workflow.add_node("aggregate", aggregate_diagnoses)
    workflow.add_edge(START, "execute_agents")
    workflow.add_edge("execute_agents", "aggregate")
    workflow.add_edge("aggregate", END)
    return workflow.compile()


_GRAPH = _build_graph()


# Run the workflow and return the final result with all diagnoses.
def run(
    text: str,
    method: str,
    active_agents: list[str],
    active_categories: list[str] | None = None,
) -> tuple[str, list[Diagnosis]]:
    if not active_agents:
        raise ValueError("At least one active agent is required")

    state: ClinicalGraphState = {
        "text": text,
        "active_agents": active_agents,
        "active_categories": active_categories or [],
        "method": method,
        "diagnoses": [],
        "result": "",
    }
    final_state = _GRAPH.invoke(state)
    return final_state["result"], final_state["diagnoses"]
