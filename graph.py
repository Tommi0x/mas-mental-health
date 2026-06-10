from typing import Callable, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import AgentProgressCallback, run_all_agents
from aggregate import (
    aggregate_copeland,
    aggregate_majority,
    aggregate_superconsent,
    aggregate_weighted,
)
from knowledge_base.prompt import build_knowledge_context, list_allowed_diseases
from models import Diagnosis

AggregationRule = Callable[[list[Diagnosis]], str]


class ClinicalGraphState(TypedDict):
    text: str
    active_agents: list[str]
    active_categories: list[str]
    method: str
    agreement_percent: int
    agent_weights: dict[str, float]
    api_keys: dict[str, str]
    diagnoses: list[Diagnosis]
    agent_failures: list[str]
    result: str
    progress_callback: NotRequired[AgentProgressCallback | None]


RULES: dict[str, AggregationRule] = {
    "majority": aggregate_majority,
    "superconsent": aggregate_superconsent,
    "copeland": aggregate_copeland,
    "weighted": aggregate_weighted,
}


# Run the selected agents and store their diagnoses in state.
def execute_agents(state: ClinicalGraphState) -> dict[str, list[Diagnosis] | list[str]]:
    active_categories = state["active_categories"]
    knowledge_context = build_knowledge_context(active_categories)
    allowed_diseases = list_allowed_diseases(active_categories)
    diagnoses, failures = run_all_agents(
        state["text"],
        state["active_agents"],
        knowledge_context,
        allowed_diseases,
        state.get("api_keys"),
        progress_callback=state.get("progress_callback"),
    )
    if not diagnoses:
        detail = "; ".join(failures) if failures else "no valid responses"
        raise ValueError(f"All agents failed: {detail}")
    return {"diagnoses": diagnoses, "agent_failures": failures}


# Apply the selected aggregation rule to the diagnoses in state.
def aggregate_diagnoses(state: ClinicalGraphState) -> dict[str, str]:
    method = state["method"]
    diagnoses = state["diagnoses"]

    if method == "superconsent":
        result = aggregate_superconsent(
            diagnoses,
            agreement_percent=state["agreement_percent"],
        )
        return {"result": result}

    if method == "weighted":
        result = aggregate_weighted(diagnoses, state["agent_weights"])
        return {"result": result}

    rule = RULES.get(method)
    if rule is None:
        raise ValueError(f"Unknown aggregation method: {method}")
    return {"result": rule(diagnoses)}


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


# Run the workflow and return the final result with diagnoses and agent failures.
def run(
    text: str,
    method: str,
    active_agents: list[str],
    active_categories: list[str] | None = None,
    agreement_percent: int = 75,
    agent_weights: dict[str, float] | None = None,
    api_keys: dict[str, str] | None = None,
    progress_callback: AgentProgressCallback | None = None,
) -> tuple[str, list[Diagnosis], list[str]]:
    if not active_agents:
        raise ValueError("At least one active agent is required")

    state: ClinicalGraphState = {
        "text": text,
        "active_agents": active_agents,
        "active_categories": active_categories or [],
        "method": method,
        "agreement_percent": agreement_percent,
        "agent_weights": agent_weights or {},
        "api_keys": api_keys or {},
        "diagnoses": [],
        "agent_failures": [],
        "result": "",
        "progress_callback": progress_callback,
    }
    final_state = _GRAPH.invoke(state)
    return (
        final_state["result"],
        final_state["diagnoses"],
        final_state["agent_failures"],
    )
