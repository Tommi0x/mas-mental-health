import json
from typing import Any

from llm import ask_llm
from models import Diagnosis

PROMPT_CLINICAL = """You are a psychiatrist. Analyze the case using DSM-5 criteria.
Respond with ONLY valid JSON (no markdown, no text outside the JSON object) using exactly these keys:
{"disease": "<primary psychiatric diagnosis>", "explanation": "<brief clinical reasoning>"}

Case:
"""

AGENTS: list[dict[str, str]] = [
    {"name": "Judge_Llama", "model": "ollama/llama3.2"},
    {"name": "Judge_Mistral", "model": "ollama/mistral"},
    {"name": "Judge_Phi", "model": "ollama/phi3"},
    {"name": "Judge_Gemma", "model": "ollama/gemma2:2b"},
]

_AGENT_BY_NAME = {a["name"]: a for a in AGENTS}


# Take the substring from the first `{` through the last `}` in the model output.
def _extract_json_object(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return raw[start : end + 1]


# Parse the model JSON into a Diagnosis with agent and model fields filled in.
def _parse_diagnosis(raw: str, agent_name: str, model_used: str) -> Diagnosis:
    payload: dict[str, Any] = json.loads(_extract_json_object(raw))
    disease = payload.get("disease")
    explanation = payload.get("explanation")
    if not isinstance(disease, str) or not disease.strip():
        raise ValueError("Missing or invalid 'disease' in model response")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Missing or invalid 'explanation' in model response")
    return Diagnosis(
        agent_name=agent_name,
        model_used=model_used,
        disease=disease.strip(),
        explanation=explanation.strip(),
    )


# Run one agent on the case text and return its structured diagnosis.
def run_agent(cfg: dict[str, str], case_text: str) -> Diagnosis:
    prompt = PROMPT_CLINICAL + case_text.strip()
    raw = ask_llm(prompt, cfg["model"])
    return _parse_diagnosis(raw, cfg["name"], cfg["model"])


# Run each named active agent on the case and return all diagnoses in order.
def run_all_agents(case_text: str, active_agent_names: list[str]) -> list[Diagnosis]:
    results: list[Diagnosis] = []
    for name in active_agent_names:
        cfg = _AGENT_BY_NAME.get(name)
        if cfg is None:
            raise ValueError(f"Unknown agent: {name}")
        results.append(run_agent(cfg, case_text))
    return results
