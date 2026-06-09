import json
from typing import Any

from llm import ask_llm
from models import Diagnosis

PROMPT_CLINICAL_HEADER = """You are a psychiatrist. Analyze the case using DSM-5 criteria."""

PROMPT_JSON_OPEN = """Respond with ONLY valid JSON (no markdown, no text outside the JSON object) using exactly these keys:"""

PROMPT_JSON_OPEN_CONSTRAINED = """Respond with ONLY valid JSON (no markdown, no text outside the JSON object) using exactly these keys.
The "disease" value MUST be exactly one of the allowed diagnosis labels listed below (copy verbatim):"""

PROMPT_JSON_TEMPLATE = """{"disease": "<primary psychiatric diagnosis>", "explanation": "<brief clinical reasoning>"}"""

PROMPT_JSON_TEMPLATE_CONSTRAINED = """{"disease": "<one allowed diagnosis label>", "explanation": "<brief clinical reasoning>"}"""

PROMPT_CASE_LABEL = """Case:
"""


# Build the JSON response instructions, optionally restricted to allowed disease labels.
def _build_json_instructions(allowed_diseases: list[str]) -> str:
    if not allowed_diseases:
        return "\n".join(
            [
                PROMPT_JSON_OPEN,
                PROMPT_JSON_TEMPLATE,
                "",
                PROMPT_CASE_LABEL.strip(),
            ]
        )

    allowed_lines = "\n".join(f"- {label}" for label in allowed_diseases)
    return "\n".join(
        [
            PROMPT_JSON_OPEN_CONSTRAINED,
            allowed_lines,
            "",
            PROMPT_JSON_TEMPLATE_CONSTRAINED,
            "",
            PROMPT_CASE_LABEL.strip(),
        ]
    )


# Build the full agent prompt, optionally including selected knowledge base sections.
def build_prompt(
    case_text: str,
    knowledge_context: str,
    allowed_diseases: list[str] | None = None,
) -> str:
    diseases = allowed_diseases or []
    parts = [PROMPT_CLINICAL_HEADER.strip()]

    if knowledge_context.strip():
        parts.extend(
            [
                "",
                "Use the following DSM-5 reference material when evaluating the case:",
                "",
                knowledge_context.strip(),
            ]
        )

    parts.extend(["", _build_json_instructions(diseases), case_text.strip()])
    return "\n".join(parts)

AGENTS: list[dict[str, str]] = [
    {"name": "Llama70B", "model": "groq/llama-3.3-70b-versatile"},
    {"name": "Llama8B", "model": "groq/llama-3.1-8b-instant"},
    {"name": "Qwen32B", "model": "groq/qwen/qwen3-32b"},
    {"name": "Llama4Scout", "model": "groq/meta-llama/llama-4-scout-17b-16e-instruct"},
    {"name": "OSS20B", "model": "groq/openai/gpt-oss-20b"},
    {"name": "OSS120B", "model": "groq/openai/gpt-oss-120b"},
    {"name": "Gemini25Flash", "model": "gemini/gemini-2.5-flash"},
    {"name": "Gemini25FlashLite", "model": "gemini/gemini-2.5-flash-lite"},
    {"name": "Gemini3Flash", "model": "gemini/gemini-3-flash-preview"},
    {"name": "Gemini31FlashLite", "model": "gemini/gemini-3.1-flash-lite"},
    {"name": "Gemini35Flash", "model": "gemini/gemini-3.5-flash"},
]

_AGENT_BY_NAME = {a["name"]: a for a in AGENTS}


# Take the substring from the first `{` through the last `}` in the model output.
def _extract_json_object(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return raw[start : end + 1]


# Match a model disease label to a canonical allowed label when constraints apply.
def _resolve_allowed_disease(disease: str, allowed_diseases: list[str]) -> str:
    if not allowed_diseases:
        return disease.strip()

    normalized = disease.strip().lower()
    for label in allowed_diseases:
        if label.lower() == normalized:
            return label

    raise ValueError(f"Disease '{disease.strip()}' is not allowed.")


# Parse the model JSON into a Diagnosis with agent and model fields filled in.
def _parse_diagnosis(
    raw: str,
    agent_name: str,
    model_used: str,
    allowed_diseases: list[str] | None = None,
) -> Diagnosis:
    payload: dict[str, Any] = json.loads(_extract_json_object(raw))
    disease = payload.get("disease")
    explanation = payload.get("explanation")
    if not isinstance(disease, str) or not disease.strip():
        raise ValueError("Missing or invalid 'disease' in model response")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Missing or invalid 'explanation' in model response")
    resolved_disease = _resolve_allowed_disease(disease, allowed_diseases or [])
    return Diagnosis(
        agent_name=agent_name,
        model_used=model_used,
        disease=resolved_disease,
        explanation=explanation.strip(),
    )


# Run one agent on the case text and return its structured diagnosis.
def run_agent(
    cfg: dict[str, str],
    case_text: str,
    knowledge_context: str = "",
    allowed_diseases: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
) -> Diagnosis:
    prompt = build_prompt(case_text, knowledge_context, allowed_diseases)
    raw = ask_llm(prompt, cfg["model"], api_keys)
    return _parse_diagnosis(raw, cfg["name"], cfg["model"], allowed_diseases)


# Run each named active agent and return valid diagnoses plus per-agent failures.
def run_all_agents(
    case_text: str,
    active_agent_names: list[str],
    knowledge_context: str = "",
    allowed_diseases: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
) -> tuple[list[Diagnosis], list[str]]:
    results: list[Diagnosis] = []
    failures: list[str] = []
    for name in active_agent_names:
        cfg = _AGENT_BY_NAME.get(name)
        if cfg is None:
            raise ValueError(f"Unknown agent: {name}")
        try:
            results.append(
                run_agent(
                    cfg, case_text, knowledge_context, allowed_diseases, api_keys
                )
            )
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return results, failures
