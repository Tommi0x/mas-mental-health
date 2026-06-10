import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypedDict

from llm import ask_llm, provider_key_for_model
from models import Diagnosis


class AgentProgressState(TypedDict):
    running: list[str]
    completed: list[str]
    failed: list[str]
    total: int
    batch_index: int
    batch_count: int


AgentProgressCallback = Callable[[AgentProgressState], None]

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
    {"name": "Gemini31FlashLite", "model": "gemini/gemini-3.1-flash-lite"},
    {"name": "Gemini35Flash", "model": "gemini/gemini-3.5-flash"},
    {"name": "OR_DolphinMistral24B", "model": "openrouter/cognitivecomputations/dolphin-mistral-24b-venice-edition:free"},
    {"name": "OR_Gemma4_26B", "model": "openrouter/google/gemma-4-26b-a4b-it:free"},
    {"name": "OR_KimiK26", "model": "openrouter/moonshotai/kimi-k2.6:free"},
    {"name": "OR_NemotronNano9B", "model": "openrouter/nvidia/nemotron-nano-9b-v2:free"},
    {"name": "OR_LagunaXS2", "model": "openrouter/poolside/laguna-xs.2:free"},
    {"name": "OR_NexN2Pro", "model": "openrouter/nex-agi/nex-n2-pro:free"},
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


# Run one agent and return a diagnosis or a failure message for that agent.
def _run_agent_safe(
    cfg: dict[str, str],
    case_text: str,
    knowledge_context: str,
    allowed_diseases: list[str] | None,
    api_keys: dict[str, str] | None,
) -> tuple[Diagnosis | None, str | None]:
    try:
        return (
            run_agent(cfg, case_text, knowledge_context, allowed_diseases, api_keys),
            None,
        )
    except Exception as exc:
        return None, f"{cfg['name']}: {exc}"


# Notify the optional progress callback with the current agent run state.
def _report_progress(
    callback: AgentProgressCallback | None,
    *,
    running: list[str],
    completed: list[str],
    failed: list[str],
    total: int,
    batch_index: int,
    batch_count: int,
) -> None:
    if callback is None:
        return
    callback(
        {
            "running": list(running),
            "completed": list(completed),
            "failed": list(failed),
            "total": total,
            "batch_index": batch_index,
            "batch_count": batch_count,
        }
    )


# Run a batch of agents in parallel with at most one agent per provider.
def _run_agent_batch(
    batch: list[dict[str, str]],
    case_text: str,
    knowledge_context: str,
    allowed_diseases: list[str] | None,
    api_keys: dict[str, str] | None,
    *,
    progress_callback: AgentProgressCallback | None = None,
    completed: list[str] | None = None,
    failed: list[str] | None = None,
    total: int = 0,
    batch_index: int = 0,
    batch_count: int = 0,
) -> tuple[list[Diagnosis], list[str]]:
    if not batch:
        return [], []

    completed_names = completed if completed is not None else []
    failed_names = failed if failed is not None else []
    running_names = [cfg["name"] for cfg in batch]
    _report_progress(
        progress_callback,
        running=running_names,
        completed=completed_names,
        failed=failed_names,
        total=total,
        batch_index=batch_index,
        batch_count=batch_count,
    )

    if len(batch) == 1:
        diagnosis, failure = _run_agent_safe(
            batch[0], case_text, knowledge_context, allowed_diseases, api_keys
        )
        agent_name = batch[0]["name"]
        running_names.remove(agent_name)
        if diagnosis is not None:
            completed_names.append(agent_name)
        elif failure is not None:
            failed_names.append(agent_name)
        _report_progress(
            progress_callback,
            running=running_names,
            completed=completed_names,
            failed=failed_names,
            total=total,
            batch_index=batch_index,
            batch_count=batch_count,
        )
        diagnoses = [diagnosis] if diagnosis is not None else []
        failures = [failure] if failure is not None else []
        return diagnoses, failures

    diagnoses: list[Diagnosis] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        future_to_name = {
            executor.submit(
                _run_agent_safe,
                cfg,
                case_text,
                knowledge_context,
                allowed_diseases,
                api_keys,
            ): cfg["name"]
            for cfg in batch
        }
        for future in as_completed(future_to_name):
            agent_name = future_to_name[future]
            diagnosis, failure = future.result()
            running_names.remove(agent_name)
            if diagnosis is not None:
                diagnoses.append(diagnosis)
                completed_names.append(agent_name)
            if failure is not None:
                failures.append(failure)
                failed_names.append(agent_name)
            _report_progress(
                progress_callback,
                running=running_names,
                completed=completed_names,
                failed=failed_names,
                total=total,
                batch_index=batch_index,
                batch_count=batch_count,
            )
    return diagnoses, failures


# Build per-provider queues of agent configs in the user's selection order.
def _provider_agent_queues(
    active_agent_names: list[str],
) -> dict[str, deque[dict[str, str]]]:
    queues: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for name in active_agent_names:
        cfg = _AGENT_BY_NAME.get(name)
        if cfg is None:
            raise ValueError(f"Unknown agent: {name}")
        queues[provider_key_for_model(cfg["model"])].append(cfg)
    return queues


# Sort diagnoses to match the order agents were selected in the UI.
def _sort_diagnoses_by_agent_order(
    diagnoses: list[Diagnosis], active_agent_names: list[str]
) -> list[Diagnosis]:
    order = {name: index for index, name in enumerate(active_agent_names)}
    return sorted(
        diagnoses,
        key=lambda diagnosis: order.get(diagnosis.agent_name, len(active_agent_names)),
    )


# Return how many parallel batches are needed for the given provider queues.
def _batch_count(provider_queues: dict[str, deque[dict[str, str]]]) -> int:
    if not provider_queues:
        return 0
    return max(len(queue) for queue in provider_queues.values())


# Run active agents in parallel batches with at most one call per provider at a time.
def run_all_agents(
    case_text: str,
    active_agent_names: list[str],
    knowledge_context: str = "",
    allowed_diseases: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
    progress_callback: AgentProgressCallback | None = None,
) -> tuple[list[Diagnosis], list[str]]:
    provider_queues = _provider_agent_queues(active_agent_names)
    batch_count = _batch_count(provider_queues)
    total = len(active_agent_names)
    diagnoses: list[Diagnosis] = []
    failures: list[str] = []
    completed_names: list[str] = []
    failed_names: list[str] = []
    batch_index = 0

    _report_progress(
        progress_callback,
        running=[],
        completed=[],
        failed=[],
        total=total,
        batch_index=0,
        batch_count=batch_count,
    )

    while provider_queues:
        batch_index += 1
        batch: list[dict[str, str]] = []
        for provider_key in list(provider_queues.keys()):
            queue = provider_queues[provider_key]
            if queue:
                batch.append(queue.popleft())
            if not queue:
                del provider_queues[provider_key]

        batch_diagnoses, batch_failures = _run_agent_batch(
            batch,
            case_text,
            knowledge_context,
            allowed_diseases,
            api_keys,
            progress_callback=progress_callback,
            completed=completed_names,
            failed=failed_names,
            total=total,
            batch_index=batch_index,
            batch_count=batch_count,
        )
        diagnoses.extend(batch_diagnoses)
        failures.extend(batch_failures)

    return _sort_diagnoses_by_agent_order(diagnoses, active_agent_names), failures
