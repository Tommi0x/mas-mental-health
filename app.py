import os

import streamlit as st

from agents import AGENTS, AgentProgressState, build_prompt
from graph import RULES, run
from llm import (
    API_PROVIDERS,
    is_model_available,
    missing_api_key_errors,
    provider_session_key,
    resolve_api_key,
    save_api_keys_to_env,
)
from knowledge_base.prompt import build_knowledge_context, list_allowed_diseases, list_categories
from models import Diagnosis

_METHOD_DESCRIPTIONS: dict[str, str] = {
    "majority": "The diagnosis with the most agent votes wins. Ties are reported explicitly.",
    "superconsent": "A diagnosis wins only if enough agents agree, based on the threshold below.",
    "copeland": (
        "Each diagnosis earns points by beating others in pairwise vote comparisons; "
        "the highest score wins."
    ),
    "weighted": "Each agent vote is multiplied by its weight; the diagnosis with the highest total wins.",
}

_AGENT_MODELS = {agent["name"]: agent["model"] for agent in AGENTS}


# Render live agent progress at the bottom of the page during analysis.
def _render_agent_progress(
    progress_area: st.delta_generator.DeltaGenerator,
    state: AgentProgressState,
    agent_order: list[str],
) -> None:
    finished = len(state["completed"]) + len(state["failed"])
    progress_value = finished / state["total"] if state["total"] else 0.0
    running = set(state["running"])
    completed = set(state["completed"])
    failed = set(state["failed"])

    with progress_area.container():
        st.subheader("Analysis progress")
        st.caption(
            f"Batch {state['batch_index']} of {state['batch_count']} · "
            f"{finished} of {state['total']} agents finished"
        )
        st.progress(progress_value)
        for agent_name in agent_order:
            model = _AGENT_MODELS.get(agent_name, "")
            if agent_name in running:
                st.markdown(f"- :orange[{agent_name}] `{model}` — running")
            elif agent_name in completed:
                st.markdown(f"- :green[{agent_name}] `{model}` — done")
            elif agent_name in failed:
                st.markdown(f"- :red[{agent_name}] `{model}` — failed")
            else:
                st.markdown(f"- {agent_name} `{model}` — queued")


# Return agents whose provider API key is configured.
def _available_agents(api_keys: dict[str, str]) -> list[dict[str, str]]:
    return [
        agent
        for agent in AGENTS
        if is_model_available(agent["model"], api_keys)
    ]


# Return a balanced default selection across configured providers when possible.
def _default_active_agent_names(available_names: list[str]) -> list[str]:
    if not available_names:
        return []

    available_set = set(available_names)
    provider_groups: list[list[str]] = []
    for provider in API_PROVIDERS:
        group = [
            agent["name"]
            for agent in AGENTS
            if agent["model"].startswith(provider["model_prefix"])
            and agent["name"] in available_set
        ]
        if group:
            provider_groups.append(group)

    if not provider_groups:
        return available_names[:2]

    defaults: list[str] = []
    index = 0
    while len(defaults) < 2:
        added = False
        for group in provider_groups:
            if index < len(group):
                name = group[index]
                if name not in defaults:
                    defaults.append(name)
                    added = True
                if len(defaults) >= 2:
                    return defaults
        if not added:
            for name in available_names:
                if name not in defaults:
                    defaults.append(name)
                if len(defaults) >= 2:
                    break
            return defaults[:2]
        index += 1

    return defaults[:2]


# Return category ids and a lookup map from id to display title.
def _category_options() -> tuple[list[str], dict[str, str]]:
    pairs = list_categories()
    category_ids = [category_id for category_id, _ in pairs]
    labels = {category_id: title for category_id, title in pairs}
    return category_ids, labels


# Convert diagnoses into table rows for Streamlit.
def _diagnoses_to_rows(diagnoses: list[Diagnosis]) -> list[dict[str, str]]:
    return [diagnosis.model_dump() for diagnosis in diagnoses]


# Build the agent prompt shown to every active model.
def _preview_prompt(case_text: str, active_categories: list[str]) -> str:
    knowledge_context = build_knowledge_context(active_categories)
    allowed_diseases = list_allowed_diseases(active_categories)
    return build_prompt(case_text.strip(), knowledge_context, allowed_diseases)


# Render a read-only preview of the prompt sent to each active agent.
def _render_prompt_preview(case_text: str, active_categories: list[str]) -> None:
    prompt = _preview_prompt(case_text, active_categories)
    with st.expander("Agent prompt preview", expanded=bool(case_text.strip())):
        st.caption("The same prompt is sent to each active agent.")
        st.code(prompt, language="text")


# Load sidebar API key fields from the environment on first run.
def _init_api_key_fields() -> None:
    for provider in API_PROVIDERS:
        field_key = provider_session_key(provider["env_name"])
        if field_key not in st.session_state:
            st.session_state[field_key] = os.environ.get(provider["env_name"], "")


# Return API keys entered in the sidebar.
def _sidebar_api_keys() -> dict[str, str]:
    return {
        provider["env_name"]: st.session_state.get(
            provider_session_key(provider["env_name"]), ""
        )
        for provider in API_PROVIDERS
    }


# Render the sidebar API key fields.
def _render_sidebar() -> dict[str, str]:
    st.sidebar.header("Configuration")
    _init_api_key_fields()

    st.sidebar.subheader("API keys")
    st.sidebar.caption(
        "All providers are optional. Add a key only for the providers you want to use; "
        "their agents appear in Active agents after the key is set. Sidebar values override "
        "`.env`. Save writes non-empty keys to `.env`."
    )
    for provider in API_PROVIDERS:
        st.sidebar.text_input(
            f"{provider['display_name']} API key",
            type="password",
            key=provider_session_key(provider["env_name"]),
        )

    if st.sidebar.button("Save API keys"):
        keys = _sidebar_api_keys()
        if not any(value.strip() for value in keys.values()):
            st.sidebar.warning("Enter at least one API key before saving.")
        else:
            save_api_keys_to_env(keys)
            st.sidebar.success("API keys saved to `.env`.")

    api_keys = _sidebar_api_keys()
    for provider in API_PROVIDERS:
        if resolve_api_key(provider["env_name"], api_keys):
            st.sidebar.success(f"{provider['display_name']} key configured.")
        else:
            st.sidebar.info(
                f"{provider['display_name']} not configured. "
                f"[Get a free key]({provider['signup_url']})."
            )

    return api_keys


# Render the Streamlit interface and run the analysis workflow.
def main() -> None:
    st.set_page_config(page_title="MAS Clinical Diagnosis", layout="wide")
    st.title("MAS Clinical Diagnosis")
    st.write(
        "Enter a free-text clinical case, choose the active agents and the "
        "aggregation rule, then run the analysis."
    )

    api_keys = _render_sidebar()

    available_names = [agent["name"] for agent in _available_agents(api_keys)]
    if not available_names:
        st.info("Add at least one provider API key in the sidebar to enable agents.")
    active_agents = st.multiselect(
        "Active agents",
        options=available_names,
        default=_default_active_agent_names(available_names),
        disabled=not available_names,
    )

    method = st.selectbox("Aggregation method", options=list(RULES.keys()))
    st.caption(_METHOD_DESCRIPTIONS[method])
    agreement_percent = 75
    agent_weights: dict[str, float] = {}
    if method == "superconsent":
        agreement_percent = st.slider(
            "Agreement threshold (%)",
            min_value=50,
            max_value=100,
            value=75,
            step=1,
            help="A diagnosis must reach this share of agent votes to win.",
        )
    elif method == "weighted":
        st.caption(
            "Set a weight for each active agent. Higher weights count more toward the final diagnosis."
        )
        if not active_agents:
            st.info("Select at least one active agent to configure weights.")
        else:
            for agent_name in active_agents:
                agent_weights[agent_name] = st.number_input(
                    agent_name,
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key=f"agent_weight_{agent_name}",
                )

    category_ids, category_labels = _category_options()
    active_categories = st.multiselect(
        "Knowledge base categories",
        options=category_ids,
        default=[],
        format_func=lambda category_id: category_labels[category_id],
        help="Selected DSM-5 reference sections are appended to each agent prompt.",
    )
    case_text = st.text_area(
        "Clinical case",
        height=240,
        placeholder="Enter the clinical case in free text...",
    )

    _render_prompt_preview(case_text, active_categories)

    if st.button("Analyze", type="primary"):
        if not case_text.strip():
            st.error("Please enter a clinical case before running the analysis.")
            return
        if not active_agents:
            st.error("Select at least one active agent.")
            return
        agent_models = [
            agent["model"] for agent in AGENTS if agent["name"] in active_agents
        ]
        key_errors = missing_api_key_errors(agent_models, api_keys)
        if key_errors:
            st.error("\n\n".join(key_errors))
            return

        progress_area = st.empty()

        def _on_agent_progress(state: AgentProgressState) -> None:
            _render_agent_progress(progress_area, state, active_agents)

        try:
            result, diagnoses, agent_failures = run(
                case_text,
                method,
                active_agents,
                active_categories,
                agreement_percent=agreement_percent,
                agent_weights=agent_weights,
                api_keys=api_keys,
                progress_callback=_on_agent_progress,
            )
        except Exception as exc:
            progress_area.empty()
            st.error(f"Analysis failed: {exc}")
            return

        progress_area.empty()

        if agent_failures:
            st.warning(
                "Some agents returned invalid responses and were excluded "
                "from aggregation:\n\n" + "\n".join(f"- {item}" for item in agent_failures)
            )

        st.subheader("Final result")
        st.success(result)

        st.subheader("Per-agent diagnoses")
        st.dataframe(
            _diagnoses_to_rows(diagnoses),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
