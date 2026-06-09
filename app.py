import os

import streamlit as st

from agents import AGENTS, build_prompt
from graph import RULES, run
from llm import is_model_available, missing_api_key_errors, resolve_api_key, save_api_keys_to_env
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


# Return agents whose provider API key is configured.
def _available_agents(api_keys: dict[str, str]) -> list[dict[str, str]]:
    return [
        agent
        for agent in AGENTS
        if is_model_available(agent["model"], api_keys)
    ]


# Return the default multiselect selection from the available agent list.
def _default_active_agent_names(available_names: list[str]) -> list[str]:
    return available_names[:2]


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
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")


# Return API keys entered in the sidebar.
def _sidebar_api_keys() -> dict[str, str]:
    return {
        "GROQ_API_KEY": st.session_state.get("groq_api_key", ""),
        "GEMINI_API_KEY": st.session_state.get("gemini_api_key", ""),
    }


# Render the sidebar API key fields.
def _render_sidebar() -> dict[str, str]:
    st.sidebar.header("Configuration")
    _init_api_key_fields()

    st.sidebar.subheader("API keys")
    st.sidebar.caption(
        "Enter keys here or set them in `.env`. Sidebar values take precedence. "
        "Save writes non-empty keys to `.env`."
    )
    st.sidebar.text_input(
        "Groq API key",
        type="password",
        key="groq_api_key",
    )
    st.sidebar.text_input(
        "Gemini API key",
        type="password",
        key="gemini_api_key",
    )

    if st.sidebar.button("Save API keys"):
        keys = _sidebar_api_keys()
        if not any(value.strip() for value in keys.values()):
            st.sidebar.warning("Enter at least one API key before saving.")
        else:
            save_api_keys_to_env(keys)
            st.sidebar.success("API keys saved to `.env`.")

    api_keys = _sidebar_api_keys()
    if resolve_api_key("GROQ_API_KEY", api_keys):
        st.sidebar.success("Groq key configured.")
    else:
        st.sidebar.warning(
            "Groq key not set. [Get a free key at console.groq.com](https://console.groq.com)."
        )

    if resolve_api_key("GEMINI_API_KEY", api_keys):
        st.sidebar.success("Gemini key configured.")
    else:
        st.sidebar.warning(
            "Gemini key not set. [Get a free key at Google AI Studio](https://aistudio.google.com/apikey)."
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
        st.info("Add API keys in the sidebar to enable agents.")
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

        try:
            with st.spinner("Running clinical analysis..."):
                result, diagnoses, agent_failures = run(
                    case_text,
                    method,
                    active_agents,
                    active_categories,
                    agreement_percent=agreement_percent,
                    agent_weights=agent_weights,
                    api_keys=api_keys,
                )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

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
