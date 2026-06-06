import os

import streamlit as st

from agents import AGENTS, build_prompt
from graph import RULES, run
from knowledge_base.prompt import build_knowledge_context, list_allowed_diseases, list_categories
from models import Diagnosis


# Return all configured agent names for multiselect options.
def _all_agent_names() -> list[str]:
    return [agent["name"] for agent in AGENTS]


# Return the agent names used as the default selection in the UI.
def _default_active_agents() -> list[str]:
    return _all_agent_names()[:4]


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


# Render the sidebar with agent details.
def _render_sidebar() -> None:
    st.sidebar.header("Configuration")

    groq_configured = bool(os.environ.get("GROQ_API_KEY", "").strip())
    st.sidebar.subheader("Groq API")
    if groq_configured:
        st.sidebar.success("GROQ_API_KEY is configured.")
    else:
        st.sidebar.warning(
            "Set `GROQ_API_KEY` in a `.env` file or your environment. "
            "Create a free key at https://console.groq.com"
        )

    st.sidebar.subheader("Available agents")
    for agent in AGENTS:
        st.sidebar.markdown(f"- `{agent['name']}` -> `{agent['model']}`")


# Render the Streamlit interface and run the analysis workflow.
def main() -> None:
    st.set_page_config(page_title="MAS Clinical Diagnosis", layout="wide")
    st.title("MAS Clinical Diagnosis")
    st.write(
        "Enter a free-text clinical case, choose the active agents and the "
        "aggregation rule, then run the analysis."
    )

    _render_sidebar()

    active_agents = st.multiselect(
        "Active agents",
        options=_all_agent_names(),
        default=_default_active_agents(),
    )

    category_ids, category_labels = _category_options()
    active_categories = st.multiselect(
        "Knowledge base categories",
        options=category_ids,
        default=[],
        format_func=lambda category_id: category_labels[category_id],
        help="Selected DSM-5 reference sections are appended to each agent prompt.",
    )

    method = st.selectbox("Aggregation method", options=list(RULES.keys()))
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
        if not os.environ.get("GROQ_API_KEY", "").strip():
            st.error(
                "GROQ_API_KEY is not set. Copy `.env.example` to `.env`, add your free "
                "Groq key from https://console.groq.com, then restart the app."
            )
            return

        try:
            with st.spinner("Running clinical analysis..."):
                result, diagnoses, agent_failures = run(
                    case_text,
                    method,
                    active_agents,
                    active_categories,
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
