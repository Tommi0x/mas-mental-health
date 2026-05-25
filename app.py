import streamlit as st

from agents import AGENTS
from graph import RULES, run
from models import Diagnosis


# Return the agent names used as the default selection in the UI.
def _default_active_agents() -> list[str]:
    return [agent["name"] for agent in AGENTS]


# Convert diagnoses into table rows for Streamlit.
def _diagnoses_to_rows(diagnoses: list[Diagnosis]) -> list[dict[str, str]]:
    return [diagnosis.model_dump() for diagnosis in diagnoses]


# Render the sidebar with agent details.
def _render_sidebar() -> None:
    st.sidebar.header("Configuration")
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
        options=_default_active_agents(),
        default=_default_active_agents(),
    )
    method = st.selectbox("Aggregation method", options=list(RULES.keys()))
    case_text = st.text_area(
        "Clinical case",
        height=240,
        placeholder="Enter the clinical case in free text...",
    )

    if st.button("Analyze", type="primary"):
        if not case_text.strip():
            st.error("Please enter a clinical case before running the analysis.")
            return
        if not active_agents:
            st.error("Select at least one active agent.")
            return

        try:
            with st.spinner("Running clinical analysis..."):
                result, diagnoses = run(case_text, method, active_agents)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

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
