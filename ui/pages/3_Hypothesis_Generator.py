"""Hypothesis Generator page for ML Gap Finder UI."""

import streamlit as st

from ui.api_client import APIClient, APIError
from ui.components.results import render_hypothesis
from ui.components.sidebar import render_sidebar
from ui.components.styles import apply_custom_styles

st.set_page_config(
    page_title="Hypothesis Generator - ML Gap Finder",
    page_icon="magnifying_glass_tilted_left",
    layout="wide",
)

apply_custom_styles()


def get_api_client() -> APIClient:
    """Get or create API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    return st.session_state.api_client


def main():
    """Main page content."""
    api_client = get_api_client()
    render_sidebar(api_client)

    st.title("Hypothesis Generator")
    st.markdown("Generate evidence-grounded research hypotheses")

    # Check for pre-selected gap
    selected_gap = st.session_state.get("selected_gap")
    if selected_gap:
        st.info(
            f"Gap selected: {selected_gap.get('method_a', {}).get('name', 'Unknown')} + "
            f"{selected_gap.get('method_b', {}).get('name', 'Unknown')} "
            f"for {selected_gap.get('task', 'Unknown')}"
        )
        if st.button("Clear selection"):
            st.session_state.pop("selected_gap", None)
            st.session_state.pop("selected_gap_id", None)
            st.rerun()

    # Tabs for different input modes
    tab_from_gap, tab_manual = st.tabs(["From Gap ID", "Manual Input"])

    # From gap ID tab
    with tab_from_gap:
        st.markdown("### Generate from Existing Gap")
        st.markdown(
            "Enter a gap ID from a previous gap detection result."
        )

        with st.form("hypothesis_gap_form"):
            gap_id = st.text_input(
                "Gap ID",
                value=st.session_state.get("selected_gap_id", ""),
                placeholder="e.g., abc123def456",
                help="Gap ID from gap detection",
            )
            include_evidence = st.checkbox(
                "Include supporting evidence",
                value=True,
                help="Fetch and include relevant papers",
            )

            submitted = st.form_submit_button("Generate Hypothesis", type="primary")

        if submitted:
            if not gap_id:
                st.error("Please enter a gap ID")
            else:
                with st.spinner("Generating hypothesis... This may take a moment."):
                    try:
                        result = api_client.generate_hypothesis(
                            gap_id=gap_id,
                            include_evidence=include_evidence,
                        )
                        st.session_state["last_hypothesis"] = result
                    except APIError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    # Manual input tab
    with tab_manual:
        st.markdown("### Generate from Manual Input")
        st.markdown(
            "Specify methods and task directly to generate a hypothesis."
        )

        with st.form("hypothesis_manual_form"):
            col1, col2 = st.columns(2)

            with col1:
                method_a = st.text_input(
                    "Method A",
                    placeholder="e.g., contrastive learning",
                    help="First method",
                )
                method_b = st.text_input(
                    "Method B",
                    placeholder="e.g., graph neural network",
                    help="Second method",
                )

            with col2:
                task = st.text_input(
                    "Task",
                    placeholder="e.g., recommendation",
                    help="Target task",
                )
                manual_include_evidence = st.checkbox(
                    "Include supporting evidence",
                    value=True,
                    key="manual_evidence",
                )

            manual_submitted = st.form_submit_button(
                "Generate Hypothesis",
                type="primary",
            )

        if manual_submitted:
            if not method_a or not method_b or not task:
                st.error("Please fill in all fields")
            else:
                with st.spinner("Generating hypothesis... This may take a moment."):
                    try:
                        result = api_client.generate_hypothesis(
                            method_a=method_a,
                            method_b=method_b,
                            task=task,
                            include_evidence=manual_include_evidence,
                        )
                        st.session_state["last_hypothesis"] = result
                    except APIError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    # Display hypothesis results
    if "last_hypothesis" in st.session_state:
        st.divider()
        render_hypothesis(st.session_state["last_hypothesis"])

        # Evaluation section
        st.divider()
        st.markdown("### Evaluate Hypothesis")

        hypothesis_id = st.session_state["last_hypothesis"].get("hypothesis_id")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Tier 1 Evaluation"):
                with st.spinner("Running Tier 1 evaluation..."):
                    try:
                        eval_result = api_client.evaluate_hypothesis(
                            hypothesis_id=hypothesis_id,
                            tier=1,
                        )
                        st.session_state["last_evaluation"] = eval_result
                    except APIError as e:
                        st.error(f"Error: {e}")
            st.caption("Objective metrics")

        with col2:
            if st.button("Tier 2 Evaluation"):
                with st.spinner("Running Tier 2 evaluation..."):
                    try:
                        eval_result = api_client.evaluate_hypothesis(
                            hypothesis_id=hypothesis_id,
                            tier=2,
                        )
                        st.session_state["last_evaluation"] = eval_result
                    except APIError as e:
                        st.error(f"Error: {e}")
            st.caption("LLM-as-judge")

        with col3:
            if st.button("Tier 3 Evaluation"):
                with st.spinner("Running Tier 3 evaluation..."):
                    try:
                        eval_result = api_client.evaluate_hypothesis(
                            hypothesis_id=hypothesis_id,
                            tier=3,
                        )
                        st.session_state["last_evaluation"] = eval_result
                    except APIError as e:
                        st.error(f"Error: {e}")
            st.caption("Human calibration")

        # Display evaluation results
        if "last_evaluation" in st.session_state:
            eval_result = st.session_state["last_evaluation"]
            st.markdown("#### Evaluation Results")

            if eval_result.get("passed"):
                st.success("Hypothesis passed evaluation")
            else:
                st.warning("Hypothesis did not pass evaluation")

            st.markdown(f"**Tier:** {eval_result.get('tier', 'Unknown')}")

            scores = eval_result.get("scores", {})
            if scores:
                cols = st.columns(len(scores))
                for i, (key, value) in enumerate(scores.items()):
                    with cols[i]:
                        st.metric(key.replace("_", " ").title(), value)

            if eval_result.get("explanation"):
                st.markdown("**Explanation:**")
                st.markdown(eval_result["explanation"])


if __name__ == "__main__":
    main()
