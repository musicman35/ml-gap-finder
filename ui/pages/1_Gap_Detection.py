"""Gap Detection page for ML Gap Finder UI."""

import streamlit as st

from ui.api_client import APIClient, APIError
from ui.components.results import render_gap_card
from ui.components.sidebar import render_sidebar
from ui.components.styles import apply_custom_styles

st.set_page_config(
    page_title="Gap Detection - ML Gap Finder",
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

    st.title("Gap Detection")
    st.markdown("Find underexplored combinations of ML methods")

    # Tabs for different modes
    tab_search, tab_discover = st.tabs(["Search Specific Gap", "Auto-Discover Gaps"])

    # Search tab
    with tab_search:
        st.markdown("### Search for a Specific Gap")
        st.markdown(
            "Enter two methods and a task to check if their combination "
            "represents a research gap."
        )

        with st.form("gap_search_form"):
            col1, col2 = st.columns(2)

            with col1:
                method_a = st.text_input(
                    "Method A",
                    placeholder="e.g., transformer, attention mechanism",
                    help="First method or technique",
                )
                method_b = st.text_input(
                    "Method B",
                    placeholder="e.g., graph neural network, contrastive learning",
                    help="Second method or technique",
                )

            with col2:
                task = st.text_input(
                    "Task",
                    placeholder="e.g., recommendation, text classification",
                    help="The ML task or application area",
                )
                min_papers = st.slider(
                    "Min papers per method",
                    min_value=1,
                    max_value=50,
                    value=5,
                    help="Minimum papers using each method individually",
                )
                max_combo = st.slider(
                    "Max combination papers",
                    min_value=0,
                    max_value=10,
                    value=2,
                    help="Maximum papers combining both methods (lower = bigger gap)",
                )

            submitted = st.form_submit_button("Search for Gap", type="primary")

        if submitted:
            if not method_a or not method_b or not task:
                st.error("Please fill in all fields")
            else:
                with st.spinner("Searching for gap..."):
                    try:
                        result = api_client.search_gap(
                            method_a=method_a,
                            method_b=method_b,
                            task=task,
                            min_individual_papers=min_papers,
                            max_combination_papers=max_combo,
                        )
                        st.session_state["last_gap_search"] = result
                    except APIError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        # Display search results
        if "last_gap_search" in st.session_state:
            st.divider()
            st.markdown("### Result")
            render_gap_card(st.session_state["last_gap_search"], expanded=True)

    # Discover tab
    with tab_discover:
        st.markdown("### Auto-Discover Gaps")
        st.markdown(
            "Automatically find promising research gaps for a given task."
        )

        with st.form("gap_discover_form"):
            col1, col2 = st.columns(2)

            with col1:
                discover_task = st.text_input(
                    "Task",
                    placeholder="e.g., recommendation, image classification",
                    help="The ML task to find gaps for",
                    key="discover_task",
                )
                method_type = st.selectbox(
                    "Method Type Filter (optional)",
                    options=["", "architecture", "technique", "loss_function", "optimizer"],
                    help="Optionally filter by method type",
                )

            with col2:
                top_k = st.slider(
                    "Number of gaps to find",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="How many gaps to return",
                )
                discover_min_papers = st.slider(
                    "Min papers per method",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="discover_min",
                )
                discover_max_combo = st.slider(
                    "Max combination papers",
                    min_value=0,
                    max_value=10,
                    value=2,
                    key="discover_max",
                )

            discover_submitted = st.form_submit_button("Discover Gaps", type="primary")

        if discover_submitted:
            if not discover_task:
                st.error("Please enter a task")
            else:
                with st.spinner("Discovering gaps..."):
                    try:
                        result = api_client.discover_gaps(
                            task=discover_task,
                            method_type=method_type if method_type else None,
                            min_individual_papers=discover_min_papers,
                            max_combination_papers=discover_max_combo,
                            top_k=top_k,
                        )
                        st.session_state["last_gap_discover"] = result
                    except APIError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        # Display discover results
        if "last_gap_discover" in st.session_state:
            result = st.session_state["last_gap_discover"]
            st.divider()
            st.markdown(f"### Found {result.get('total_found', 0)} Gaps")

            gaps = result.get("gaps", [])
            if gaps:
                for i, gap in enumerate(gaps):
                    render_gap_card(gap, expanded=(i == 0))
            else:
                st.info("No gaps found. Try adjusting the parameters.")


if __name__ == "__main__":
    main()
