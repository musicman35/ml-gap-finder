"""Literature Positioning page for ML Gap Finder UI."""

import streamlit as st

from ui.api_client import APIClient, APIError
from ui.components.results import render_literature_position, render_related_work_outline
from ui.components.sidebar import render_sidebar
from ui.components.styles import apply_custom_styles

st.set_page_config(
    page_title="Literature Position - ML Gap Finder",
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

    st.title("Literature Positioning")
    st.markdown("Position your research approach in existing literature")

    # Tabs
    tab_position, tab_related = st.tabs(["Position Approach", "Related Work Outline"])

    # Position approach tab
    with tab_position:
        st.markdown("### Position Your Approach")
        st.markdown(
            "Describe your research approach to find similar papers "
            "and understand how it relates to existing work."
        )

        with st.form("position_form"):
            approach_description = st.text_area(
                "Approach Description",
                placeholder=(
                    "Describe your research approach in detail (at least 50 characters). "
                    "For example: We propose a novel method that combines contrastive learning "
                    "with graph neural networks for recommendation systems..."
                ),
                height=150,
                help="Detailed description of your approach (min 50 characters)",
            )

            methods_input = st.text_input(
                "Methods Used",
                placeholder="e.g., transformer, attention, contrastive learning",
                help="Comma-separated list of methods used in your approach",
            )

            max_similar = st.slider(
                "Max similar papers",
                min_value=5,
                max_value=50,
                value=10,
                help="Maximum number of similar papers to retrieve",
            )

            submitted = st.form_submit_button("Find Position", type="primary")

        if submitted:
            if not approach_description or len(approach_description) < 50:
                st.error("Please provide a description of at least 50 characters")
            elif not methods_input:
                st.error("Please specify at least one method")
            else:
                methods = [m.strip() for m in methods_input.split(",") if m.strip()]
                if not methods:
                    st.error("Please specify at least one valid method")
                else:
                    with st.spinner("Analyzing literature position..."):
                        try:
                            result = api_client.position_in_literature(
                                approach_description=approach_description,
                                methods=methods,
                                max_similar_papers=max_similar,
                            )
                            st.session_state["last_position"] = result
                        except APIError as e:
                            st.error(f"Error: {e}")
                        except Exception as e:
                            st.error(f"Unexpected error: {e}")

        # Display position results
        if "last_position" in st.session_state:
            st.divider()
            render_literature_position(st.session_state["last_position"])

    # Related work tab
    with tab_related:
        st.markdown("### Generate Related Work Outline")
        st.markdown(
            "Generate a structured outline for your related work section "
            "with recommended citations."
        )

        with st.form("related_work_form"):
            related_description = st.text_area(
                "Approach Description",
                placeholder=(
                    "Describe your research approach in detail (at least 50 characters). "
                    "This will be used to identify relevant areas of related work..."
                ),
                height=150,
                key="related_description",
                help="Detailed description of your approach (min 50 characters)",
            )

            max_citations = st.slider(
                "Max citations",
                min_value=5,
                max_value=50,
                value=20,
                help="Maximum number of citations to include",
            )

            related_submitted = st.form_submit_button(
                "Generate Outline",
                type="primary",
            )

        if related_submitted:
            if not related_description or len(related_description) < 50:
                st.error("Please provide a description of at least 50 characters")
            else:
                with st.spinner("Generating related work outline..."):
                    try:
                        result = api_client.generate_related_work(
                            approach_description=related_description,
                            max_citations=max_citations,
                        )
                        st.session_state["last_related_work"] = result
                    except APIError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        # Display related work results
        if "last_related_work" in st.session_state:
            st.divider()
            render_related_work_outline(st.session_state["last_related_work"])


if __name__ == "__main__":
    main()
