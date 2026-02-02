"""Evidence Retrieval page for ML Gap Finder UI."""

import streamlit as st

from ui.api_client import APIClient, APIError
from ui.components.results import render_paper_card
from ui.components.sidebar import render_sidebar
from ui.components.styles import apply_custom_styles

st.set_page_config(
    page_title="Evidence Retrieval - ML Gap Finder",
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

    st.title("Evidence Retrieval")
    st.markdown("Find papers supporting method-task claims")

    # Tabs
    tab_evidence, tab_validate = st.tabs(["Get Evidence", "Validate Citation"])

    # Evidence tab
    with tab_evidence:
        st.markdown("### Find Supporting Evidence")
        st.markdown(
            "Enter a method and task to find papers that support claims "
            "about this combination."
        )

        with st.form("evidence_form"):
            col1, col2 = st.columns(2)

            with col1:
                method = st.text_input(
                    "Method",
                    placeholder="e.g., attention mechanism, BERT",
                    help="The method or technique",
                )
                task = st.text_input(
                    "Task",
                    placeholder="e.g., sentiment analysis, object detection",
                    help="The ML task",
                )

            with col2:
                claim_type = st.selectbox(
                    "Claim Type",
                    options=["improves", "outperforms", "enables"],
                    help="Type of claim to find evidence for",
                )
                max_papers = st.slider(
                    "Max papers",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Maximum number of papers to retrieve",
                )

            submitted = st.form_submit_button("Find Evidence", type="primary")

        if submitted:
            if not method or not task:
                st.error("Please fill in both method and task")
            else:
                with st.spinner("Retrieving evidence..."):
                    try:
                        result = api_client.get_evidence(
                            method=method,
                            task=task,
                            claim_type=claim_type,
                            max_papers=max_papers,
                        )
                        st.session_state["last_evidence"] = result
                    except APIError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        # Display evidence results
        if "last_evidence" in st.session_state:
            result = st.session_state["last_evidence"]
            st.divider()

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence = result.get("confidence", 0)
                st.metric("Confidence", f"{confidence:.0%}")
            with col2:
                strength = result.get("claim_support_strength", "unknown")
                st.metric("Support Strength", strength.capitalize())
            with col3:
                papers = result.get("papers", [])
                st.metric("Papers Found", len(papers))

            # Confidence indicator
            confidence = result.get("confidence", 0)
            if confidence >= 0.7:
                st.success("Strong evidence found")
            elif confidence >= 0.4:
                st.warning("Moderate evidence")
            else:
                st.error("Weak evidence")

            # Papers
            st.markdown("### Supporting Papers")
            papers = result.get("papers", [])
            if papers:
                for paper in papers:
                    render_paper_card(paper)
            else:
                st.info("No papers found")

    # Validate citation tab
    with tab_validate:
        st.markdown("### Validate Citation")
        st.markdown(
            "Check if a citation properly supports a claimed contribution."
        )

        with st.form("validate_form"):
            citing_id = st.text_input(
                "Citing Paper ID",
                placeholder="e.g., 2301.00001",
                help="arXiv ID of the citing paper",
            )
            cited_id = st.text_input(
                "Cited Paper ID",
                placeholder="e.g., 2205.00002",
                help="arXiv ID of the cited paper",
            )
            contribution = st.text_area(
                "Claimed Contribution",
                placeholder="e.g., This method achieves state-of-the-art results...",
                help="What the citation claims",
            )

            validate_submitted = st.form_submit_button("Validate", type="primary")

        if validate_submitted:
            if not citing_id or not cited_id or not contribution:
                st.error("Please fill in all fields")
            else:
                with st.spinner("Validating citation..."):
                    try:
                        result = api_client.validate_citation(
                            citing_paper_id=citing_id,
                            cited_paper_id=cited_id,
                            claimed_contribution=contribution,
                        )
                        st.session_state["last_validation"] = result
                    except APIError as e:
                        st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        # Display validation results
        if "last_validation" in st.session_state:
            result = st.session_state["last_validation"]
            st.divider()

            # Overall validity
            if result.get("is_valid"):
                st.success("Citation is valid")
            else:
                st.error("Citation may not be valid")

            # Details
            col1, col2, col3 = st.columns(3)
            with col1:
                exists = result.get("cited_paper_exists", False)
                st.metric("Paper Exists", "Yes" if exists else "No")
            with col2:
                supported = result.get("claim_supported", False)
                st.metric("Claim Supported", "Yes" if supported else "No")
            with col3:
                similarity = result.get("similarity_score", 0)
                st.metric("Similarity", f"{similarity:.0%}")

            # Explanation
            if result.get("explanation"):
                st.markdown("**Explanation:**")
                st.markdown(result["explanation"])


if __name__ == "__main__":
    main()
