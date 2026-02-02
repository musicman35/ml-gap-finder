"""ML Gap Finder - Streamlit UI."""

import streamlit as st

from ui.api_client import APIClient
from ui.components.sidebar import render_sidebar
from ui.components.styles import apply_custom_styles

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="ML Gap Finder",
    page_icon="magnifying_glass_tilted_left",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()


def get_api_client() -> APIClient:
    """Get or create API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    return st.session_state.api_client


def main():
    """Main application entry point."""
    api_client = get_api_client()

    # Render sidebar
    render_sidebar(api_client)

    # Main content
    st.title("ML Gap Finder")
    st.markdown("Discover research gaps and generate evidence-grounded hypotheses")

    # Check API status
    health = api_client.health_check()
    api_available = health.get("status") == "healthy"

    if not api_available:
        st.error("API server is not available.")
        st.markdown("### Getting Started")
        st.markdown("1. Make sure Docker containers are running:")
        st.code("docker-compose up -d", language="bash")
        st.markdown("2. Start the API server:")
        st.code("uv run uvicorn src.main:app --port 8000", language="bash")
        st.markdown("3. Refresh this page")
        return

    # Dashboard overview
    st.success("System is ready")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Quick Actions")

        st.markdown("**Gap Detection**")
        st.markdown(
            "Find underexplored combinations of ML methods. "
            "Search for specific gaps or auto-discover promising research directions."
        )

        st.markdown("**Evidence Retrieval**")
        st.markdown(
            "Find papers supporting method-task claims. "
            "Validate citations and build evidence bundles."
        )

    with col2:
        st.markdown("### Research Workflow")

        st.markdown("**Hypothesis Generation**")
        st.markdown(
            "Generate research hypotheses from identified gaps. "
            "Includes mechanisms, assumptions, and evaluation plans."
        )

        st.markdown("**Literature Positioning**")
        st.markdown(
            "Position your approach in existing literature. "
            "Generate related work outlines with citations."
        )

    st.divider()

    # System info
    st.markdown("### System Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("API Version", health.get("version", "unknown"))

    with col2:
        st.metric("LLM Provider", health.get("llm_provider", "unknown"))

    with col3:
        dbs = health.get("databases", {})
        connected = sum(1 for v in dbs.values() if v)
        st.metric("Databases", f"{connected}/4 connected")

    # Navigation hint
    st.info("Use the sidebar to navigate between pages")


if __name__ == "__main__":
    main()
