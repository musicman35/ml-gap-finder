"""Sidebar component for ML Gap Finder UI."""

import streamlit as st

from ui.api_client import APIClient


def render_sidebar(api_client: APIClient) -> None:
    """Render the sidebar with status and info."""
    with st.sidebar:
        st.title("ML Gap Finder")
        st.caption("Research Gap Detection & Hypothesis Generation")

        st.divider()

        # API Status
        st.subheader("System Status")
        health = api_client.health_check()

        if health.get("status") == "healthy":
            st.success("API Connected")
        elif health.get("status") == "unavailable":
            st.error("API Unavailable")
            st.caption("Start the API server with:")
            st.code("uv run uvicorn src.main:app --port 8000", language="bash")
        else:
            st.warning(f"API Status: {health.get('status', 'unknown')}")

        # Database Status
        if "databases" in health:
            st.caption("Database Connections:")
            dbs = health["databases"]
            cols = st.columns(2)
            with cols[0]:
                if dbs.get("postgres"):
                    st.caption("PostgreSQL: OK")
                else:
                    st.caption("PostgreSQL: --")
                if dbs.get("neo4j"):
                    st.caption("Neo4j: OK")
                else:
                    st.caption("Neo4j: --")
            with cols[1]:
                if dbs.get("qdrant"):
                    st.caption("Qdrant: OK")
                else:
                    st.caption("Qdrant: --")
                if dbs.get("redis"):
                    st.caption("Redis: OK")
                else:
                    st.caption("Redis: --")

        # LLM Provider
        if health.get("llm_provider"):
            st.divider()
            st.caption(f"LLM Provider: {health['llm_provider']}")

        st.divider()

        # Version info
        st.caption(f"Version: {health.get('version', 'unknown')}")


def render_api_status_badge(api_client: APIClient) -> bool:
    """Render a compact API status badge. Returns True if API is available."""
    health = api_client.health_check()
    is_available = health.get("status") == "healthy"

    if is_available:
        st.caption("API: Connected")
    else:
        st.error("API not available. Please start the server.")

    return is_available
