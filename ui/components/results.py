"""Reusable result display components."""

import json
from typing import Any

import streamlit as st


def render_gap_card(gap: dict[str, Any], expanded: bool = False) -> None:
    """Render a gap result as an expandable card."""
    method_a = gap.get("method_a", {})
    method_b = gap.get("method_b", {})
    is_gap = gap.get("is_gap", False)

    title = f"{method_a.get('name', 'Unknown')} + {method_b.get('name', 'Unknown')}"
    subtitle = f"Task: {gap.get('task', 'Unknown')}"

    with st.expander(f"{title} - {subtitle}", expanded=expanded):
        # Gap indicator
        if is_gap:
            st.success("Research Gap Identified")
        else:
            st.info("Not a significant gap")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gap Score", f"{gap.get('gap_score', 0):.1f}")
        with col2:
            st.metric(
                method_a.get("name", "Method A"),
                f"{gap.get('method_a_paper_count', 0)} papers",
            )
        with col3:
            st.metric(
                method_b.get("name", "Method B"),
                f"{gap.get('method_b_paper_count', 0)} papers",
            )

        # Combination count
        st.caption(
            f"Combined in {gap.get('combination_paper_count', 0)} papers"
        )

        # Gap ID for reference
        st.caption(f"Gap ID: {gap.get('gap_id', 'N/A')}")

        # Store in session for hypothesis generation
        if st.button("Use for Hypothesis", key=f"use_gap_{gap.get('gap_id')}"):
            st.session_state["selected_gap_id"] = gap.get("gap_id")
            st.session_state["selected_gap"] = gap
            st.success("Gap selected. Go to Hypothesis Generator page.")


def render_paper_card(paper: dict[str, Any]) -> None:
    """Render a paper evidence card."""
    with st.container():
        st.markdown(f"**{paper.get('title', 'Untitled')}**")

        cols = st.columns([2, 1, 1])
        with cols[0]:
            st.caption(f"arXiv: {paper.get('arxiv_id', 'N/A')}")
        with cols[1]:
            st.caption(f"Year: {paper.get('year', 'N/A')}")
        with cols[2]:
            relevance = paper.get("relevance_score", 0)
            if relevance >= 0.8:
                st.caption(f"Relevance: {relevance:.0%}")
            elif relevance >= 0.5:
                st.caption(f"Relevance: {relevance:.0%}")
            else:
                st.caption(f"Relevance: {relevance:.0%}")

        if paper.get("excerpt"):
            st.markdown(f"> {paper['excerpt'][:300]}...")

        if paper.get("citation_count"):
            st.caption(f"Citations: {paper['citation_count']}")

        st.divider()


def render_hypothesis(hypothesis: dict[str, Any]) -> None:
    """Render a hypothesis with all details."""
    st.subheader("Generated Hypothesis")

    # Main hypothesis text
    st.markdown(f"**{hypothesis.get('hypothesis_text', 'No hypothesis text')}**")

    # Mechanism
    if hypothesis.get("mechanism"):
        st.markdown("**Proposed Mechanism:**")
        st.markdown(hypothesis["mechanism"])

    # Assumptions
    assumptions = hypothesis.get("assumptions", [])
    if assumptions:
        st.markdown("**Key Assumptions:**")
        for i, assumption in enumerate(assumptions, 1):
            text = assumption.get("text", str(assumption))
            st.markdown(f"{i}. {text}")
            if assumption.get("evidence_paper_id"):
                st.caption(f"   Evidence: {assumption['evidence_paper_id']}")

    # Evaluation Plan
    eval_plan = hypothesis.get("evaluation_plan", {})
    if eval_plan:
        st.markdown("**Evaluation Plan:**")
        cols = st.columns(2)
        with cols[0]:
            datasets = eval_plan.get("datasets", [])
            if datasets:
                st.markdown("*Datasets:*")
                for d in datasets:
                    st.markdown(f"- {d}")

            baselines = eval_plan.get("baselines", [])
            if baselines:
                st.markdown("*Baselines:*")
                for b in baselines:
                    st.markdown(f"- {b}")

        with cols[1]:
            metrics = eval_plan.get("metrics", [])
            if metrics:
                st.markdown("*Metrics:*")
                for m in metrics:
                    st.markdown(f"- {m}")

            if eval_plan.get("expected_outcome"):
                st.markdown("*Expected Outcome:*")
                st.markdown(eval_plan["expected_outcome"])

    # Scores if available
    scores_present = any([
        hypothesis.get("coherence_score"),
        hypothesis.get("evidence_relevance_score"),
        hypothesis.get("specificity_score"),
    ])
    if scores_present:
        st.markdown("**Evaluation Scores:**")
        cols = st.columns(3)
        with cols[0]:
            if hypothesis.get("coherence_score"):
                st.metric("Coherence", f"{hypothesis['coherence_score']}/10")
        with cols[1]:
            if hypothesis.get("evidence_relevance_score"):
                st.metric("Evidence", f"{hypothesis['evidence_relevance_score']}/10")
        with cols[2]:
            if hypothesis.get("specificity_score"):
                st.metric("Specificity", f"{hypothesis['specificity_score']}/10")

    # Metadata
    st.divider()
    cols = st.columns(2)
    with cols[0]:
        st.caption(f"Hypothesis ID: {hypothesis.get('hypothesis_id', 'N/A')}")
    with cols[1]:
        st.caption(f"Model: {hypothesis.get('model_version', 'N/A')}")

    # Download options
    st.markdown("**Export:**")
    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            label="Download JSON",
            data=json.dumps(hypothesis, indent=2, default=str),
            file_name=f"hypothesis_{hypothesis.get('hypothesis_id', 'export')}.json",
            mime="application/json",
        )
    with cols[1]:
        markdown_content = format_hypothesis_markdown(hypothesis)
        st.download_button(
            label="Download Markdown",
            data=markdown_content,
            file_name=f"hypothesis_{hypothesis.get('hypothesis_id', 'export')}.md",
            mime="text/markdown",
        )


def format_hypothesis_markdown(hypothesis: dict[str, Any]) -> str:
    """Format hypothesis as markdown."""
    lines = [
        f"# Research Hypothesis",
        "",
        f"## Hypothesis",
        hypothesis.get("hypothesis_text", ""),
        "",
    ]

    if hypothesis.get("mechanism"):
        lines.extend([
            "## Proposed Mechanism",
            hypothesis["mechanism"],
            "",
        ])

    assumptions = hypothesis.get("assumptions", [])
    if assumptions:
        lines.append("## Key Assumptions")
        for i, a in enumerate(assumptions, 1):
            text = a.get("text", str(a))
            lines.append(f"{i}. {text}")
        lines.append("")

    eval_plan = hypothesis.get("evaluation_plan", {})
    if eval_plan:
        lines.append("## Evaluation Plan")
        if eval_plan.get("datasets"):
            lines.append(f"- **Datasets:** {', '.join(eval_plan['datasets'])}")
        if eval_plan.get("baselines"):
            lines.append(f"- **Baselines:** {', '.join(eval_plan['baselines'])}")
        if eval_plan.get("metrics"):
            lines.append(f"- **Metrics:** {', '.join(eval_plan['metrics'])}")
        if eval_plan.get("expected_outcome"):
            lines.append(f"- **Expected Outcome:** {eval_plan['expected_outcome']}")
        lines.append("")

    lines.extend([
        "---",
        f"*Generated by ML Gap Finder*",
        f"*Hypothesis ID: {hypothesis.get('hypothesis_id', 'N/A')}*",
    ])

    return "\n".join(lines)


def render_literature_position(result: dict[str, Any]) -> None:
    """Render literature positioning results."""
    # Similar papers
    similar_papers = result.get("most_similar_papers", [])
    if similar_papers:
        st.subheader("Most Similar Papers")
        for paper in similar_papers:
            with st.container():
                st.markdown(f"**{paper.get('title', 'Untitled')}**")
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    st.caption(f"arXiv: {paper.get('arxiv_id', 'N/A')}")
                with cols[1]:
                    st.caption(f"Year: {paper.get('year', 'N/A')}")
                with cols[2]:
                    st.caption(f"Similarity: {paper.get('similarity_score', 0):.0%}")
                if paper.get("abstract_excerpt"):
                    st.markdown(f"> {paper['abstract_excerpt'][:200]}...")
                st.divider()

    # Method lineage
    lineage = result.get("method_lineage", [])
    if lineage:
        st.subheader("Method Lineage")
        for method in lineage:
            with st.expander(method.get("method_name", "Unknown Method")):
                if method.get("origin_paper"):
                    st.markdown(f"**Origin:** {method['origin_paper']}")
                evolution = method.get("evolution_papers", [])
                if evolution:
                    st.markdown("**Evolution:**")
                    for paper in evolution:
                        st.markdown(f"- {paper}")

    # Differentiation points
    diff_points = result.get("differentiation_points", [])
    if diff_points:
        st.subheader("Differentiation Points")
        for point in diff_points:
            st.markdown(f"- {point}")


def render_related_work_outline(result: dict[str, Any]) -> None:
    """Render related work outline."""
    sections = result.get("sections", [])

    if sections:
        st.subheader("Related Work Outline")
        for i, section in enumerate(sections, 1):
            with st.expander(f"{i}. {section.get('title', 'Section')}"):
                st.markdown(f"**Theme:** {section.get('theme', 'N/A')}")

                papers = section.get("papers_to_cite", [])
                if papers:
                    st.markdown("**Papers to cite:**")
                    for paper in papers:
                        st.markdown(f"- {paper}")

                if section.get("transition"):
                    st.markdown(f"**Transition:** {section['transition']}")

    if result.get("positioning_summary"):
        st.subheader("Positioning Summary")
        st.markdown(result["positioning_summary"])
