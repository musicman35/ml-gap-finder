"""Prompt templates for ML Gap Finder LLM operations."""


class PromptTemplates:
    """Collection of prompt templates for various LLM tasks."""

    # Method Extraction
    METHOD_EXTRACTION = """Extract machine learning methods from the following paper abstract.
For each method, identify:
1. The method name (as mentioned in the text)
2. The method type: "architecture" (model/network types), "technique" (training approaches), or "objective" (loss functions)
3. Your confidence (high/medium/low)

Abstract:
{abstract}

Respond in this exact format, one method per line:
METHOD: [name] | TYPE: [type] | CONFIDENCE: [high/medium/low]

If no methods are found, respond with: NO_METHODS_FOUND
"""

    # Hypothesis Generation
    HYPOTHESIS_GENERATION = """You are a research hypothesis generator. Given a research gap and supporting evidence, generate a structured hypothesis.

## Research Gap
{gap_description}

## Method A: {method_a_name}
Known applications and results:
{method_a_evidence}

## Method B: {method_b_name}
Known applications and results:
{method_b_evidence}

## Task: {task_name}
Current state-of-the-art:
{task_sota}

Generate a structured hypothesis with the following format:

### Hypothesis
[1-2 sentence main claim about combining Method A and Method B for Task]

### Proposed Mechanism
[2-3 sentences explaining WHY this combination should work, based on the complementary strengths of each method]

### Key Assumptions
1. [Assumption 1] - Evidence: [cite specific paper]
2. [Assumption 2] - Evidence: [cite specific paper]
3. [Assumption 3] - Evidence: [cite specific paper]

### Evaluation Plan
- Dataset: [Recommended dataset]
- Baselines: [What to compare against]
- Metrics: [How to measure success]
- Expected outcome: [Quantitative prediction if possible]
"""

    # Tier 2 Evaluation - Coherence
    COHERENCE_RATING = """Rate the coherence of this research hypothesis on a scale of 1-5.

Hypothesis: {hypothesis}
Mechanism: {mechanism}
Assumptions: {assumptions}

Criteria:
1 = Incoherent, logical gaps
2 = Weak connections
3 = Reasonable but some gaps
4 = Well-connected logic
5 = Highly coherent, compelling argument

Respond with just the number and a one-sentence justification.
Format: [1-5] - [justification]
"""

    # Tier 2 Evaluation - Evidence Relevance
    EVIDENCE_RELEVANCE_RATING = """Rate the evidence relevance of this hypothesis on a scale of 1-5.

Hypothesis: {hypothesis}
Cited Evidence:
{evidence_summaries}

Criteria:
1 = Evidence unrelated to claims
2 = Tangentially related
3 = Somewhat supportive
4 = Clearly supportive
5 = Strong, direct support for all claims

Respond with just the number and a one-sentence justification.
Format: [1-5] - [justification]
"""

    # Tier 2 Evaluation - Specificity
    SPECIFICITY_RATING = """Rate the specificity/actionability of this hypothesis on a scale of 1-5.

Hypothesis: {hypothesis}
Evaluation Plan: {evaluation_plan}

Criteria:
1 = Vague, cannot be tested
2 = General direction only
3 = Some concrete elements
4 = Clear methods and metrics
5 = Fully actionable with specific experiments

Respond with just the number and a one-sentence justification.
Format: [1-5] - [justification]
"""

    # Literature Positioning
    LITERATURE_POSITIONING = """Analyze how the proposed approach relates to existing work.

## Proposed Approach
{approach_description}

## Methods Used
{methods}

## Related Papers
{related_papers}

Provide:

### Positioning Summary
[2-3 sentences describing where this work fits in the literature]

### Key Differentiators
1. [How this differs from closest prior work]
2. [What new contribution it makes]
3. [What gap it addresses]

### Method Lineage
- [Method 1]: Evolved from [prior method], first introduced in [paper]
- [Method 2]: Evolved from [prior method], first introduced in [paper]

### Suggested Citations
List the most relevant papers that should be cited:
1. [Paper title] - [Why to cite]
2. [Paper title] - [Why to cite]
3. [Paper title] - [Why to cite]
"""

    # Related Work Outline Generation
    RELATED_WORK_OUTLINE = """Generate a related work section outline for a paper with the following approach.

## Paper Approach
{approach_description}

## Available Papers to Cite
{available_papers}

Generate a structured related work outline with 3-4 thematic subsections:

### 2. Related Work

#### 2.1 [Subsection Title 1]
- Main theme: [description]
- Key papers to cite: [list paper IDs with brief reason]
- Transition to next section: [one sentence]

#### 2.2 [Subsection Title 2]
- Main theme: [description]
- Key papers to cite: [list paper IDs with brief reason]
- Transition to next section: [one sentence]

#### 2.3 [Subsection Title 3]
- Main theme: [description]
- Key papers to cite: [list paper IDs with brief reason]
- How this connects to your work: [one sentence]

### Summary
How to position your work relative to prior art: [2-3 sentences]
"""

    # Gap Explanation
    GAP_EXPLANATION = """Explain why the following method combination represents a research gap.

## Gap Details
- Method A: {method_a}
- Method B: {method_b}
- Task: {task}
- Papers using Method A: {papers_a_count}
- Papers using Method B: {papers_b_count}
- Papers combining both: {combined_count}

## Evidence
Method A successes: {method_a_evidence}
Method B successes: {method_b_evidence}

Provide a clear explanation of:
1. Why this is a meaningful gap (not just absence of work)
2. What synergies might exist between the methods
3. What challenges might explain why this hasn't been explored
4. The potential impact of filling this gap

Keep the response concise (3-4 paragraphs).
"""

    @classmethod
    def format_hypothesis_prompt(
        cls,
        gap_description: str,
        method_a_name: str,
        method_a_evidence: str,
        method_b_name: str,
        method_b_evidence: str,
        task_name: str,
        task_sota: str,
    ) -> str:
        """Format the hypothesis generation prompt.

        Args:
            gap_description: Description of the research gap.
            method_a_name: Name of first method.
            method_a_evidence: Evidence for method A effectiveness.
            method_b_name: Name of second method.
            method_b_evidence: Evidence for method B effectiveness.
            task_name: Target task name.
            task_sota: Current state-of-the-art for the task.

        Returns:
            Formatted prompt string.
        """
        return cls.HYPOTHESIS_GENERATION.format(
            gap_description=gap_description,
            method_a_name=method_a_name,
            method_a_evidence=method_a_evidence,
            method_b_name=method_b_name,
            method_b_evidence=method_b_evidence,
            task_name=task_name,
            task_sota=task_sota,
        )

    @classmethod
    def format_coherence_prompt(
        cls,
        hypothesis: str,
        mechanism: str,
        assumptions: str,
    ) -> str:
        """Format the coherence rating prompt.

        Args:
            hypothesis: Hypothesis text.
            mechanism: Proposed mechanism.
            assumptions: List of assumptions.

        Returns:
            Formatted prompt string.
        """
        return cls.COHERENCE_RATING.format(
            hypothesis=hypothesis,
            mechanism=mechanism,
            assumptions=assumptions,
        )

    @classmethod
    def format_evidence_relevance_prompt(
        cls,
        hypothesis: str,
        evidence_summaries: str,
    ) -> str:
        """Format the evidence relevance rating prompt.

        Args:
            hypothesis: Hypothesis text.
            evidence_summaries: Summaries of cited evidence.

        Returns:
            Formatted prompt string.
        """
        return cls.EVIDENCE_RELEVANCE_RATING.format(
            hypothesis=hypothesis,
            evidence_summaries=evidence_summaries,
        )

    @classmethod
    def format_specificity_prompt(
        cls,
        hypothesis: str,
        evaluation_plan: str,
    ) -> str:
        """Format the specificity rating prompt.

        Args:
            hypothesis: Hypothesis text.
            evaluation_plan: Proposed evaluation plan.

        Returns:
            Formatted prompt string.
        """
        return cls.SPECIFICITY_RATING.format(
            hypothesis=hypothesis,
            evaluation_plan=evaluation_plan,
        )
