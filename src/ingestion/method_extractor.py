"""Method extraction from paper abstracts and text."""

import re
import uuid
from dataclasses import dataclass
from typing import Any

import structlog

from src.llm.client import BaseLLMClient, get_llm_client

logger = structlog.get_logger()

# Common ML method patterns
METHOD_PATTERNS = [
    # Architectures
    r"\b(transformer|bert|gpt|llm|cnn|rnn|lstm|gru|resnet|vit|vgg|unet)\b",
    r"\b(autoencoder|variational autoencoder|vae|gan|diffusion model)\b",
    r"\b(graph neural network|gnn|gcn|gat|graphsage)\b",
    # Techniques
    r"\b(attention mechanism|self-attention|cross-attention|multi-head attention)\b",
    r"\b(contrastive learning|self-supervised learning|representation learning)\b",
    r"\b(transfer learning|fine-tuning|pre-training|domain adaptation)\b",
    r"\b(reinforcement learning|q-learning|policy gradient|ppo|dqn)\b",
    r"\b(knowledge distillation|model compression|pruning|quantization)\b",
    # Objectives
    r"\b(cross-entropy|triplet loss|contrastive loss|focal loss)\b",
    r"\b(mean squared error|mse|mae|huber loss)\b",
    # Regularization
    r"\b(dropout|batch normalization|layer normalization|weight decay)\b",
    # Optimization
    r"\b(adam|sgd|adamw|rmsprop|lion optimizer)\b",
]


@dataclass
class ExtractedMethod:
    """Represents an extracted method from a paper."""

    method_id: str
    name: str
    method_type: str  # "architecture", "technique", "objective", "unknown"
    confidence: float
    context_snippet: str
    source: str  # "rule_based" or "llm"

    @classmethod
    def create(
        cls,
        name: str,
        method_type: str = "unknown",
        confidence: float = 1.0,
        context: str = "",
        source: str = "rule_based",
    ) -> "ExtractedMethod":
        """Create a new extracted method."""
        return cls(
            method_id=str(uuid.uuid4()),
            name=name,
            method_type=method_type,
            confidence=confidence,
            context_snippet=context[:500] if context else "",
            source=source,
        )


class MethodExtractor:
    """Extracts ML methods from paper text using rules and LLM."""

    # Method type classification
    ARCHITECTURE_KEYWORDS = [
        "transformer", "bert", "gpt", "cnn", "rnn", "lstm", "gru", "resnet",
        "vit", "vgg", "unet", "autoencoder", "vae", "gan", "gnn", "gcn", "gat",
        "network", "model", "encoder", "decoder", "diffusion",
    ]
    TECHNIQUE_KEYWORDS = [
        "attention", "contrastive", "self-supervised", "transfer learning",
        "fine-tuning", "pre-training", "distillation", "pruning", "quantization",
        "reinforcement", "meta-learning", "few-shot", "zero-shot",
    ]
    OBJECTIVE_KEYWORDS = [
        "loss", "objective", "criterion", "cross-entropy", "mse", "mae",
    ]

    # LLM extraction prompt
    EXTRACTION_PROMPT = """Extract machine learning methods from the following paper abstract.
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

    def __init__(self, llm_client: BaseLLMClient | None = None):
        """Initialize method extractor.

        Args:
            llm_client: Optional LLM client for enhanced extraction.
        """
        self._llm_client = llm_client
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in METHOD_PATTERNS]

    @property
    def llm_client(self) -> BaseLLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def _classify_method_type(self, method_name: str) -> str:
        """Classify method type based on keywords.

        Args:
            method_name: Name of the method.

        Returns:
            Method type classification.
        """
        name_lower = method_name.lower()

        for keyword in self.ARCHITECTURE_KEYWORDS:
            if keyword in name_lower:
                return "architecture"

        for keyword in self.TECHNIQUE_KEYWORDS:
            if keyword in name_lower:
                return "technique"

        for keyword in self.OBJECTIVE_KEYWORDS:
            if keyword in name_lower:
                return "objective"

        return "unknown"

    def extract_rule_based(
        self,
        text: str,
        context_window: int = 100,
    ) -> list[ExtractedMethod]:
        """Extract methods using regex patterns.

        Args:
            text: Paper abstract or full text.
            context_window: Characters around match for context.

        Returns:
            List of extracted methods.
        """
        methods = []
        seen_names = set()

        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                name = match.group().strip()
                name_normalized = name.lower()

                if name_normalized in seen_names:
                    continue
                seen_names.add(name_normalized)

                # Extract context around match
                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                context = text[start:end]

                method = ExtractedMethod.create(
                    name=name,
                    method_type=self._classify_method_type(name),
                    confidence=0.8,  # Rule-based has good precision
                    context=context,
                    source="rule_based",
                )
                methods.append(method)

        logger.debug("Rule-based extraction", method_count=len(methods))
        return methods

    async def extract_llm_based(
        self,
        abstract: str,
    ) -> list[ExtractedMethod]:
        """Extract methods using LLM.

        Args:
            abstract: Paper abstract.

        Returns:
            List of extracted methods.
        """
        try:
            prompt = self.EXTRACTION_PROMPT.format(abstract=abstract[:2000])

            response = await self.llm_client.generate(
                prompt=prompt,
                system="You are an expert at identifying machine learning methods in research papers.",
                temperature=0.3,
            )

            return self._parse_llm_response(response, abstract)

        except Exception as e:
            logger.warning("LLM extraction failed", error=str(e))
            return []

    def _parse_llm_response(
        self,
        response: str,
        abstract: str,
    ) -> list[ExtractedMethod]:
        """Parse LLM extraction response.

        Args:
            response: LLM response text.
            abstract: Original abstract for context.

        Returns:
            List of extracted methods.
        """
        methods = []

        if "NO_METHODS_FOUND" in response:
            return methods

        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}

        for line in response.strip().split("\n"):
            if not line.startswith("METHOD:"):
                continue

            try:
                # Parse: METHOD: [name] | TYPE: [type] | CONFIDENCE: [level]
                parts = line.split("|")
                if len(parts) != 3:
                    continue

                name = parts[0].replace("METHOD:", "").strip()
                method_type = parts[1].replace("TYPE:", "").strip().lower()
                confidence_str = parts[2].replace("CONFIDENCE:", "").strip().lower()

                if method_type not in ["architecture", "technique", "objective"]:
                    method_type = "unknown"

                confidence = confidence_map.get(confidence_str, 0.6)

                method = ExtractedMethod.create(
                    name=name,
                    method_type=method_type,
                    confidence=confidence,
                    context=abstract[:500],
                    source="llm",
                )
                methods.append(method)

            except Exception as e:
                logger.warning("Failed to parse LLM line", line=line, error=str(e))

        logger.debug("LLM extraction", method_count=len(methods))
        return methods

    async def extract(
        self,
        abstract: str,
        full_text: str | None = None,
        use_llm: bool = True,
    ) -> list[ExtractedMethod]:
        """Extract methods using both rule-based and LLM approaches.

        Args:
            abstract: Paper abstract.
            full_text: Optional full text for additional extraction.
            use_llm: Whether to use LLM for extraction.

        Returns:
            Deduplicated list of extracted methods.
        """
        # Start with rule-based extraction
        text = full_text or abstract
        methods = self.extract_rule_based(text)

        # Add LLM extraction if enabled
        if use_llm:
            llm_methods = await self.extract_llm_based(abstract)

            # Merge results, preferring higher confidence
            seen = {m.name.lower(): m for m in methods}
            for method in llm_methods:
                key = method.name.lower()
                if key not in seen or method.confidence > seen[key].confidence:
                    seen[key] = method

            methods = list(seen.values())

        # Sort by confidence
        methods.sort(key=lambda m: m.confidence, reverse=True)

        logger.info(
            "Method extraction complete",
            total_methods=len(methods),
            high_confidence=sum(1 for m in methods if m.confidence >= 0.8),
        )

        return methods

    def validate_against_pwc(
        self,
        extracted: list[ExtractedMethod],
        pwc_methods: list[str],
    ) -> dict[str, Any]:
        """Validate extracted methods against Papers With Code ground truth.

        Args:
            extracted: Extracted methods.
            pwc_methods: Method names from Papers With Code.

        Returns:
            Validation metrics (precision, recall, F1).
        """
        extracted_names = {m.name.lower() for m in extracted}
        pwc_names = {m.lower() for m in pwc_methods}

        if not pwc_names:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "note": "no_ground_truth"}

        true_positives = len(extracted_names & pwc_names)
        precision = true_positives / len(extracted_names) if extracted_names else 0
        recall = true_positives / len(pwc_names) if pwc_names else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "extracted_count": len(extracted_names),
            "ground_truth_count": len(pwc_names),
        }
