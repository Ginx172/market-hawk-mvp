"""
Citation formatter for Knowledge Advisor responses.
Formats RAG results with proper academic-style citations.
"""
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FormattedResponse:
    """Formatted response with citations."""

    text: str
    citations: List[str]
    summary: str
    source_count: int


class CitationFormatter:
    """Formats RAG responses with proper citations and source attribution."""

    @staticmethod
    def format_response(answer: str, sources: list, query: str) -> FormattedResponse:
        """
        Format a RAG response with numbered citations.

        Args:
            answer: LLM-generated answer text.
            sources: List of RAGResult (or similar) objects with ``source``
                     and optional ``page`` attributes.
            query: Original question (kept for future use / logging).

        Returns:
            FormattedResponse with formatted text and citation list.
        """
        seen: set = set()
        citations: List[str] = []
        for src in sources:
            source_name = getattr(src, "source", str(src))
            page = getattr(src, "page", None)
            key = f"{source_name}:{page}"
            if key not in seen:
                seen.add(key)
                if page:
                    citations.append(f"{source_name}, p.{page}")
                else:
                    citations.append(source_name)

        lines = [answer, "", "---", "📚 Sources:"]
        for i, cite in enumerate(citations, 1):
            lines.append(f"  [{i}] {cite}")

        summary = (
            f"Answer based on {len(citations)} source(s) from the trading knowledge base."
        )

        return FormattedResponse(
            text="\n".join(lines),
            citations=citations,
            summary=summary,
            source_count=len(citations),
        )

    @staticmethod
    def format_for_brain(sources: list, max_sources: int = 3) -> str:
        """
        Compact format for Brain orchestrator consumption.

        Args:
            sources: List of RAGResult objects.
            max_sources: Maximum number of sources to include.

        Returns:
            Pipe-separated string of ``[source] text`` snippets.
        """
        parts = []
        for src in sources[:max_sources]:
            source_name = getattr(src, "source", "unknown")
            text = getattr(src, "text", "")[:150]
            parts.append(f"[{source_name}] {text}")
        return " | ".join(parts)

    @staticmethod
    def extract_direction_hints(sources: list) -> Dict[str, Any]:
        """
        Analyze retrieved chunks for directional trading hints.

        Counts occurrences of bullish and bearish terminology across all
        source chunks and returns an aggregated direction signal.

        Args:
            sources: List of RAGResult objects that have a ``text`` attribute.

        Returns:
            Dict with keys:
                - ``direction``: "BULLISH", "BEARISH", or "NEUTRAL"
                - ``confidence``: float in [0.0, 0.7]
                - ``bullish_signals``: raw bullish term count
                - ``bearish_signals``: raw bearish term count
        """
        bullish_terms = {
            "buy",
            "long",
            "bullish",
            "accumulation",
            "support",
            "breakout",
            "uptrend",
            "demand zone",
            "order block bullish",
            "buy signal",
            "golden cross",
            "oversold",
            "reversal up",
            "higher high",
            "higher low",
            "bull flag",
            "ascending",
        }
        bearish_terms = {
            "sell",
            "short",
            "bearish",
            "distribution",
            "resistance",
            "breakdown",
            "downtrend",
            "supply zone",
            "sell signal",
            "death cross",
            "overbought",
            "reversal down",
            "lower low",
            "lower high",
            "bear flag",
            "descending",
        }

        bullish_count = 0
        bearish_count = 0

        for src in sources:
            text = getattr(src, "text", "").lower()
            for term in bullish_terms:
                bullish_count += text.count(term)
            for term in bearish_terms:
                bearish_count += text.count(term)

        total = bullish_count + bearish_count
        if total == 0:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "bullish_signals": 0,
                "bearish_signals": 0,
            }

        bull_ratio = bullish_count / total
        if bull_ratio > 0.6:
            direction = "BULLISH"
            confidence = min(0.3 + (bull_ratio - 0.6) * 2, 0.7)
        elif bull_ratio < 0.4:
            direction = "BEARISH"
            confidence = min(0.3 + (0.4 - bull_ratio) * 2, 0.7)
        else:
            direction = "NEUTRAL"
            confidence = 0.2

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
        }
