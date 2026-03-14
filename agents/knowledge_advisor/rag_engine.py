"""
SCRIPT NAME: rag_engine.py
====================================
Execution Location: K:\\_DEV_MVP_2026\\Market_Hawk_3\\agents\\knowledge_advisor\\
Purpose: Knowledge Advisor — ChromaDB RAG Engine for Trading Literature
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01
Last Modified: 2026-03-01

Connects to EXISTING ChromaDB v2 at:
    K:\\_DEV_MVP_2026\\Agent_Trading_AI\\AgentTradingAI\\baza_date_vectoriala_v2\\
    Collection: "algo_trading" — 140K+ chunks from 1,000+ trading books
    Embeddings: nomic-embed-text (via Ollama)
    Retrieval: MMR with k=15, fetch_k=60, lambda=0.7

DEPENDENCIES:
    pip install chromadb langchain-chroma langchain-ollama langchain langchain-core
"""

import gc
import sys
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from agents.knowledge_advisor.query_cache import QueryCache
from agents.knowledge_advisor.citation_formatter import CitationFormatter, FormattedResponse

logger = logging.getLogger("market_hawk.knowledge_advisor")

# Minimum confidence assigned to a NEUTRAL direction signal in analyze()
_NEUTRAL_MIN_CONFIDENCE: float = 0.3


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RAGResult:
    """Single result from RAG retrieval."""
    text: str
    source: str
    page: Optional[int] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Full response from Knowledge Advisor including LLM answer."""
    answer: str
    sources: List[RAGResult]
    query: str
    response_time: float = 0.0
    chunks_retrieved: int = 0


# ============================================================
# KNOWLEDGE ADVISOR
# ============================================================

class KnowledgeAdvisor:
    """
    Knowledge Advisor Agent — RAG-powered trading literature expert.

    Connects to existing ChromaDB v2 (140K+ chunks from 1,000+ books).
    Uses nomic-embed-text embeddings via Ollama and MMR retrieval.
    Designed to be called by the Brain orchestrator.

    Usage:
        advisor = KnowledgeAdvisor()
        advisor.initialize()

        # Simple retrieval (chunks only)
        results = advisor.retrieve("What are order block entry criteria?")

        # Full RAG with LLM answer
        response = advisor.query("How to identify smart money accumulation?")
        print(response.answer)
        for src in response.sources:
            print(f"  [{src.source}] {src.text[:100]}")

        # Brain-compatible interface
        agent_response = advisor.analyze("AAPL", {"timeframe": "4h"})
    """

    def __init__(self, config=None):
        """
        Initialize the Knowledge Advisor.

        Args:
            config: RAGConfig instance (uses default from settings if None)
        """
        if config is None:
            # Import here to avoid circular imports
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from config.settings import RAG_CONFIG
            config = RAG_CONFIG

        self.config = config
        self._vectorstore = None
        self._retriever = None
        self._llm = None
        self._chain = None
        self._initialized = False
        self._collection_count = 0
        self._cache = QueryCache()

    def initialize(self) -> bool:
        """
        Initialize ChromaDB connection, embeddings, and LLM.
        Called automatically on first query if not already initialized.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings, ChatOllama
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            logger.info("=" * 60)
            logger.info("KNOWLEDGE ADVISOR — Initializing")
            logger.info("=" * 60)
            logger.info("  ChromaDB: %s", self.config.chromadb_path)
            logger.info("  Collection: %s", self.config.collection_name)
            logger.info("  Embeddings: %s", self.config.embedding_model)
            logger.info("  LLM: %s", self.config.llm_model)

            # 1. Initialize embeddings (same as 04_rag_trading.py)
            logger.info("Loading embedding model...")
            embeddings = OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.ollama_host,
            )

            # 2. Connect to existing ChromaDB
            logger.info("Connecting to ChromaDB...")
            self._vectorstore = Chroma(
                persist_directory=self.config.chromadb_path,
                collection_name=self.config.collection_name,
                embedding_function=embeddings,
            )

            # 3. Verify connection
            self._collection_count = self._vectorstore._collection.count()
            logger.info("  ✅ Connected — %d chunks available", self._collection_count)

            # 4. Create MMR retriever
            self._retriever = self._vectorstore.as_retriever(
                search_type=self.config.retrieval_type,
                search_kwargs={
                    "k": self.config.n_results,
                    "fetch_k": self.config.fetch_k,
                    "lambda_mult": self.config.mmr_lambda,
                },
            )
            logger.info("  ✅ MMR Retriever configured (k=%d, fetch_k=%d, λ=%.1f)",
                         self.config.n_results, self.config.fetch_k, self.config.mmr_lambda)

            # 5. Initialize LLM
            logger.info("Connecting to Ollama LLM (%s)...", self.config.llm_model)
            self._llm = ChatOllama(
                model=self.config.llm_model,
                base_url=self.config.ollama_host,
                temperature=0.3,
                num_ctx=8192,
            )

            # 6. Build RAG chain
            prompt = ChatPromptTemplate.from_template(
                """You are a senior trading strategist with deep expertise in algorithmic trading,
smart money concepts, technical analysis, and risk management. You have access to
a comprehensive library of 1,000+ professional trading books.

Based on the following context from trading literature, provide a detailed,
actionable answer. Include specific strategies, entry/exit criteria, and
risk management rules where applicable.

CONTEXT FROM TRADING LITERATURE:
{context}

QUESTION: {question}

DETAILED ANSWER (with practical trading recommendations):"""
            )

            def format_docs(docs):
                """Format retrieved documents for the prompt."""
                formatted = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    formatted.append(f"[Source {i}: {source}, p.{page}]\n{doc.page_content}")
                return "\n\n---\n\n".join(formatted)

            self._format_docs = format_docs

            self._chain = (
                {"context": self._retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self._llm
                | StrOutputParser()
            )

            logger.info("  ✅ RAG Chain built successfully")
            logger.info("=" * 60)
            logger.info("KNOWLEDGE ADVISOR — READY (%d chunks)", self._collection_count)
            logger.info("=" * 60)

            self._initialized = True
            return True

        except Exception:
            logger.exception("Failed to initialize Knowledge Advisor")
            logger.error("   Make sure Ollama is running: ollama serve")
            logger.error("   And nomic-embed-text is pulled: ollama pull nomic-embed-text")
            return False

    # ============================================================
    # RETRIEVAL (chunks only, no LLM)
    # ============================================================

    def retrieve(self, question: str, n_results: int = None) -> List[RAGResult]:
        """
        Retrieve relevant chunks without LLM generation.
        Fast — useful for Brain to get raw knowledge.

        Args:
            question: Natural language question
            n_results: Override number of results

        Returns:
            List of RAGResult with source attribution
        """
        if not self._initialized:
            if not self.initialize():
                return []

        try:
            start_time = time.time()
            effective_n = n_results if n_results else self.config.n_results

            # Check cache first
            cached = self._cache.get(question, effective_n)
            if cached is not None:
                logger.debug("Cache hit for '%s...' (n=%d)", question[:50], effective_n)
                return cached

            if n_results and n_results != self.config.n_results:
                # Create temporary retriever with different k
                docs = self._vectorstore.max_marginal_relevance_search(
                    question,
                    k=n_results,
                    fetch_k=max(n_results * 4, self.config.fetch_k),
                    lambda_mult=self.config.mmr_lambda,
                )
            else:
                docs = self._retriever.invoke(question)

            results = []
            for doc in docs:
                results.append(RAGResult(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    relevance_score=doc.metadata.get("relevance_score", 0.0),
                    metadata=doc.metadata,
                ))

            elapsed = time.time() - start_time
            logger.info("Retrieved %d chunks for '%s...' in %.2fs",
                         len(results), question[:50], elapsed)

            # Store in cache
            self._cache.put(question, effective_n, results)
            return results

        except Exception:
            logger.exception("Retrieval failed")
            return []

    # ============================================================
    # FULL RAG QUERY (retrieval + LLM answer)
    # ============================================================

    def query(self, question: str) -> RAGResponse:
        """
        Full RAG query: retrieve chunks + generate LLM answer.

        Retrieves documents once and feeds the formatted context directly to
        the LLM, avoiding the double-retrieval that occurred when the chain
        internally called the retriever a second time.

        Args:
            question: Natural language question

        Returns:
            RAGResponse with answer, sources, and timing
        """
        if not self._initialized:
            if not self.initialize():
                return RAGResponse(
                    answer="Knowledge Advisor not initialized. Is Ollama running?",
                    sources=[], query=question
                )

        try:
            start_time = time.time()

            # Retrieve documents once (cache-aware)
            docs_raw = self._retriever.invoke(question)

            # Build source list from the single retrieval
            sources = [
                RAGResult(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    metadata=doc.metadata,
                )
                for doc in docs_raw
            ]

            # Generate answer by passing pre-fetched context directly to LLM
            context_text = self._format_docs(docs_raw)
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_template(
                """You are a senior trading strategist with deep expertise in algorithmic trading,
smart money concepts, technical analysis, and risk management. You have access to
a comprehensive library of 1,000+ professional trading books.

Based on the following context from trading literature, provide a detailed,
actionable answer. Include specific strategies, entry/exit criteria, and
risk management rules where applicable.

CONTEXT FROM TRADING LITERATURE:
{context}

QUESTION: {question}

DETAILED ANSWER (with practical trading recommendations):"""
            )
            answer_chain = prompt | self._llm | StrOutputParser()
            answer = answer_chain.invoke({"context": context_text, "question": question})

            elapsed = time.time() - start_time
            logger.info("RAG query answered in %.2fs (%d sources)", elapsed, len(sources))

            return RAGResponse(
                answer=answer,
                sources=sources,
                query=question,
                response_time=elapsed,
                chunks_retrieved=len(sources),
            )

        except Exception as e:
            logger.exception("RAG query failed")
            return RAGResponse(
                answer=f"Query failed: {str(e)}",
                sources=[], query=question
            )

    def query_formatted(self, question: str) -> FormattedResponse:
        """
        Full RAG query that returns a citation-formatted response.

        Calls :meth:`query` and passes the result through
        :class:`~agents.knowledge_advisor.citation_formatter.CitationFormatter`.

        Args:
            question: Natural language question

        Returns:
            FormattedResponse with numbered citations and summary.
        """
        response = self.query(question)
        return CitationFormatter.format_response(
            answer=response.answer,
            sources=response.sources,
            query=question,
        )

    # ============================================================
    # BRAIN-COMPATIBLE INTERFACE
    # ============================================================

    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        """
        Brain-compatible interface: analyze a symbol using the knowledge base.

        Extracts directional hints from retrieved chunks using
        :meth:`~agents.knowledge_advisor.citation_formatter.CitationFormatter.extract_direction_hints`
        so the recommendation is no longer always HOLD.

        Args:
            symbol: Trading symbol
            context: Additional context from Brain

        Returns:
            Dict compatible with AgentResponse including a directional
            ``recommendation`` (BUY / SELL / HOLD) derived from knowledge
            base content.
        """
        context = context or {}
        timeframe = context.get("timeframe", "4h")

        query = (f"What does trading literature recommend for trading {symbol} "
                 f"on {timeframe} timeframe? Include entry criteria, "
                 f"risk management, and smart money concepts.")

        # Use retrieval only (fast) for Brain decisions
        results = self.retrieve(query, n_results=5)

        if not results:
            return {
                "recommendation": "HOLD",
                "confidence": 0.0,
                "reasoning": "No relevant knowledge found in 140K+ chunks",
            }

        # Extract directional signal from retrieved knowledge
        hints = CitationFormatter.extract_direction_hints(results)
        direction = hints["direction"]
        confidence = hints["confidence"]

        if direction == "BULLISH":
            recommendation = "BUY"
        elif direction == "BEARISH":
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
            confidence = max(confidence, _NEUTRAL_MIN_CONFIDENCE)

        reasoning = CitationFormatter.format_for_brain(results, max_sources=3)

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": f"Knowledge base insights ({direction}): {reasoning[:500]}",
            "metadata": {
                "sources": [r.source for r in results],
                "chunks_used": len(results),
                "total_knowledge_base": self._collection_count,
                "direction_hints": hints,
            }
        }

    # ============================================================
    # UTILITIES
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self._initialized:
            self.initialize()

        return {
            "total_chunks": self._collection_count,
            "collection_name": self.config.collection_name,
            "embedding_model": self.config.embedding_model,
            "chromadb_path": self.config.chromadb_path,
            "retrieval_type": self.config.retrieval_type,
            "llm_model": self.config.llm_model,
            "initialized": self._initialized,
        }

    def cleanup(self):
        """Release memory."""
        self._vectorstore = None
        self._retriever = None
        self._llm = None
        self._chain = None
        self._initialized = False
        self._cache.clear()
        gc.collect()
        logger.info("Knowledge Advisor cleanup complete")


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    """Quick test — run directly to verify Knowledge Advisor works."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    print("\n" + "=" * 60)
    print("KNOWLEDGE ADVISOR — Standalone Test")
    print("=" * 60)

    advisor = KnowledgeAdvisor()

    if not advisor.initialize():
        print("\n❌ Initialization failed!")
        print("   Make sure Ollama is running: ollama serve")
        print("   And models are pulled:")
        print("     ollama pull nomic-embed-text")
        print("     ollama pull qwen3:8b")
        sys.exit(1)

    # Print stats
    stats = advisor.get_stats()
    print(f"\n📊 Knowledge Base Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Test retrieval (fast)
    print("\n" + "-" * 60)
    print("TEST 1: Retrieval Only (fast)")
    print("-" * 60)
    results = advisor.retrieve("What are the best entry criteria for order block trades?", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Source: {r.source}")
        print(f"      Text: {r.text[:200]}...")

    # Test full RAG (slower — needs LLM)
    print("\n" + "-" * 60)
    print("TEST 2: Full RAG Query (with LLM)")
    print("-" * 60)
    response = advisor.query("How to identify smart money accumulation zones?")
    print(f"\n  Answer ({response.response_time:.1f}s):")
    print(f"  {response.answer[:500]}...")
    print(f"\n  Sources used: {response.chunks_retrieved}")
    for src in response.sources[:3]:
        print(f"    - {src.source}")

    print("\n✅ All tests passed!")
    advisor.cleanup()
