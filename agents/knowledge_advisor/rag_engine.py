"""
SCRIPT NAME: rag_engine.py
====================================
Execution Location: K:\_DEV_MVP_2026\Market_Hawk_3\agents\knowledge_advisor\
Purpose: Knowledge Advisor — ChromaDB RAG Engine for Trading Literature
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

RAG system with 140K+ chunks from 280+ trading books.
Target: 1,600+ documents from J:\E-Books\Trading Database
Uses MMR retrieval for diverse, high-quality responses.
"""

import gc
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("market_hawk.knowledge_advisor")


@dataclass
class RAGResult:
    """Single result from RAG retrieval."""
    text: str
    source: str
    page: Optional[int] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = None


class KnowledgeAdvisor:
    """
    Knowledge Advisor Agent — RAG-powered trading literature expert.

    Wraps ChromaDB with sentence-transformer embeddings and MMR retrieval.
    Designed to be called by the Brain orchestrator.

    Usage:
        advisor = KnowledgeAdvisor()
        results = advisor.query("What are order block entry criteria?")
        for r in results:
            print(f"[{r.source}] {r.text[:200]}")
    """

    def __init__(self, config=None):
        from config.settings import RAG_CONFIG
        self.config = config or RAG_CONFIG
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False

    def initialize(self) -> bool:
        """Lazy initialization — load ChromaDB and embeddings."""
        if self._initialized:
            return True

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            logger.info("Initializing Knowledge Advisor...")
            logger.info("  ChromaDB path: %s", self.config.chromadb_path)
            logger.info("  Collection: %s", self.config.collection_name)

            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )

            self._client = chromadb.PersistentClient(path=self.config.chromadb_path)

            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

            doc_count = self._collection.count()
            logger.info("Knowledge Advisor ready — %d chunks available", doc_count)
            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize Knowledge Advisor: %s", str(e))
            return False

    def query(self, question: str, n_results: int = None,
              mmr_lambda: float = None) -> List[RAGResult]:
        """
        Query the knowledge base with MMR diversity.

        Args:
            question: Natural language question
            n_results: Number of results (default from config)
            mmr_lambda: Diversity parameter (0=max diversity, 1=max relevance)
        """
        if not self._initialized:
            if not self.initialize():
                return []

        n_results = n_results or self.config.n_results
        mmr_lambda = mmr_lambda if mmr_lambda is not None else self.config.mmr_lambda

        try:
            raw_results = self._collection.query(
                query_texts=[question],
                n_results=min(n_results * 3, 20),
                include=["documents", "metadatas", "distances"]
            )

            if not raw_results or not raw_results["documents"][0]:
                logger.warning("No results for: %s", question[:100])
                return []

            results = self._apply_mmr(
                documents=raw_results["documents"][0],
                metadatas=raw_results["metadatas"][0],
                distances=raw_results["distances"][0],
                n_results=n_results,
                mmr_lambda=mmr_lambda
            )

            logger.info("Query '%s...' → %d results", question[:50], len(results))
            return results

        except Exception as e:
            logger.error("Query failed: %s", str(e))
            return []

    def _apply_mmr(self, documents, metadatas, distances, n_results, mmr_lambda):
        """Maximal Marginal Relevance for diverse results."""
        if len(documents) <= n_results:
            return [
                RAGResult(text=doc, source=meta.get("source", "unknown"),
                          page=meta.get("page"), relevance_score=1.0 - dist, metadata=meta)
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]

        selected = [0]
        remaining = list(range(1, len(documents)))

        while len(selected) < n_results and remaining:
            best_idx = None
            best_score = -float('inf')

            for idx in remaining:
                relevance = 1.0 - distances[idx]
                max_sim = 0.0
                for sel_idx in selected:
                    words_a = set(documents[idx].lower().split())
                    words_b = set(documents[sel_idx].lower().split())
                    if words_a or words_b:
                        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
                        max_sim = max(max_sim, overlap)

                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [
            RAGResult(text=documents[i], source=metadatas[i].get("source", "unknown"),
                      page=metadatas[i].get("page"), relevance_score=1.0 - distances[i],
                      metadata=metadatas[i])
            for i in selected
        ]

    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        """Brain-compatible interface."""
        context = context or {}
        timeframe = context.get("timeframe", "4h")
        query = (f"What does trading literature recommend for {symbol} "
                 f"on {timeframe}? Entry criteria, risk management, smart money concepts.")

        results = self.query(query, n_results=3)
        if not results:
            return {"recommendation": "HOLD", "confidence": 0.0,
                    "reasoning": "No relevant knowledge found"}

        reasoning = " | ".join([f"[{r.source}] {r.text[:150]}" for r in results])
        return {
            "recommendation": "HOLD",
            "confidence": max(r.relevance_score for r in results),
            "reasoning": f"Knowledge insights: {reasoning[:500]}",
            "metadata": {"sources": [r.source for r in results]}
        }

    def get_stats(self) -> Dict:
        if not self._initialized:
            self.initialize()
        if self._collection:
            return {"total_chunks": self._collection.count(),
                    "collection": self.config.collection_name,
                    "embedding_model": self.config.embedding_model}
        return {"status": "not_initialized"}

    def cleanup(self):
        self._collection = self._client = self._embedding_fn = None
        self._initialized = False
        gc.collect()
        logger.info("Knowledge Advisor cleanup complete")
