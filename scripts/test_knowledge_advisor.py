"""
Quick test — verify Knowledge Advisor connects to existing ChromaDB.
Run from: K:\\_DEV_MVP_2026\\Market_Hawk_3\\

Usage:
    python scripts/test_knowledge_advisor.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    print("\n" + "=" * 60)
    print("MARKET HAWK MVP — Knowledge Advisor Test")
    print("=" * 60)

    # 1. Test config loads
    print("\n[1/4] Loading config...")
    from config.settings import RAG_CONFIG
    print(f"  ChromaDB path: {RAG_CONFIG.chromadb_path}")
    print(f"  Collection: {RAG_CONFIG.collection_name}")
    print(f"  Embedding: {RAG_CONFIG.embedding_model}")
    print(f"  LLM: {RAG_CONFIG.llm_model}")

    # 2. Test ChromaDB connection (no Ollama needed)
    print("\n[2/4] Testing ChromaDB connection (direct)...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=RAG_CONFIG.chromadb_path)
        collections = client.list_collections()
        print(f"  Collections found: {len(collections)}")
        for col in collections:
            count = client.get_collection(col.name).count()
            print(f"    - {col.name}: {count:,} chunks")
        print("  ✅ ChromaDB connection OK")
    except Exception as e:
        print(f"  ❌ ChromaDB failed: {e}")
        return

    # 3. Test Knowledge Advisor initialization
    print("\n[3/4] Initializing Knowledge Advisor...")
    from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
    advisor = KnowledgeAdvisor()

    if advisor.initialize():
        stats = advisor.get_stats()
        print(f"  ✅ Knowledge Advisor ready")
        print(f"  Total chunks: {stats['total_chunks']:,}")
    else:
        print("  ⚠️  Full init failed (Ollama might not be running)")
        print("  ChromaDB is accessible — RAG will work when Ollama starts")

    # 4. Test retrieval (needs Ollama for embeddings)
    print("\n[4/4] Testing retrieval...")
    try:
        results = advisor.retrieve("order block trading entry criteria", n_results=3)
        if results:
            print(f"  ✅ Retrieved {len(results)} chunks")
            for i, r in enumerate(results, 1):
                print(f"    [{i}] {r.source} — {r.text[:100]}...")
        else:
            print("  ⚠️  No results (Ollama may not be running)")
    except Exception as e:
        print(f"  ⚠️  Retrieval test skipped: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    advisor.cleanup()

if __name__ == "__main__":
    main()
