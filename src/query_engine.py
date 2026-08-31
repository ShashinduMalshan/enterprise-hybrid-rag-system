"""
Hybrid Retrieval & Postprocessing Query Engine.

Assembles the two-stage RAG inference pipeline:
1. Hybrid Search (Vector + BM25) with Reciprocal Rank Fusion (RRF).
2. Two-Stage Context Reranking via LLMRerank (Top 20 -> Top 5).
3. Grounded Synthesis via Google Gemini with source node attribution.
"""

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import LLMRerank
from src.build_index import build_indexes


def build_query_engine(
    similarity_top_k: int = 20,
    rerank_top_n: int = 5,
    rerank_batch_size: int = 5
) -> RetrieverQueryEngine:
    """
    Constructs the end-to-end Hybrid RAG query engine.

    Args:
        similarity_top_k (int): Number of candidate nodes retrieved by each retriever (Default: 20).
        rerank_top_n (int): Final number of high-relevance nodes passed to the LLM (Default: 5).
        rerank_batch_size (int): Batch size for LLM candidate reranking evaluations.

    Returns:
        RetrieverQueryEngine: Fully configured query engine ready for user queries.
    """
    # 1. Initialize dual indices
    vector_index, bm25_retriever, _ = build_indexes()

    # 2. Dense Semantic Vector Retriever
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=similarity_top_k,
    )

    # 3. Hybrid Fusion Retriever with Reciprocal Rank Fusion (RRF)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        similarity_top_k=similarity_top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    # 4. LLM-Powered Cross-Encoder Reranker
    reranker = LLMRerank(
        top_n=rerank_top_n,
        choice_batch_size=rerank_batch_size,
    )

    # 5. Assemble Retriever Query Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
    )

    return query_engine