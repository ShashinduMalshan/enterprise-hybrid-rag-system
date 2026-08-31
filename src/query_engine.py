from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank

from src.build_index import build_indexes


def build_query_engine():
    vector_index, bm25_retriever, _ = build_indexes()

    vector_retriever = vector_index.as_retriever(
        similarity_top_k=20,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        similarity_top_k=20,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    reranker = LLMRerank(
        top_n=5,
        choice_batch_size=5,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
    )

    return query_engine
