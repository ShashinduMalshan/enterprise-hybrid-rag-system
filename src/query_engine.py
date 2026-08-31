# QueryFusionRetriever is a retriever that combines results from multiple retrievers 
# (e.g., vector-based and keyword-based) and fuses them together to get a more comprehensive set of relevant nodes.
# 1. Vector retriever  -> semantic search
# 2. BM25 retriever    -> keyword search
from llama_index.core.retrievers import QueryFusionRetriever
# RetrieverQueryEngine is a LlamaIndex query engine that uses a retriever to fetch relevant nodes and 
# then generates answers based on those nodes.
# Imports the query engine class that takes a retriever, retrieves relevant nodes, 
# sends them to the LLM, and returns an answer.
from llama_index.core.query_engine import RetrieverQueryEngine
# LLMRerank is a postprocessor that takes the retrieved nodes and re-ranks them using 
# the LLM to determine which ones are most relevant for answering the question.
from llama_index.core.postprocessor import LLMRerank

from src.build_index import build_indexes

'''
1. User asks a question
2. Vector retriever finds candidate chunks
3. BM25 retriever finds candidate chunks
4. QueryFusionRetriever combines those candidates
5. LLMRerank reviews the candidates and selects the best ones
6. Send selected chunks + question to Gemini
7. Gemini generates the final answer and source nodes

'''


def build_query_engine():
    vector_index, bm25_retriever, _ = build_indexes()

    # We create a vector retriever from the vector index. This retriever will use the vector embeddings in ChromaDB to find relevant nodes based on semantic similarity.
    # 1. Take the vector index
    # 2. Convert it into a searchable retriever
    # 3. Configure it to return top 20 semantic matches
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=20,
    )

    # We create a QueryFusionRetriever that combines the results from both the vector retriever and the BM25 retriever.
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[
            vector_retriever, # This retriever will return nodes based on semantic similarity using the vector index.
            bm25_retriever,   # This retriever will return nodes based on exact keyword matches using the BM25 index.
        ],
        similarity_top_k=20, # When the QueryFusionRetriever combines results from both retrievers, it will keep the top 20 most relevant nodes based on similarity scores.
        num_queries=1, # This means that the QueryFusionRetriever will treat the input question as a single query when retrieving results from both retrievers. Tells LlamaIndex to use only one query version: the original user query. Some fusion retrievers can generate multiple rewritten versions of the query, but here we are not doing query expansion.
        mode="reciprocal_rerank", # The QueryFusionRetriever will re-rank the combined results using a reciprocal rank fusion method to determine the final relevance of each node. If a chunk appears high in both vector search and BM25 search, it gets a stronger final score.
        use_async=False, # This means that the retrieval process will be synchronous, meaning it will wait for each retriever to finish before moving on to the next step.
    )

    # We create an LLMRerank postprocessor that will take the retrieved nodes from the QueryFusionRetriever and 
    # re-rank them using the LLM to determine which ones are most relevant for answering the question.
    reranker = LLMRerank(
        top_n=5, # Keep the best 5 chunks after reranking.
        choice_batch_size=5, # Send candidate chunks to the LLM in batches of 5 for comparison/reranking.
    )

    # This creates the final query engine that will answer user questions.
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever, # Use the hybrid retriever to find candidate chunks.
        node_postprocessors=[reranker], # After retrieval, run these postprocessing steps on the retrieved nodes.
    )

    return query_engine