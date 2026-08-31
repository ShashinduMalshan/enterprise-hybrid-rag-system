"""
Dual Indexing Engine: ChromaDB Dense Vector Store & BM25 Sparse Keyword Index.

Constructs context-aware document chunks via sentence splitting and builds:
1. Dense Vector Embeddings stored persistently in ChromaDB.
2. Sparse Inverted Index for lexical BM25 exact-keyword matching.
"""

from typing import List, Tuple
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

from src.config import CHROMA_PATH
from src.load_data import load_documents


def clean_metadata_for_chroma(nodes: List[BaseNode]) -> List[BaseNode]:
    """
    Sanitizes node metadata dictionaries to conform with ChromaDB type constraints.
    ChromaDB accepts primitive scalar types (str, int, float, bool, None). Complex
    nested structures are serialized to string representations.

    Args:
        nodes (List[BaseNode]): Input parsed document nodes.

    Returns:
        List[BaseNode]: Nodes with normalized scalar metadata.
    """
    allowed_types = (str, int, float, bool, type(None))
    for node in nodes:
        clean_metadata = {}
        for key, value in node.metadata.items():
            if isinstance(value, allowed_types):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)
        node.metadata = clean_metadata
    return nodes


def build_indexes(
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    collection_name: str = "enterprise_kb"
) -> Tuple[VectorStoreIndex, BM25Retriever, List[BaseNode]]:
    """
    Splits documents into overlapping chunks and constructs both dense vector and sparse indexes.

    Args:
        chunk_size (int): Token count limit per text node (Default: 512).
        chunk_overlap (int): Token sliding window overlap between consecutive nodes (Default: 80).
        collection_name (str): ChromaDB collection namespace identifier.

    Returns:
        Tuple[VectorStoreIndex, BM25Retriever, List[BaseNode]]:
            - vector_index: Searchable dense vector index backed by persistent ChromaDB.
            - bm25_retriever: Lexical exact-term retriever.
            - nodes: Full list of chunked document nodes.
    """
    # 1. Load multi-format documents
    documents = load_documents()

    # 2. Chunking with contextual overlap
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = clean_metadata_for_chroma(nodes)

    # 3. Persistent ChromaDB Vector Store
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Dense Vector Index Creation
    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    # 5. Sparse Keyword BM25 Index Creation
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=20,
    )

    return vector_index, bm25_retriever, nodes