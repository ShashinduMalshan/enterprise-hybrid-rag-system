import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

from src.config import CHROMA_PATH
from src.load_data import load_documents


def clean_metadata_for_chroma(nodes):
    allowed_types = (str, int, float, type(None))
    for node in nodes:
        clean_metadata = {}
        for key, value in node.metadata.items():
            if isinstance(value, allowed_types):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)
        node.metadata = clean_metadata
    return nodes


def build_indexes():
    documents = load_documents()

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=80,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    nodes = clean_metadata_for_chroma(nodes)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    chroma_collection = chroma_client.get_or_create_collection("enterprise_kb")

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=20,
    )

    return vector_index, bm25_retriever, nodes
