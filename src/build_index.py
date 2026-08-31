# Imports the ChromaDB client library so we can create/open the local vector database.
import chromadb

# StorageContext is the class provided by LlamaIndex for configuring storage backends.
# VectorStoreIndex takes document nodes, creates embeddings for them, 
# and stores/searches those embeddings in a vector store like ChromaDB.
from llama_index.core import StorageContext, VectorStoreIndex
# Imports the splitter that breaks loaded documents into smaller nodes/chunks for retrieval.
from llama_index.core.node_parser import SentenceSplitter
# ChromaVectorStore is the LlamaIndex adapter/wrapper for ChromaDB.
# It lets LlamaIndex use ChromaDB as its vector database.
from llama_index.vector_stores.chroma import ChromaVectorStore
# BM25Retriever is a simple retriever that uses the BM25 algorithm to find relevant nodes 
# based on keyword matching.
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
        chunk_size=512, # means each chunk should be around 512 tokens.
        chunk_overlap=80, # means that when splitting text into chunks, there will be an overlap of 80 tokens between consecutive chunks. This helps preserve context across chunks.
    )

    # The splitter takes the loaded documents and breaks them into smaller nodes/chunks based on the specified chunk size and overlap. 
    # Each node will contain a portion of the original document's text along with its metadata.
    nodes = splitter.get_nodes_from_documents(documents)
    # Before we can store the nodes in ChromaDB, we need to ensure that their metadata is in a format that ChromaDB can handle. 
    # ChromaDB expects metadata values to be of simple types like strings, numbers, or None. If a value is more complex (e.g., a list or dict), we need to convert it to a string.
    nodes = clean_metadata_for_chroma(nodes)

    # We create a persistent ChromaDB client that will store the vector database on disk at the specified path.
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # We create a collection in ChromaDB called "enterprise_kb". 
    # A collection is like a table in a traditional database where we will store our vector embeddings and their associated metadata.
    chroma_collection = chroma_client.get_or_create_collection("enterprise_kb")

    # We create a ChromaVectorStore, which is the LlamaIndex wrapper for ChromaDB. 
    # This allows LlamaIndex to use ChromaDB as its vector store for storing and retrieving embeddings.
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    # We create a StorageContext for LlamaIndex and tell it to use our ChromaVectorStore as the vector store backend.
    # StorageContext is LlamaIndex’s storage configuration object.
    # It tells LlamaIndex where to store index data.
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # We create a VectorStoreIndex, which is the main LlamaIndex index type that uses a vector store for retrieval.
    # We pass in the nodes (text chunks) that we want to index and the storage context that tells it to use ChromaDB for storing embeddings.
    # This is the line that actually builds the vector index.
    # It takes your chunks/nodes, creates embeddings for them, and stores them in ChromaDB.
    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    # We create a BM25Retriever, which is a simple retriever that uses the BM25 algorithm to find relevant nodes based on keyword matching.
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, # Use the same chunks/nodes to build a separate BM25 keyword index for exact-term search.
        similarity_top_k=20, # means that when retrieving nodes using BM25, it will return the top 20 most relevant nodes based on keyword matching.
    )

    return vector_index, bm25_retriever, nodes