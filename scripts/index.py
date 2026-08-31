from src.config import setup_llamaindex
from src.build_index import build_indexes


def main():
    # Setup LlamaIndex configuration and environment variables
    setup_llamaindex()
    # Build the vector index and BM25 retriever, and get the list of nodes that were indexed.
    vector_index, bm25_retriever, nodes = build_indexes()

    print(f"Indexed {len(nodes)} nodes into ChromaDB.")
    print("BM25 retriever created.")


if __name__ == "__main__":
    main()