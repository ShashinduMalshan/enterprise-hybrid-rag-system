from src.config import setup_llamaindex
from src.build_index import build_indexes


def main():
    setup_llamaindex()
    vector_index, bm25_retriever, nodes = build_indexes()

    print(f"Indexed {len(nodes)} nodes into ChromaDB.")
    print("BM25 retriever created.")


if __name__ == "__main__":
    main()
