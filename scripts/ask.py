import argparse
from src.config import setup_llamaindex
from src.query_engine import build_query_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    setup_llamaindex()
    query_engine = build_query_engine()

    response = query_engine.query(args.question)

    print("\nANSWER")
    print(response)

    print("\nSOURCES")
    for source_node in response.source_nodes:
        metadata = source_node.node.metadata
        score = source_node.score
        print("-" * 60)
        print("Score:", score)
        print("Source:", metadata)
        print(source_node.node.text[:500])


if __name__ == "__main__":
    main()
