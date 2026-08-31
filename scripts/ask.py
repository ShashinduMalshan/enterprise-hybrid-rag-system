"""
Interactive Command-Line Query Interface for Enterprise Hybrid RAG System.

Usage:
    python scripts/ask.py "What are the company guidelines on data retention?"
"""

import argparse
import sys
from src.config import setup_llamaindex
from src.query_engine import build_query_engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Enterprise Hybrid RAG System with natural language questions."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to query against the enterprise knowledge base."
    )
    args = parser.parse_args()

    try:
        print("⚡ Initializing Knowledge Base & Generative Pipeline...")
        setup_llamaindex()
        query_engine = build_query_engine()

        print(f"\n🔍 Query: {args.question}")
        print("─" * 70)

        response = query_engine.query(args.question)

        print("\n💡 SYNTHESIZED ANSWER:")
        print(response)

        print("\n📑 GROUNDING CITATIONS & SOURCE PASSAGES:")
        for idx, source_node in enumerate(response.source_nodes, 1):
            score = getattr(source_node, "score", "N/A")
            metadata = source_node.node.metadata
            file_name = metadata.get("file_name", "Unknown File")
            page_label = metadata.get("page_label", "N/A")

            print(f"\n[{idx}] Source: {file_name} (Page: {page_label}) | Relevance Score: {score}")
            print("─" * 70)
            print(source_node.node.text[:400].strip() + "...")

    except Exception as e:
        print(f"\n❌ Error executing query: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()