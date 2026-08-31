import json
from pathlib import Path

from src.config import setup_llamaindex, DATA_DIR
from src.query_engine import build_query_engine


def contains_expected(text: str, expected: list[str]) -> bool:
    lower = text.lower()
    return all(item.lower() in lower for item in expected)


def source_hit(response, expected_sources: list[str]) -> bool:
    all_sources = []

    for source_node in response.source_nodes:
        metadata = source_node.node.metadata
        all_sources.append(str(metadata))

    joined = " ".join(all_sources).lower()
    return any(src.lower() in joined for src in expected_sources)


def main():
    setup_llamaindex()
    query_engine = build_query_engine()

    eval_path = DATA_DIR / "eval" / "questions.jsonl"
    questions = [
        json.loads(line)
        for line in eval_path.read_text().splitlines()
        if line.strip()
    ]

    correct = 0
    retrieval_hits = 0

    for row in questions:
        response = query_engine.query(row["question"])
        answer = str(response)

        answer_ok = contains_expected(
            answer,
            row["expected_answer_contains"],
        )

        retrieval_ok = source_hit(
            response,
            row["expected_sources"],
        )

        correct += int(answer_ok)
        retrieval_hits += int(retrieval_ok)

        print("=" * 80)
        print(row["id"], row["question"])
        print("Answer OK:", answer_ok)
        print("Retrieval OK:", retrieval_ok)
        print("Answer:", answer)

    total = len(questions)
    print("\nSUMMARY")
    print("Answer correctness:", correct / total)
    print("Retrieval hit rate:", retrieval_hits / total)


if __name__ == "__main__":
    main()