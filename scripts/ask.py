# argparse is a standard Python library for parsing command-line arguments. 
# In this script, we use argparse to allow the user to input a question when they run the script
import argparse
from src.config import setup_llamaindex
from src.query_engine import build_query_engine


def main():
    # We create an ArgumentParser object and define a required positional argument called "question".
    parser = argparse.ArgumentParser()
    # The "question" argument will be the user's query that they want to ask the system. 
    # When the user runs the script, they will provide their question as a command-line argument, and we will use that question to query our LlamaIndex knowledge base.
    parser.add_argument("question")
    # We call parser.parse_args() to parse the command-line arguments and store them in the args variable.
    args = parser.parse_args()

    setup_llamaindex()
    query_engine = build_query_engine()

    # This sends the user’s question into the RAG query engine.
    response = query_engine.query(args.question)
    '''
    When this line runs, LlamaIndex internally does:

    1. embed the question for vector search
    2. search ChromaDB for semantically similar nodes
    3. search BM25 for keyword-matching nodes
    4. fuse/combine the retrieval results
    5. rerank the candidate nodes
    6. send the question + selected chunks to Gemini
    7. return Gemini's final answer with source nodes
    
    '''

    print("\nANSWER")
    print(response)

    print("\nSOURCES")
    for source_node in response.source_nodes:
        # Each source_node contains the original text chunk (source_node.node.text), its metadata (source_node.node.metadata), and the relevance score assigned by the retriever (source_node.score).
        metadata = source_node.node.metadata
        # score is a relevance score that indicates how relevant this source node is to the user's question. Higher scores mean more relevant.
        score = source_node.score
        print("-" * 60)
        print("Score:", score)
        print("Source:", metadata)
        print(source_node.node.text[:500])

'''
Last block prints the evidence used by the RAG system:

which chunks were used
where they came from
how relevant they were
what text they contained

'''


if __name__ == "__main__":
    main()