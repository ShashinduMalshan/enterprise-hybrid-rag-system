from llama_index.core import SimpleDirectoryReader
from src.config import RAW_DIR


def load_documents():
    return SimpleDirectoryReader(
        input_dir=str(RAW_DIR),
        recursive=True,
        required_exts=[".pdf", ".md", ".txt", ".csv", ".pptx", ".jsonl"],
    ).load_data()
