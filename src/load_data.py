"""
Document Ingestion Pipeline for Multi-Format Enterprise Knowledge Bases.

Supports automated recursive ingestion of unstructured and semi-structured documents:
- Portable Document Formats (.pdf)
- Markdown Documentation (.md)
- Plain Text & Transcripts (.txt)
- Tabular Comma-Separated Values (.csv)
- Presentation Slides (.pptx)
- Structured JSON Lines (.jsonl)
"""

from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from src.config import RAW_DIR


def load_documents() -> List[Document]:
    """
    Recursively scans and parses raw enterprise documents into LlamaIndex Document nodes.

    Returns:
        List[Document]: Extracted document objects containing text payloads and file metadata.
        
    Raises:
        FileNotFoundError: If the source data directory does not exist.
    """
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        return []

    reader = SimpleDirectoryReader(
        input_dir=str(RAW_DIR),
        recursive=True,
        required_exts=[".pdf", ".md", ".txt", ".csv", ".pptx", ".jsonl"],
    )
    return reader.load_data()