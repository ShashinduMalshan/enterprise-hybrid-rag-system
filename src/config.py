"""
Configuration Module for Enterprise Hybrid RAG System.

This module initializes environment variables, filesystem paths, and global
LlamaIndex settings for Google Gemini LLM and Embedding models.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Project Root Directory Resolution
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STORAGE_DIR = ROOT_DIR / "storage"
CHROMA_PATH = STORAGE_DIR / "chroma"

# Load environment configuration
load_dotenv(ROOT_DIR / ".env")


def setup_llamaindex(
    llm_model: str = "gemini-2.5-flash",
    embedding_model: str = "gemini-embedding-001",
    temperature: float = 0.0
) -> None:
    """
    Initializes global LlamaIndex settings with Google Gemini generative and embedding models.
    
    Args:
        llm_model (str): Google Gemini model identifier for text generation.
        embedding_model (str): Google Gemini embedding model identifier.
        temperature (float): Sampling temperature (0.0 for deterministic factual responses).
        
    Raises:
        RuntimeError: If GEMINI_API_KEY environment variable is not defined.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not found. "
            "Please copy .env.example to .env and insert your API key."
        )

    # Configure global LLM generator
    resolved_llm_model = os.getenv("GEMINI_LLM_MODEL", llm_model)
    Settings.llm = GoogleGenAI(
        model=resolved_llm_model,
        api_key=api_key,
        temperature=temperature,
    )

    # Configure global dense vector embedding model
    resolved_embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", embedding_model)
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=resolved_embed_model,
        api_key=api_key,
    )