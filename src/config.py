import os
from pathlib import Path
from dotenv import load_dotenv

# Imports LlamaIndex’s global configuration object, where we set the default LLM and embedding model.
from llama_index.core import Settings
# Imports the Gemini LLM wrapper so LlamaIndex can use Gemini to generate answers.
from llama_index.llms.google_genai import GoogleGenAI
# Imports the Gemini embedding wrapper so LlamaIndex can convert text chunks into vector embeddings.
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STORAGE_DIR = ROOT_DIR / "storage"
CHROMA_PATH = STORAGE_DIR / "chroma"

load_dotenv(ROOT_DIR / ".env")


def setup_llamaindex() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")
    
    print("Using Gemini LLM model:", os.getenv("GEMINI_LLM_MODEL"))

    Settings.llm = GoogleGenAI(
        model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
        api_key=api_key,
        temperature=0,
    )

    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        api_key=api_key,
    )