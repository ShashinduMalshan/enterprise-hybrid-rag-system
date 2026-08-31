# SimpleDirectoryReader is a built-in LlamaIndex class that can read documents from a directory 
# and convert them into LlamaIndex’s Document format. It supports various file types like PDFs, 
# text files, markdown files, CSVs, PPTX, and JSONL.
from llama_index.core import SimpleDirectoryReader
from src.config import RAW_DIR


def load_documents():
    return SimpleDirectoryReader(
        input_dir=str(RAW_DIR),
        recursive=True, # Tells LlamaIndex to also look inside subfolders of data/raw/.
        required_exts=[".pdf", ".md", ".txt", ".csv", ".pptx", ".jsonl"], # Tells LlamaIndex to load only files with these extensions.
    ).load_data()

# It reads the files, extracts text, attaches metadata like file name/path, and 
# returns a list of LlamaIndex Document objects.
'''
It returns something like:

[
    Document(text="AcmeCloud Refund Policy 2024...", metadata={...}),
    Document(text="# Enterprise Assistant FAQ...", metadata={...}),
    Document(text="plan,monthly_price_usd,...", metadata={...}),
]

'''