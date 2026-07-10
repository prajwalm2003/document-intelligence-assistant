# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- API / model config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0

# --- Embeddings ---
# Runs 100% locally
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CSV_ROW_CHUNK_SIZE = 50

# --- Storage ---
CHROMA_PERSIST_DIR = "./chroma_db"

# --- File types ---
TEXT_HEAVY_TYPES = ["PDF", "DOCX", "TXT", "PPTX"]
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls", ".pptx"]
