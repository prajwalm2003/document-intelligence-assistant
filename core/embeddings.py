# core/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME


def get_embeddings():
    # Runs 100% locally - no API key, no rate limits, no cost.
    # Downloads once (~90MB) and is cached on disk after that.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
