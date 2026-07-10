# loaders/txt_loader.py
from loaders.base import make_document


def load_txt(file_path):
    """Simplest loader — just read the file as plain text."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [make_document(text, file_path, "TXT")]
