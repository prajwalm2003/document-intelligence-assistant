# loaders/base.py
# Small shared helper so every loader builds metadata the same way
# instead of repeating the same dict literal in five files.

import os
from langchain_core.documents import Document


def make_document(text, source_path, file_type, page=0):
    """Build a LangChain Document with consistent metadata fields."""
    return Document(
        page_content=text,
        metadata={
            "source": source_path,
            "file_type": file_type,
            "page": page,
            "source_file": os.path.basename(source_path),
        },
    )
