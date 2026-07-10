# core/indexer.py
# Handles turning a list of uploaded file paths into a searchable
# Chroma vectorstore: load -> chunk -> embed -> persist.

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import TEXT_HEAVY_TYPES, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR
from loaders import load_any_file
from core.embeddings import get_embeddings


def load_and_index_files(file_paths):
    """
    Handles any supported file type, not just PDFs.
    Text-heavy formats (PDF/DOCX/TXT/PPTX) get split into overlapping
    chunks; CSV/Excel rows are already small enough and skip splitting.
    """
    all_chunks = []

    for file_path in file_paths:
        try:
            documents = load_any_file(file_path)
            file_type = documents[0].metadata.get("file_type", "Unknown")

            if file_type in TEXT_HEAVY_TYPES:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(documents)
            else:
                chunks = documents

            all_chunks.extend(chunks)
            print(f"✅ {len(chunks)} chunks from {os.path.basename(file_path)}")

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            continue

    print(f"\n🔢 Total chunks: {len(all_chunks)}")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print("✅ All files indexed!")
    return vectorstore
