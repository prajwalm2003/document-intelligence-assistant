# loaders/pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    """Load a PDF using LangChain's PyPDFLoader, one Document per page."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_type"] = "PDF"
    return docs
