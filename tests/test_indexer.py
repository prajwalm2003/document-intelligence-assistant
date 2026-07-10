# tests/test_indexer.py
# Tests the chunking/indexing decision logic without needing a live
# Gemini API key or a real embedding model download — the embeddings
# and Chroma calls are mocked out since those are external services.

from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


@patch("core.indexer.Chroma")
@patch("core.indexer.get_embeddings")
@patch("core.indexer.load_any_file")
def test_text_heavy_files_get_split(mock_load_any_file, mock_get_embeddings, mock_chroma):
    from core.indexer import load_and_index_files

    long_text = "word " * 500  # long enough to be split into multiple chunks
    mock_load_any_file.return_value = [
        Document(page_content=long_text, metadata={"file_type": "TXT"})
    ]
    mock_chroma.from_documents.return_value = MagicMock()

    load_and_index_files(["fake.txt"])

    call_args = mock_chroma.from_documents.call_args
    indexed_docs = call_args.kwargs["documents"]
    assert len(indexed_docs) > 1  # got split into more than one chunk


@patch("core.indexer.Chroma")
@patch("core.indexer.get_embeddings")
@patch("core.indexer.load_any_file")
def test_csv_rows_are_not_resplit(mock_load_any_file, mock_get_embeddings, mock_chroma):
    from core.indexer import load_and_index_files

    mock_load_any_file.return_value = [
        Document(page_content="overview", metadata={"file_type": "CSV/Excel"}),
        Document(page_content="rows 0-50", metadata={"file_type": "CSV/Excel"}),
    ]
    mock_chroma.from_documents.return_value = MagicMock()

    load_and_index_files(["fake.csv"])

    call_args = mock_chroma.from_documents.call_args
    indexed_docs = call_args.kwargs["documents"]
    assert len(indexed_docs) == 2  # untouched, not re-split


@patch("core.indexer.Chroma")
@patch("core.indexer.get_embeddings")
@patch("core.indexer.load_any_file")
def test_load_error_on_one_file_does_not_stop_others(
    mock_load_any_file, mock_get_embeddings, mock_chroma
):
    from core.indexer import load_and_index_files

    def side_effect(path):
        if path == "bad.pdf":
            raise ValueError("corrupt file")
        return [Document(page_content="ok", metadata={"file_type": "TXT"})]

    mock_load_any_file.side_effect = side_effect
    mock_chroma.from_documents.return_value = MagicMock()

    load_and_index_files(["bad.pdf", "good.txt"])

    call_args = mock_chroma.from_documents.call_args
    indexed_docs = call_args.kwargs["documents"]
    assert len(indexed_docs) == 1  # only the good file made it through
