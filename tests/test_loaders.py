# tests/test_loaders.py
# Basic unit tests for the file loaders. These don't need any API key
# since loaders only read files - no LLM calls happen here.

import os
import tempfile
import pytest

from loaders.txt_loader import load_txt
from loaders.factory import load_any_file


def test_load_txt_returns_single_document():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world\nsecond line")
        path = f.name

    try:
        docs = load_txt(path)
        assert len(docs) == 1
        assert "hello world" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "TXT"
        assert docs[0].metadata["source_file"] == os.path.basename(path)
    finally:
        os.remove(path)


def test_factory_dispatches_txt_to_txt_loader():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("dispatch check")
        path = f.name

    try:
        docs = load_any_file(path)
        assert docs[0].metadata["file_type"] == "TXT"
    finally:
        os.remove(path)


def test_factory_raises_on_unsupported_extension():
    with pytest.raises(ValueError):
        load_any_file("somefile.exe")


def test_csv_loader_produces_overview_chunk():
    from loaders.tabular_loader import load_csv_excel

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,age\nAlice,30\nBob,25\n")
        path = f.name

    try:
        docs = load_csv_excel(path)
        assert len(docs) >= 1
        assert "2 rows" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "CSV/Excel"
    finally:
        os.remove(path)
