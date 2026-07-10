# core/table_extractor.py
# CSV/Excel are already tabular, so we just load them as dataframes.
# PDF/DOCX/TXT/PPTX have no native table structure, so we ask the LLM
# to pull out anything table-like or key-value structured.

import os
import pandas as pd

from core.llm import get_llm
from loaders import load_any_file

_TABULAR_EXTS = {".csv", ".xlsx", ".xls"}
_DOC_EXTS = {".pdf", ".docx", ".txt", ".pptx"}


def _extract_from_tabular(file_path, filename):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    return {
        "file": filename,
        "type": "Structured Data",
        "content": df,
        "summary": f"{len(df)} rows × {len(df.columns)} columns",
    }


def _extract_from_document(llm, file_path, filename):
    docs = load_any_file(file_path)
    text = "\n".join(d.page_content for d in docs)[:4000]

    prompt = f"""Extract ALL tables, lists, and structured data from this document.
Format each table clearly with headers and rows.
If no tables found, extract key-value pairs or bullet point lists.
Document:
{text}"""

    response = llm.invoke(prompt)
    return {
        "file": filename,
        "type": "Extracted Data",
        "content": response.content,
        "summary": "AI-extracted structured data",
    }


def extract_tables(file_paths):
    llm = get_llm()
    results = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext in _TABULAR_EXTS:
            results.append(_extract_from_tabular(file_path, filename))
        elif ext in _DOC_EXTS:
            results.append(_extract_from_document(llm, file_path, filename))

    return results
