# loaders/tabular_loader.py
# CSV/Excel need a different strategy than text files: pandas reads
# the table, then we convert rows into text chunks so the LLM can
# reason over them like any other document chunk.

import pandas as pd
from config import CSV_ROW_CHUNK_SIZE
from loaders.base import make_document


def load_csv_excel(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    docs = []

    # First chunk: a quick overview so the model knows the shape of the data.
    col_info = (
        f"This document has {len(df)} rows and {len(df.columns)} columns.\n"
        f"Columns are: {', '.join(df.columns.tolist())}\n\n"
        f"First few rows:\n{df.head(5).to_string()}"
    )
    docs.append(make_document(col_info, file_path, "CSV/Excel", page=0))

    # Remaining rows, chunked so nothing is dropped for large sheets.
    for i in range(0, len(df), CSV_ROW_CHUNK_SIZE):
        chunk = df.iloc[i : i + CSV_ROW_CHUNK_SIZE]
        text = f"Rows {i} to {i + len(chunk)}:\n{chunk.to_string()}"
        docs.append(
            make_document(
                text, file_path, "CSV/Excel", page=i // CSV_ROW_CHUNK_SIZE
            )
        )

    return docs
