# loaders/factory.py
# Single entry point the rest of the app imports — everything else
# just calls load_any_file() and doesn't need to know which loader
# handles which extension. This is the Factory pattern.

import os

from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.txt_loader import load_txt
from loaders.tabular_loader import load_csv_excel
from loaders.pptx_loader import load_pptx

_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".csv": load_csv_excel,
    ".xlsx": load_csv_excel,
    ".xls": load_csv_excel,
    ".pptx": load_pptx,
}


def load_any_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"📂 Loading {ext} file: {os.path.basename(file_path)}")

    loader_fn = _LOADERS.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader_fn(file_path)
