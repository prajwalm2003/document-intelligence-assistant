# ui/sidebar.py
import tempfile
import streamlit as st

from core.indexer import load_and_index_files

_ICONS = {"PDF": "📄", "DOCX": "📝", "TXT": "📃", "CSV": "📊", "XLSX": "📊", "PPTX": "📑"}


def _save_to_tempfile(uploaded_file):
    ext = "." + uploaded_file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def render_sidebar():
    with st.sidebar:
        st.header("📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt", "csv", "xlsx", "pptx"],
            accept_multiple_files=True,
            help="Supports PDF, Word, Text, CSV, Excel, PowerPoint",
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            for f in uploaded_files:
                ext = f.name.split(".")[-1].upper()
                icon = _ICONS.get(ext, "📁")
                st.markdown(f"{icon} {f.name}")

            if st.button("🔍 Index Documents", type="primary", use_container_width=True):
                file_paths = [_save_to_tempfile(f) for f in uploaded_files]

                with st.spinner("Reading and indexing all documents... ⏳"):
                    st.session_state.vectorstore = load_and_index_files(file_paths)
                    st.session_state.file_names = [f.name for f in uploaded_files]
                    st.session_state.file_paths = file_paths
                    st.session_state.chat_history = []

                st.success("🎉 All documents indexed!")
                st.balloons()

        st.divider()
        st.markdown("**Supports:** PDF · DOCX · TXT · CSV · XLSX · PPTX")
        st.markdown("**Powered by:** Gemini + LangChain + ChromaDB")
