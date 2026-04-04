import streamlit as st
from rag_pipeline import (load_and_index_files, get_answer,
                          generate_summary, generate_qa, extract_tables)
import os
import tempfile
import pandas as pd

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Document Intelligence Assistant")
st.markdown("Upload any document → Chat, Summarize, Generate Q&A, Extract Tables")
st.divider()

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "csv", "xlsx", "pptx"],
        accept_multiple_files=True,
        help="Supports PDF, Word, Text, CSV, Excel, PowerPoint"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for f in uploaded_files:
            ext = f.name.split(".")[-1].upper()
            icons = {"PDF":"📄","DOCX":"📝","TXT":"📃",
                     "CSV":"📊","XLSX":"📊","PPTX":"📑"}
            icon = icons.get(ext, "📁")
            st.markdown(f"{icon} {f.name}")

        if st.button("🔍 Index Documents", type="primary",
                     use_container_width=True):
            file_paths = []
            for uploaded_file in uploaded_files:
                ext = "." + uploaded_file.name.split(".")[-1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp.write(uploaded_file.getbuffer())
                tmp.close()
                file_paths.append(tmp.name)

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

# ---- MAIN AREA ----
if "vectorstore" not in st.session_state:
    st.info("👈 Upload documents from the sidebar to get started!")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📄", "5 File Types", "PDF,DOCX,TXT,CSV,PPTX")
    with col2:
        st.metric("💬", "Chat History", "Remembers context")
    with col3:
        st.metric("📍", "Source Pages", "Explainable AI")
    with col4:
        st.metric("❓", "Auto Q&A", "MCQ Generator")
    with col5:
        st.metric("📊", "Table Extractor", "Structured data")
else:
    # ---- 4 TABS ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat",
        "📝 Summary",
        "❓ Q&A Generator",
        "📊 Table Extractor"
    ])

    # ==========================
    # TAB 1: CHAT
    # ==========================
    with tab1:
        if ("chat_history" in st.session_state
                and st.session_state.chat_history):
            st.subheader("🗨️ Conversation")
            for msg in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(msg["question"])
                with st.chat_message("assistant"):
                    if msg.get("mode") == "document":
                        st.success("📄 Answered from uploaded document")
                    else:
                        st.info("🌐 Answered from General Knowledge")
                    st.write(msg["answer"])
                    if msg.get("mode") == "document" and msg["sources"]:
                        with st.expander("📍 View Sources"):
                            for doc in msg["sources"]:
                                st.markdown(f"""
**{doc.metadata.get('file_type','?')} File:** `{doc.metadata.get('source_file','Unknown')}`
**Page/Slide:** {doc.metadata.get('page', 0) + 1}
> {doc.page_content[:250]}...
""")

        question = st.chat_input("Ask anything about your documents...")
        if question:
            with st.spinner("🔍 Searching documents..."):
                answer, sources, mode = get_answer(
                    st.session_state.vectorstore,
                    question,
                    st.session_state.get("chat_history", [])
                )
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources,
                "mode": mode
            })
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                if mode == "document":
                    st.success("📄 Answered from uploaded document")
                else:
                    st.info("🌐 Answered from General Knowledge")
                st.write(answer)
                if mode == "document" and sources:
                    with st.expander("📍 View Sources"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"""
**{doc.metadata.get('file_type','?')} File:** `{doc.metadata.get('source_file','Unknown')}`
**Page/Slide:** {doc.metadata.get('page', 0) + 1}
> {doc.page_content[:250]}...
""")
                            if i < len(sources)-1:
                                st.divider()

        if st.session_state.get("chat_history"):
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ==========================
    # TAB 2: SUMMARY
    # ==========================
    with tab2:
        st.subheader("📝 Document Summary Generator")
        if st.button("✨ Generate Summary", type="primary",
                     use_container_width=True):
            with st.spinner("Summarizing your documents..."):
                summary = generate_summary(st.session_state.file_paths)
            st.markdown("### 📋 Summary")
            st.markdown(summary)
            st.download_button(
                "⬇️ Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

    # ==========================
    # TAB 3: Q&A GENERATOR
    # ==========================
    with tab3:
        st.subheader("❓ Auto Q&A Generator")
        st.markdown("Automatically generate MCQ questions from your documents")

        num_q = st.slider(
            "Number of questions to generate",
            min_value=3,
            max_value=10,
            value=5
        )

        if st.button("🎯 Generate Questions", type="primary",
                     use_container_width=True):
            with st.spinner(f"Generating {num_q} questions..."):
                qa_content = generate_qa(st.session_state.vectorstore,
                                        num_q)
            st.markdown("### 📋 Generated Questions")
            st.markdown(qa_content)
            st.download_button(
                "⬇️ Download Q&A",
                data=qa_content,
                file_name="generated_qa.txt",
                mime="text/plain"
            )

    # ==========================
    # TAB 4: TABLE EXTRACTOR
    # ==========================
    with tab4:
        st.subheader("📊 Table & Data Extractor")
        st.markdown("Extract structured data and tables from your documents")

        if st.button("📊 Extract Tables", type="primary",
                     use_container_width=True):
            with st.spinner("Extracting tables and structured data..."):
                results = extract_tables(st.session_state.file_paths)

            for result in results:
                st.markdown(f"### {result['file']}")
                st.caption(f"Type: {result['type']} | {result['summary']}")

                # If it's a real dataframe (CSV/Excel)
                if isinstance(result["content"], pd.DataFrame):
                    st.dataframe(result["content"], use_container_width=True)
                    csv = result["content"].to_csv(index=False)
                    st.download_button(
                        f"⬇️ Download {result['file']} as CSV",
                        data=csv,
                        file_name=f"{result['file']}_extracted.csv",
                        mime="text/csv"
                    )
                else:
                    # AI extracted text tables
                    st.markdown(result["content"])
                    st.download_button(
                        f"⬇️ Download extracted data",
                        data=result["content"],
                        file_name=f"{result['file']}_tables.txt",
                        mime="text/plain"
                    )
                st.divider()