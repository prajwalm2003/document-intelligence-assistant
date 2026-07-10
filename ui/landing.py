# ui/landing.py
# The empty-state screen shown before any documents are indexed.

import streamlit as st


def render_landing():
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
