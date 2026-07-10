# ui/tab_qa.py
import streamlit as st
from core.qa_generator import generate_qa


def render_qa_tab():
    st.subheader("❓ Auto Q&A Generator")
    st.markdown("Automatically generate MCQ questions from your documents")

    num_q = st.slider("Number of questions to generate", min_value=3, max_value=10, value=5)

    if st.button("🎯 Generate Questions", type="primary", use_container_width=True):
        with st.spinner(f"Generating {num_q} questions..."):
            qa_content = generate_qa(st.session_state.vectorstore, num_q)
        st.markdown("### 📋 Generated Questions")
        st.markdown(qa_content)
        st.download_button(
            "⬇️ Download Q&A", data=qa_content, file_name="generated_qa.txt", mime="text/plain"
        )
