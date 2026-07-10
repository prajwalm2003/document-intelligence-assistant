# ui/tab_chat.py
import streamlit as st
from core.qa_engine import get_answer


def _render_sources(sources):
    with st.expander("📍 View Sources"):
        for i, doc in enumerate(sources):
            st.markdown(f"""
**{doc.metadata.get('file_type', '?')} File:** `{doc.metadata.get('source_file', 'Unknown')}`
**Page/Slide:** {doc.metadata.get('page', 0) + 1}
> {doc.page_content[:250]}...
""")
            if i < len(sources) - 1:
                st.divider()


def _render_answer_bubble(answer, mode, sources):
    with st.chat_message("assistant"):
        if mode == "document":
            st.success("📄 Answered from uploaded document")
        else:
            st.info("🌐 Answered from General Knowledge")
        st.write(answer)
        if mode == "document" and sources:
            _render_sources(sources)


def render_chat_tab():
    if st.session_state.get("chat_history"):
        st.subheader("🗨️ Conversation")
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg["question"])
            _render_answer_bubble(msg["answer"], msg.get("mode"), msg.get("sources"))

    question = st.chat_input("Ask anything about your documents...")
    if question:
        with st.spinner("🔍 Searching documents..."):
            answer, sources, mode = get_answer(
                st.session_state.vectorstore,
                question,
                st.session_state.get("chat_history", []),
            )
        st.session_state.chat_history.append(
            {"question": question, "answer": answer, "sources": sources, "mode": mode}
        )

        with st.chat_message("user"):
            st.write(question)
        _render_answer_bubble(answer, mode, sources)

    if st.session_state.get("chat_history"):
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
