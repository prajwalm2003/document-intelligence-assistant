# ui/tab_summary.py
import streamlit as st
from core.summarizer import generate_summary


def render_summary_tab():
    st.subheader("📝 Document Summary Generator")
    if st.button("✨ Generate Summary", type="primary", use_container_width=True):
        with st.spinner("Summarizing your documents..."):
            summary = generate_summary(st.session_state.file_paths)
        st.markdown("### 📋 Summary")
        st.markdown(summary)
        st.download_button(
            "⬇️ Download Summary", data=summary, file_name="summary.txt", mime="text/plain"
        )
