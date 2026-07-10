# app.py
# Entry point — sets up the page and wires the sidebar + tabs together.
# All the actual logic lives in core/, loaders/, and ui/.

import streamlit as st

from ui.sidebar import render_sidebar
from ui.landing import render_landing
from ui.tab_chat import render_chat_tab
from ui.tab_summary import render_summary_tab
from ui.tab_qa import render_qa_tab
from ui.tab_tables import render_tables_tab

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Document Intelligence Assistant")
st.markdown("Upload any document → Chat, Summarize, Generate Q&A, Extract Tables")
st.divider()

render_sidebar()

if "vectorstore" not in st.session_state:
    render_landing()
else:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "📝 Summary", "❓ Q&A Generator", "📊 Table Extractor"]
    )

    with tab1:
        render_chat_tab()
    with tab2:
        render_summary_tab()
    with tab3:
        render_qa_tab()
    with tab4:
        render_tables_tab()
