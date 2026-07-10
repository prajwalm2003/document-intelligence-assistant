# ui/tab_tables.py
import pandas as pd
import streamlit as st
from core.table_extractor import extract_tables


def _render_dataframe_result(result):
    st.dataframe(result["content"], use_container_width=True)
    csv = result["content"].to_csv(index=False)
    st.download_button(
        f"⬇️ Download {result['file']} as CSV",
        data=csv,
        file_name=f"{result['file']}_extracted.csv",
        mime="text/csv",
    )


def _render_text_result(result):
    st.markdown(result["content"])
    st.download_button(
        "⬇️ Download extracted data",
        data=result["content"],
        file_name=f"{result['file']}_tables.txt",
        mime="text/plain",
    )


def render_tables_tab():
    st.subheader("📊 Table & Data Extractor")
    st.markdown("Extract structured data and tables from your documents")

    if st.button("📊 Extract Tables", type="primary", use_container_width=True):
        with st.spinner("Extracting tables and structured data..."):
            results = extract_tables(st.session_state.file_paths)

        for result in results:
            st.markdown(f"### {result['file']}")
            st.caption(f"Type: {result['type']} | {result['summary']}")

            if isinstance(result["content"], pd.DataFrame):
                _render_dataframe_result(result)
            else:
                _render_text_result(result)
            st.divider()
