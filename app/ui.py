from pathlib import Path

import streamlit as st


def load_ui():
    css_path = Path(__file__).parent / "style.css"

    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )