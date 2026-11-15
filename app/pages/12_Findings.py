from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.ui import load_ui

st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()

def main() -> None:
    st.title("Findings")

    st.markdown(
        """
        This page collects the main findings from my analysis so far.
        They come from my README and represent the best summary of what I’ve learned
        about my own play.
        """
    )

    readme_path = Path(__file__).parent.parent.parent / "README.md"
    if readme_path.exists():
        st.markdown(readme_path.read_text())
    else:
        st.warning("README.md not found at project root.")


if __name__ == "__main__":
    main()
