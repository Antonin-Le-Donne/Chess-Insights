from __future__ import annotations

import sys
from pathlib import Path

from app.ui import load_ui

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()

def main() -> None:
    st.title("Pipeline decisions")

    st.markdown(
        """
        This page documents important technical choices I made while building the project:
        engine limits, thresholds, definitions, and data handling decisions.

        It’s here so future me (or anyone reading the project) can understand **why**
        the pipeline behaves the way it does.
        """
    )

    tracker_path = Path(__file__).parent.parent.parent / "PROJECT_TRACKER.md"
    if tracker_path.exists():
        st.markdown(tracker_path.read_text())
    else:
        st.warning("PROJECT_TRACKER.md not found at project root.")


if __name__ == "__main__":
    main()
