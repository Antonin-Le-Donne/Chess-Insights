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
    st.title("Debug log")

    st.markdown(
        """
        This page tracks blockers I hit and how I solved them.
        It’s basically my engineering log: what broke, why, and how I fixed it.

        Some of these entries are good candidates to be promoted into the README later.
        """
    )

    tracker_path = Path(__file__).parent.parent.parent / "PROJECT_TRACKER.md"
    if tracker_path.exists():
        st.markdown(tracker_path.read_text())
    else:
        st.warning("PROJECT_TRACKER.md not found at project root.")


if __name__ == "__main__":
    main()
