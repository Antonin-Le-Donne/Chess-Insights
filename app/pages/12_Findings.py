"""Findings: the README's results in plain language, with the full technical
version (docs/findings_details.md) one click away.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.filters import render_filters_not_applied
from app.ui import load_ui

ROOT = Path(__file__).parent.parent.parent
README_PATH = ROOT / "README.md"
DETAILS_PATH = ROOT / "docs" / "findings_details.md"

# README sections shown, in order (Setup / Run and the rest stay in the README).
SECTIONS = ["Main findings", "Predicting my blunders: the V2 model", "Honest limits", "Status"]


@st.cache_data
def read_sections(path: str, modified: float) -> dict[str, str]:
    """Markdown body of each "## " section; `modified` busts the cache."""
    sections: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(lines).strip()
            current, lines = line[3:].strip(), []
        elif current:
            lines.append(line)
    if current:
        sections[current] = "\n".join(lines).strip()
    return sections


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def main() -> None:
    st.title("Findings")
    render_filters_not_applied("this page shows the written results from the README.")
    st.markdown("**Question:** what has the analysis found so far, in plain words?")

    if not README_PATH.exists():
        st.warning("README.md not found at the project root.")
        return
    sections = read_sections(str(README_PATH), README_PATH.stat().st_mtime)

    st.caption(
        "The live numbers are on each metric's page and on the Overview; this text is the "
        "written-up version and carries the date of the data it was computed on."
    )
    for name in SECTIONS:
        if name in sections:
            st.markdown(f"## {name}")
            st.markdown(sections[name])

    if DETAILS_PATH.exists():
        with st.expander("Full technical version (definitions, intervals, definition history)"):
            st.markdown(DETAILS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
