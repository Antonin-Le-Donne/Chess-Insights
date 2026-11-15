"""Project log: the decisions, problems and sessions recorded in
PROJECT_TRACKER.md, one readable entry each (replaces the former Decisions
and Debug Log pages, which both dumped the whole file).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from app.filters import render_filters_not_applied
from app.ui import load_ui

TRACKER_PATH = Path(__file__).parent.parent.parent / "PROJECT_TRACKER.md"

# Unescaped pipe: the tracker writes literal pipes as "\|".
CELL_SPLIT = re.compile(r"(?<!\\)\|")
BOLD = re.compile(r"\*\*(.+?)\*\*")


@st.cache_data
def load_sections(path: str, modified: float) -> dict[str, str]:
    """Markdown body of each "## " section; `modified` busts the cache."""
    text = Path(path).read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(lines).strip()
            current, lines = line[3:].strip(), []
        elif current:
            lines.append(line)
    if current:
        sections[current] = "\n".join(lines).strip()
    return sections


def parse_table(markdown: str) -> pd.DataFrame:
    """The first markdown table of a section as a DataFrame."""
    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    if len(rows) < 2:
        return pd.DataFrame()

    def cells(line: str) -> list[str]:
        return [cell.strip().replace("\\|", "|") for cell in CELL_SPLIT.split(line.strip())[1:-1]]

    header = cells(rows[0])
    body = [cells(line) for line in rows[2:]]
    body = [row + [""] * (len(header) - len(row)) for row in body]
    return pd.DataFrame([row[: len(header)] for row in body], columns=header)


def short_title(text: str, limit: int = 110) -> str:
    """The first bold phrase of an entry, or its beginning."""
    match = BOLD.search(text)
    title = match.group(1) if match else text
    title = title.rstrip(".:")
    return title if len(title) <= limit else title[: limit - 1].rsplit(" ", 1)[0] + "…"


def render_entries(table: pd.DataFrame, title_column: str, body_columns: list[str], query: str) -> None:
    """One expander per row, newest first, filtered by the search box."""
    if table.empty:
        st.info("Nothing recorded in this section yet.")
        return
    rows = table.iloc[::-1]
    if query:
        mask = rows.apply(lambda row: query.lower() in " ".join(row).lower(), axis=1)
        rows = rows[mask]
    st.caption(f"{len(rows):,} entries" + (f" matching “{query}”" if query else "") + ", newest first.")
    for _, row in rows.iterrows():
        with st.expander(f"{row.iloc[0]} — {short_title(row[title_column])}"):
            for column in [title_column, *body_columns]:
                if row.get(column):
                    st.markdown(f"**{column}:** {row[column]}")


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def main() -> None:
    st.title("Project log")
    render_filters_not_applied("this page shows the project's written log, not game data.")
    st.markdown(
        "**Question:** why does the project work the way it does — what was decided, "
        "what broke, and how it was fixed?"
    )

    if not TRACKER_PATH.exists():
        st.warning("PROJECT_TRACKER.md not found at the project root.")
        return
    sections = load_sections(str(TRACKER_PATH), TRACKER_PATH.stat().st_mtime)

    decisions = parse_table(sections.get("Decisions log", ""))
    blockers = parse_table(sections.get("Blockers", ""))
    sessions = parse_table(sections.get("Session log", ""))
    open_problems = blockers[blockers.get("Resolved on", pd.Series(dtype=str)).isin(["", "—"])] if not blockers.empty else blockers

    st.markdown(
        f"#### {len(decisions)} recorded decisions, {len(blockers)} problems found and "
        f"{len(blockers) - len(open_problems)} of them fixed, over {len(sessions)} work sessions."
    )

    query = st.text_input("Search the log", placeholder="e.g. leak, bootstrap, chess.com, M3…")

    tab_decisions, tab_problems, tab_findings, tab_sessions = st.tabs(
        ["Decisions", "Problems and fixes", "Findings log", "Work sessions"]
    )
    with tab_decisions:
        render_entries(decisions, "Decision", ["Why"], query)
    with tab_problems:
        if not open_problems.empty:
            st.markdown(f"**Still open:** {len(open_problems)}")
        render_entries(blockers, "Blocker", ["Resolved on", "How"], query)
    with tab_findings:
        st.caption("The analysis results as they were recorded, before being written up in the README.")
        st.markdown(sections.get("Findings", "Nothing recorded yet."))
    with tab_sessions:
        render_entries(sessions, "Did", ["Next step"], query)


if __name__ == "__main__":
    main()
