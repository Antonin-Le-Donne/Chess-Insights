from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from app.filters import sidebar_filtered
from src.metrics import m3_greed_conversion

MONTHS_IN_TABLE = 12
SMALL_SAMPLE = 30


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves

def kpi_card(title: str, value, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <h3>{title}</h3>
            <p style="font-size: 28px; font-weight: 600; color: #FFFFFF;">{value}</p>
            <p style="color: #DADADA;">{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - half_width, centre + half_width


def monthly_conversion(m3: pd.DataFrame) -> pd.DataFrame:
    """M3 tracking stats per calendar month of the game's start date."""
    month = m3["date_utc"].dt.to_period("M").rename("month")
    table = m3.groupby(month).agg(
        n_games=("did_convert", "size"),
        n_converted=("did_convert", "sum"),
        median_plies=("plies_to_finish", "median"),
    )
    table["conversion"] = table["n_converted"] / table["n_games"]
    bounds = [
        wilson_interval(int(converted), int(n))
        for converted, n in zip(table["n_converted"], table["n_games"])
    ]
    table["ci_low"] = [low for low, _ in bounds]
    table["ci_high"] = [high for _, high in bounds]
    return table


def render_m3_tracking(m3: pd.DataFrame, games: pd.DataFrame) -> None:
    st.markdown("### M3 tracking — conversion from a winning position, month over month")

    if m3.empty:
        st.info("No game in the selection reaches a 80% win probability.")
        return

    table = monthly_conversion(m3)
    latest = table.index.max()
    previous = latest - 1
    latest_row = table.loc[latest]

    col1, col2 = st.columns(2)
    if previous in table.index:
        previous_row = table.loc[previous]
        delta = (latest_row["conversion"] - previous_row["conversion"]) * 100
        col1.metric(
            f"Conversion — {latest.strftime('%B %Y')}",
            f"{latest_row['conversion']:.1%}",
            delta=f"{delta:+.1f} pp vs {previous.strftime('%B %Y')}",
            help=(
                f"{int(latest_row['n_converted'])} wins out of {int(latest_row['n_games'])} "
                f"games that reached 80% (95% CI {latest_row['ci_low']:.0%}–"
                f"{latest_row['ci_high']:.0%}). Previous month: "
                f"{int(previous_row['n_converted'])}/{int(previous_row['n_games'])}."
            ),
        )
        col2.metric(
            f"Median plies to finish — {latest.strftime('%B %Y')}",
            f"{latest_row['median_plies']:.0f}",
            delta=f"{latest_row['median_plies'] - previous_row['median_plies']:+.0f} vs previous month",
            delta_color="off",
        )
    else:
        col1.metric(
            f"Conversion — {latest.strftime('%B %Y')}",
            f"{latest_row['conversion']:.1%}",
            help=f"No game reached 80% in {previous.strftime('%B %Y')}, so there is no comparison.",
        )
        col2.metric(
            f"Median plies to finish — {latest.strftime('%B %Y')}",
            f"{latest_row['median_plies']:.0f}",
        )

    notes = [
        f"Latest month with data: {latest.strftime('%B %Y')} (the data refresh date, not "
        "necessarily the current calendar month)."
    ]
    if int(latest_row["n_games"]) < SMALL_SAMPLE:
        notes.append(
            f"Only {int(latest_row['n_games'])} games this month: a monthly change of "
            "several points is within noise — read the confidence interval."
        )
    if games["perf_type"].eq("bullet").any() and games["perf_type"].ne("bullet").any():
        notes.append(
            "The selection mixes bullet with blitz/rapid. Bullet converts less, so a "
            "month with more bullet looks worse; filter one time control for a "
            "like-for-like trend."
        )
    st.caption(" ".join(notes))

    recent = table.tail(MONTHS_IN_TABLE).sort_index(ascending=False)
    display = pd.DataFrame(
        {
            "Month": recent.index.strftime("%Y-%m"),
            "Games reaching 80%": recent["n_games"].astype(int),
            "Converted": recent["n_converted"].astype(int),
            "Conversion": recent["conversion"].map("{:.1%}".format),
            "95% CI": [
                f"{low:.0%}–{high:.0%}" for low, high in zip(recent["ci_low"], recent["ci_high"])
            ],
            "Median plies to finish": recent["median_plies"].round(0).astype(int),
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def main() -> None:
    st.title("Overview")

    css_path = Path(__file__).parent.parent / "style.css"
    if css_path.exists():
        st.markdown("<style>" + css_path.read_text() + "</style>", unsafe_allow_html=True)

    games, moves = sidebar_filtered(*load_data())

    st.markdown(
        """
        <div class="header-banner">
            <h2 style="margin: 0;">My data overview</h2>
            <p style="margin: 0;">A quick summary of the games I analyze</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This page gives me a quick overview of the data I'm working with and how complete
        the engine evaluation is. It answers: **what games are included, and how much
        analysis has been done on them?**
        """
    )

    m3 = m3_greed_conversion(moves, games)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Games", f"{len(games):,}", "Games fetched from Lichess/chess.com")

    with col2:
        first_day = games["date_utc"].min()
        last_day = games["date_utc"].max()
        kpi_card(
            "Date range",
            f"{first_day:%b %Y} → {last_day:%b %Y}",
            f"{first_day:%Y-%m-%d} to {last_day:%Y-%m-%d}",
        )

    with col3:
        eval_coverage = 1 - moves["eval_cp"].isna().mean()
        kpi_card("Engine eval coverage", f"{eval_coverage:.1%}", "Moves with an engine evaluation")

    with col4:
        median_plies = m3["plies_to_finish"].median()
        kpi_card(
            "Median moves to finish",
            "—" if pd.isna(median_plies) else int(median_plies),
            "From first big advantage to game end (plies)",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    render_m3_tracking(m3, games)

    st.markdown(
        """
        <br>
        <div class="block">
            <h3>Dataset summary</h3>
            <p>- My games are fetched from online platforms and stored in parquet files.</p>
            <p>- Stockfish evaluates positions so I can measure mistakes and winning chances.</p>
            <p>- The rest of the dashboard builds on this processed data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
