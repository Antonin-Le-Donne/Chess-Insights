"""Overview (spec section 6, "Overview", revised 2026-11-15).

Question: what do my games say about me, and am I improving?
(1) headline sentence, (2) one card per metric read from
data/reports/summary.json (written by `python -m src.metrics`),
(3) monthly conversion trend, (4) "About the data" expander.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.filters import sidebar_filtered
from app.risk_bands import band_rates, find_latest_predictions
from app.ui import load_ui
from src.metrics import SUMMARY_PATH, m3_greed_conversion

MONTHS_IN_CHART = 24
SMALL_SAMPLE = 30

GREEN = "#7BE0A3"
RED = "#FF6B6B"
GRAY = "#B7C2CC"
BLUE = "#7CC4FF"
MUTED = "#8F9893"
ACCENT = "#C7F36B"

BADGES = {
    "strength": ("Personal strength", GREEN),
    "weakness": ("Personal weakness", RED),
    "shared": ("Shared at my level", GRAY),
    "no_ci": ("Compared · no interval yet", BLUE),
    "not_compared": ("Not compared with opponents", MUTED),
    "model": ("Model", ACCENT),
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")
    return games, moves


@st.cache_data
def load_summary(path: str, modified: float) -> dict | None:
    """summary.json written by src.metrics; `modified` busts the cache."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_risk_bands(path: str) -> pd.DataFrame:
    return band_rates(pd.read_parquet(path, columns=["predicted_risk", "is_blunder"]))


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


def verdict(lower: float, upper: float, good_when_negative: bool) -> str:
    """Badge from a paired-difference CI (me − opponents), spec reading rule."""
    if lower > 0:
        return "weakness" if good_when_negative else "strength"
    if upper < 0:
        return "strength" if good_when_negative else "weakness"
    return "shared"


# ---------- sections ----------

def render_headline(games: pd.DataFrame, m3: pd.DataFrame) -> None:
    first, last = games["date_utc"].min(), games["date_utc"].max()
    sentence = f"**{len(games):,} games** from {first:%B %Y} to {last:%B %Y}."
    if not m3.empty:
        table = monthly_conversion(m3)
        latest = table.index.max()
        row = table.loc[latest]
        overall = m3["did_convert"].mean()
        caveat = " (small sample)" if row["n_games"] < SMALL_SAMPLE else ""
        sentence += (
            f" In **{latest.strftime('%B %Y')}**, I reached a winning position (80%+) in "
            f"{int(row['n_games'])} games and converted **{row['conversion']:.0%}** of "
            f"them{caveat}, against **{overall:.0%}** over the whole selection."
        )
    st.markdown(f"#### {sentence}")


def finding_cards(summary: dict, risk: pd.DataFrame | None) -> list[dict]:
    """One dict per card: title, sentence, badge, page."""
    cards = []
    if "m3" in summary:
        s = summary["m3"]
        cards.append({
            "title": "Converting winning positions",
            "text": f"Once I reach 80%, I win **{s['conv_me']:.0%}** of games; my opponents "
                    f"**{s['conv_opponent']:.0%}** from the same point ({100 * s['diff']:+.1f} pts).",
            "badge": verdict(s["diff_lo"], s["diff_hi"], good_when_negative=False),
            "page": "pages/4_Advantage_Conversion.py",
        })
    if "m6" in summary:
        s = summary["m6"]
        cards.append({
            "title": "Saving lost positions",
            "text": f"From a lost position I score **{100 * s['me_vs_expected']:+.0f} pts** above "
                    f"the engine's expectation; my opponents **{100 * s['opponent_vs_expected']:+.0f} pts**.",
            "badge": "no_ci",
            "page": "pages/12_Swindle_Profile.py",
        })
    if "m4" in summary:
        s = summary["m4"]
        cards.append({
            "title": "Time trouble",
            "text": f"My clock drops below 10% in **{s['prop_me']:.0%}** of my games, my opponents' "
                    f"in **{s['prop_opponent']:.0%}**. It starts around move "
                    f"{s['median_onset_me']:.0f}, after the blunder peak.",
            "badge": verdict(s["diff_lo"], s["diff_hi"], good_when_negative=True),
            "page": "pages/5_Time_Management.py",
        })
    if "m2" in summary:
        s = summary["m2"]
        cards.append({
            "title": "Long thinks and blunders",
            "text": f"After a very long think, I blunder **{s['ratio_me']:.1f}×** more often in the "
                    f"next 3 moves; my opponents **{s['ratio_opponent']:.1f}×**.",
            "badge": verdict(s["diff_lo"], s["diff_hi"], good_when_negative=True),
            "page": "pages/3_Time_to_Blunder.py",
        })
    if "m1" in summary:
        s = summary["m1"]
        cards.append({
            "title": "When I blunder",
            "text": f"My blunders peak around moves **{s['peak_start']}–{s['peak_end']}** "
                    f"({s['peak_rate']:.1%} of my moves there), not at the end of the game.",
            "badge": "not_compared",
            "page": "pages/2_Fatigue_Curve.py",
        })
    if "m5" in summary:
        s = summary["m5"]
        cards.append({
            "title": "Weakest opening",
            "text": f"**{s['family']} as {s['color'].capitalize()}**: {s['score']:.0%} score over "
                    f"{s['n_games']} games (lowest family with 100+ games).",
            "badge": "not_compared",
            "page": "pages/6_Opening_Surgery.py",
        })
    if "m7" in summary:
        s = summary["m7"]
        cards.append({
            "title": "Momentum and tilt",
            "text": f"After a win I score **{100 * s['after_win']:+.1f} pts** above my rating's "
                    f"expectation; after a loss **{100 * s['after_loss']:+.1f} pts**.",
            "badge": "not_compared",
            "page": "pages/13_Time_And_Session.py",
        })
    if risk is not None:
        cards.append({
            "title": "Blunder-risk model",
            "text": f"On moves it flags as high risk I blunder **{risk.loc['High risk', 'rate']:.0%}** "
                    f"of the time, against **{risk.loc['Low risk', 'rate']:.1%}** on low-risk moves.",
            "badge": "model",
            "page": "pages/14_Risk_Timeline.py",
        })
    return cards


def render_card(card: dict) -> None:
    label, color = BADGES[card["badge"]]
    with st.container(border=True):
        st.markdown(
            f"<span style='color:{color};font-size:0.8rem;font-weight:600;"
            f"letter-spacing:0.04em;text-transform:uppercase'>● {label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{card['title']}**")
        st.markdown(card["text"])
        try:
            st.page_link(card["page"], label="See the details →")
        except Exception:  # page links only resolve inside the multipage app
            pass


def render_findings() -> None:
    st.markdown("### What my games say about me")
    summary_file = Path(SUMMARY_PATH)
    if not summary_file.exists():
        st.info(
            "No findings summary yet. Run `python -m src.metrics` (a few minutes): "
            "it writes `data/reports/summary.json`, which these cards read."
        )
        return
    summary = load_summary(str(summary_file), summary_file.stat().st_mtime)
    predictions_path = find_latest_predictions()
    risk = load_risk_bands(str(predictions_path)) if predictions_path else None

    cards = finding_cards(summary, risk)
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, card in zip(columns, cards[start:start + 3]):
            with column:
                render_card(card)
    st.caption(
        f"Blitz and rapid, all {summary['data']['n_games']:,} games, from the metrics run of "
        f"{summary['generated_at'][:10]} (the sidebar filters do not apply to these cards). "
        "Badges follow each page's reading rule: a 95% interval of the difference with my "
        "opponents that excludes 0 is personal; one that contains 0 is shared at my level."
    )


def render_progress(m3: pd.DataFrame, games: pd.DataFrame) -> None:
    st.markdown("### Am I improving? — conversion from a winning position, month by month")
    if m3.empty:
        st.info("No game in the selection reaches an 80% win probability.")
        return

    table = monthly_conversion(m3).tail(MONTHS_IN_CHART)
    months = table.index.to_timestamp()
    overall = m3["did_convert"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=table["ci_high"], mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=months, y=table["ci_low"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(199, 243, 107, 0.15)",
        hoverinfo="skip", name="95% interval",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=table["conversion"], mode="lines+markers",
        line=dict(color=ACCENT, width=2), name="Conversion",
        customdata=table[["n_converted", "n_games"]].to_numpy(),
        hovertemplate="%{x|%B %Y}: %{y:.0%} (%{customdata[0]} of %{customdata[1]} games)<extra></extra>",
    ))
    fig.add_hline(
        y=overall, line_dash="dot", line_color=MUTED,
        annotation_text=f"whole selection {overall:.0%}", annotation_position="bottom right",
    )
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter", size=14),
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    notes = ["A wide band means few games that month: a change inside it is noise."]
    if games["perf_type"].eq("bullet").any() and games["perf_type"].ne("bullet").any():
        notes.append(
            "The selection mixes bullet with blitz/rapid; bullet converts less, so filter "
            "one time control in the sidebar for a like-for-like trend."
        )
    st.caption(" ".join(notes))

    with st.expander("Monthly table"):
        recent = monthly_conversion(m3).tail(12).sort_index(ascending=False)
        st.dataframe(
            pd.DataFrame(
                {
                    "Month": recent.index.strftime("%Y-%m"),
                    "Games reaching 80%": recent["n_games"].astype(int),
                    "Converted": recent["n_converted"].astype(int),
                    "Conversion": recent["conversion"].map("{:.1%}".format),
                    "95% CI": [
                        f"{low:.0%}–{high:.0%}"
                        for low, high in zip(recent["ci_low"], recent["ci_high"])
                    ],
                    "Median plies to finish": recent["median_plies"].round(0).astype(int),
                }
            ),
            hide_index=True,
            width="stretch",
        )


def render_about(games: pd.DataFrame, moves: pd.DataFrame) -> None:
    with st.expander("About the data"):
        coverage = 1 - moves["eval_cp"].isna().mean()
        st.markdown(
            f"- **Games:** {len(games):,} rated standard games, "
            f"{games['date_utc'].min():%Y-%m-%d} to {games['date_utc'].max():%Y-%m-%d}\n"
            f"- **Engine evaluation:** {coverage:.1%} of moves evaluated (local Stockfish, "
            "plus Lichess server analysis when available)\n"
            "- **Pipeline:** Lichess + chess.com APIs → parsing → Stockfish → features → metrics"
        )
        counts = (
            games.groupby(["platform", "perf_type"]).size().unstack(fill_value=0)
            .rename_axis(index="Platform", columns=None)
        )
        st.dataframe(counts, width="stretch")


st.set_page_config(page_title="Chess Insights", layout="wide")
load_ui()


def main() -> None:
    st.title("Overview")
    st.markdown("**Question:** what do my games say about me, and am I improving?")

    games, moves = sidebar_filtered(*load_data())
    m3 = m3_greed_conversion(moves, games)

    render_headline(games, m3)
    render_findings()
    render_progress(m3, games)
    render_about(games, moves)


if __name__ == "__main__":
    main()
