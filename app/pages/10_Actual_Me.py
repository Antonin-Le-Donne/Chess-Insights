from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import load_ui
from src.metrics import m2_time_to_blunder, m3_greed_conversion
from src.scope import (
    filter_current_regime,
    m1_peak_by_move,
    per_game_quality,
    performance_rating_estimate,
    quality_of_play_trajectory,
    rating_trajectory,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    games = pd.read_parquet("data/processed/games.parquet")
    moves = pd.read_parquet("data/processed/moves.parquet")

    with open("config.toml", "rb") as f:
        config_data = tomllib.load(f)

    floors = config_data["current_regime"]

    return games, moves, floors


PLOTLY_LAYOUT = go.Layout(
    font=dict(
        family="Inter",
        size=14,
        color="#DCE3E9",
    ),
    hovermode="x unified",
    template="plotly_dark",
    margin=dict(
        l=30,
        r=30,
        t=55,
        b=25,
    ),
)


PERF_COLORS = {
    "bullet": "#61B8FF",
    "blitz": "#B8E35A",
    "rapid": "#F4BE55",
    "classical": "#A78BFA",
}


st.set_page_config(
    page_title="Chess Insights",
    layout="wide",
)

load_ui()


def main() -> None:
    st.title("Actual me — scoping out skill drift")

    st.markdown(
        """
        **Question:** how much do the findings above actually describe *current* me,
        versus an average blended across years of improvement?

        My rating climbed a huge amount over this dataset's history. This page shows
        that drift directly, then compares the main findings computed on the full
        history against the same findings computed only on games from my current,
        roughly-stable rating band.
        """
    )

    games, moves, floors = load_data()

    games_current, moves_current = filter_current_regime(
        games,
        moves,
        floors,
    )

    trajectory = rating_trajectory(games).copy()

    st.markdown("### Rating trajectory")

    tab1, tab2 = st.tabs([
        "Lichess",
        "Chess.com",
    ])

    platform_tabs = {
        "lichess": tab1,
        "chesscom": tab2,
    }

    latest_ratings: dict[str, float] = {}

    for platform, tab in platform_tabs.items():
        with tab:
            platform_data = trajectory[
                trajectory["platform"] == platform
            ].copy()

            platform_data = platform_data.sort_values(
                "date_utc"
            )

            platform_floors = {
                key.split("_", 1)[1]: value
                for key, value in floors.items()
                if key.startswith(f"{platform}_")
            }

            platform_perf_types: list[str] = (
                platform_data["perf_type"]
                .dropna()
                .unique()
                .tolist()
            )

            cols = st.columns(
                len(platform_perf_types)
            )

            for col, perf_type in zip(
                cols,
                platform_perf_types,
            ):
                perf_data = platform_data[
                    platform_data["perf_type"] == perf_type
                ].sort_values("date_utc")

                latest_rating = float(
                    perf_data.iloc[-1]["mean_rating"]
                )

                latest_ratings[
                    f"{platform}_{perf_type}"
                ] = latest_rating

                floor = platform_floors.get(
                    perf_type
                )

                if floor is not None:
                    delta = latest_rating - floor

                    col.metric(
                        perf_type.title(),
                        f"{latest_rating:.0f}",
                        f"{delta:+.0f} vs floor",
                    )

                else:
                    col.metric(
                        perf_type.title(),
                        f"{latest_rating:.0f}",
                        "No current-regime floor",
                    )

            fig = px.line(
                platform_data,
                x="date_utc",
                y="mean_rating",
                color="perf_type",
                custom_data=["n"],
                color_discrete_map=PERF_COLORS,
                labels={
                    "date_utc": "Month",
                    "mean_rating": "Rating",
                    "perf_type": "Time control",
                },
            )

            trace_colors = {}

            for trace in fig.data:
                trace_colors[trace.name] = trace.line.color

                trace.update(
                    mode="lines",
                    line=dict(
                        width=3,
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "%{x|%B %Y}<br>"
                        "Rating: %{y:.0f}<br>"
                        "Games: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                )

            for perf_type in platform_perf_types:
                perf_data = platform_data[
                    platform_data["perf_type"] == perf_type
                ].sort_values("date_utc")

                latest = perf_data.iloc[-1]

                fig.add_trace(
                    go.Scatter(
                        x=[latest["date_utc"]],
                        y=[latest["mean_rating"]],
                        mode="markers",
                        marker=dict(
                            size=9,
                            color=trace_colors.get(
                                perf_type,
                                PERF_COLORS.get(perf_type),
                            ),
                            line=dict(
                                width=2,
                                color="#0B0F12",
                            ),
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            for perf_type, floor in platform_floors.items():
                if perf_type not in trace_colors:
                    continue

                fig.add_hline(
                    y=floor,
                    line_dash="dot",
                    line_width=1.5,
                    line_color=trace_colors[perf_type],
                    opacity=0.35,
                )

            fig.update_layout(
                PLOTLY_LAYOUT.to_plotly_json()
            )

            fig.update_layout(
                height=440,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    title=None,
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    title=None,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                ),
                yaxis=dict(
                    title="Mean rating",
                    gridcolor="rgba(255,255,255,0.07)",
                    zeroline=False,
                    showline=False,
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hoverlabel=dict(
                    bgcolor="#12181E",
                    bordercolor="rgba(255,255,255,0.10)",
                    font=dict(
                        family="Inter",
                        size=13,
                    ),
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    st.markdown("### Quality of play (platform-agnostic)")

    st.markdown(
        """
        Lichess and chess.com ratings aren't on the same scale (see
        `chess_insights_spec.md`), so there's no single number above that
        says "how good am I" across both platforms at once. Win% given up
        per move doesn't care which platform a game was on -- lower is
        better, and it's the same score everywhere.
        """
    )

    quality_full = quality_of_play_trajectory(
        moves,
        games,
    )

    quality_current = quality_of_play_trajectory(
        moves_current,
        games_current,
    )

    fig_quality = px.line(
        quality_full,
        x="date_utc",
        y="mean_win_pct_lost",
        custom_data=["n"],
        labels={
            "date_utc": "Month",
            "mean_win_pct_lost": "Win% lost per move",
        },
    )

    for trace in fig_quality.data:
        trace.update(
            mode="lines",
            line=dict(
                width=3,
            ),
            hovertemplate=(
                "%{x|%B %Y}<br>"
                "Win% lost: %{y:.2f}%<br>"
                "Moves: %{customdata[0]}"
                "<extra></extra>"
            ),
        )

    if not quality_full.empty:
        latest_quality = quality_full.iloc[-1]

        fig_quality.add_trace(
            go.Scatter(
                x=[latest_quality["date_utc"]],
                y=[latest_quality["mean_win_pct_lost"]],
                mode="markers",
                marker=dict(
                    size=9,
                    line=dict(
                        width=2,
                        color="#0B0F12",
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig_quality.update_layout(
        PLOTLY_LAYOUT.to_plotly_json()
    )

    fig_quality.update_layout(
        height=420,
        showlegend=False,
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            title="Win% lost per move",
            tickformat=".1f",
            ticksuffix="%",
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False,
            showline=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#12181E",
            bordercolor="rgba(255,255,255,0.10)",
            font=dict(
                family="Inter",
                size=13,
            ),
        ),
    )

    st.plotly_chart(
        fig_quality,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    full_n = quality_full["n"].sum()
    current_n = quality_current["n"].sum()

    if full_n > 0:
        quality_full_mean = (
            quality_full["mean_win_pct_lost"] * quality_full["n"]
        ).sum() / full_n
    else:
        quality_full_mean = float("nan")

    if current_n > 0:
        quality_current_mean = (
            quality_current["mean_win_pct_lost"] * quality_current["n"]
        ).sum() / current_n
    else:
        quality_current_mean = float("nan")

    col1, col2 = st.columns(2)

    col1.metric(
        "Full history — win% lost per move",
        f"{quality_full_mean:.2f}%",
    )

    col2.metric(
        "Current regime — win% lost per move",
        f"{quality_current_mean:.2f}%",
    )

    st.markdown("### Performance rating estimate")

    st.markdown(
        """
        Fits actual rating against quality of play, per (platform, perf_type),
        then checks whether current-regime quality implies a higher or lower
        rating than what's actually being played right now. A gap here points
        to something rating-affecting outside raw move quality rather than
        general "untapped potential".

        Fit quality is shown alongside every prediction. A weak R² means the
        estimate should be interpreted cautiously.
        """
    )

    games_quality_full = per_game_quality(
        moves,
        games
    )

    games_quality_current = per_game_quality(
        moves_current,
        games_current
    )


    st.html(
    """
    <style>
    .performance-group {
        margin-top: 1.8rem;
        margin-bottom: 1.3rem;
    }

    .performance-group-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.8rem;
    }

    .performance-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 0.8rem;
    }

    .performance-card {
        position: relative;
        overflow: hidden;
        padding: 1.25rem 1.35rem;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background: linear-gradient(
            145deg,
            rgba(24, 31, 38, 0.96),
            rgba(14, 19, 24, 0.96)
        );
        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.16),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }

    .performance-card.current {
        border-color: rgba(97, 184, 255, 0.28);
        background: linear-gradient(
            145deg,
            rgba(22, 35, 45, 0.98),
            rgba(13, 20, 26, 0.98)
        );
    }

    .performance-card::before {
        content: "";
        position: absolute;
        width: 130px;
        height: 130px;
        right: -55px;
        top: -60px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.025);
        pointer-events: none;
    }

    .performance-card.current::before {
        background: rgba(97, 184, 255, 0.055);
    }

    .performance-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }

    .performance-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #94a3b8;
    }

    .performance-card.current .performance-card-title {
        color: #8acbff;
    }

    .performance-badge {
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        color: #94a3b8;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .performance-card.current .performance-badge {
        color: #9dd3ff;
        background: rgba(97, 184, 255, 0.08);
        border-color: rgba(97, 184, 255, 0.15);
    }

    .performance-main {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1.15rem;
        margin-bottom: 1.15rem;
    }

    .performance-stat-label {
        font-size: 0.74rem;
        color: #7f8b97;
        margin-bottom: 0.25rem;
    }

    .performance-stat-value {
        font-size: 1.7rem;
        line-height: 1.1;
        font-weight: 750;
        color: #f8fafc;
        letter-spacing: -0.03em;
    }

    .performance-footer {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        padding-top: 0.95rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .performance-small-label {
        font-size: 0.69rem;
        color: #687580;
        margin-bottom: 0.18rem;
    }

    .performance-small-value {
        font-size: 0.95rem;
        font-weight: 650;
        color: #cbd5e1;
    }

    .performance-gap-positive {
        color: #7dd3a7;
    }

    .performance-gap-negative {
        color: #f59b9b;
    }

    .performance-gap-neutral {
        color: #cbd5e1;
    }

    .performance-context {
        margin-top: 0.45rem;
        color: #7f8b97;
        font-size: 0.78rem;
    }

    .performance-warning {
        margin-top: 0.7rem;
        margin-bottom: 1.2rem;
        padding: 0.75rem 0.95rem;
        border-radius: 11px;
        border: 1px solid rgba(245, 190, 85, 0.18);
        background: rgba(245, 190, 85, 0.055);
        color: #c9b98c;
        font-size: 0.79rem;
        line-height: 1.45;
    }

    .performance-ok {
        margin-top: 0.7rem;
        margin-bottom: 1.2rem;
        padding: 0.75rem 0.95rem;
        border-radius: 11px;
        border: 1px solid rgba(125, 211, 167, 0.16);
        background: rgba(125, 211, 167, 0.045);
        color: #9bc9af;
        font-size: 0.79rem;
        line-height: 1.45;
    }

    @media (max-width: 850px) {
        .performance-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """
    )


    for group_key in floors:
        platform, perf_type = group_key.split("_", 1)

        current_group = games_quality_current[
            (games_quality_current["platform"] == platform)
            & (games_quality_current["perf_type"] == perf_type)
        ]

        if current_group.empty:
            continue

        current_quality = current_group[
            "quality_score"
        ].mean()

        estimate_full = performance_rating_estimate(
            games_quality_full,
            platform,
            perf_type,
            current_quality,
        )

        estimate_current = performance_rating_estimate(
            games_quality_current,
            platform,
            perf_type,
            current_quality,
        )

        predicted_full = estimate_full["predicted_rating"]
        predicted_current = estimate_current["predicted_rating"]

        r2_full = estimate_full["r2"]
        r2_current = estimate_current["r2"]

        n_full = estimate_full["n_games"]
        n_current = estimate_current["n_games"]

        actual_rating = latest_ratings.get(
            group_key,
            float("nan"),
        )

        if pd.isna(actual_rating):
            actual_rating_text = "N/A"
        else:
            actual_rating_text = f"{actual_rating:.0f}"

        if pd.isna(predicted_full):
            predicted_full_text = "N/A"
            gap_full_text = "N/A"
            gap_full_class = "performance-gap-neutral"
        else:
            predicted_full_text = f"{predicted_full:.0f}"

            if pd.isna(actual_rating):
                gap_full_text = "N/A"
                gap_full_class = "performance-gap-neutral"
            else:
                gap_full = predicted_full - actual_rating
                gap_full_text = f"{gap_full:+.0f}"

                if gap_full > 0:
                    gap_full_class = "performance-gap-positive"
                elif gap_full < 0:
                    gap_full_class = "performance-gap-negative"
                else:
                    gap_full_class = "performance-gap-neutral"

        if pd.isna(predicted_current):
            predicted_current_text = "N/A"
            gap_current_text = "N/A"
            gap_current_class = "performance-gap-neutral"
        else:
            predicted_current_text = f"{predicted_current:.0f}"

            if pd.isna(actual_rating):
                gap_current_text = "N/A"
                gap_current_class = "performance-gap-neutral"
            else:
                gap_current = predicted_current - actual_rating
                gap_current_text = f"{gap_current:+.0f}"

                if gap_current > 0:
                    gap_current_class = "performance-gap-positive"
                elif gap_current < 0:
                    gap_current_class = "performance-gap-negative"
                else:
                    gap_current_class = "performance-gap-neutral"

        if pd.isna(r2_full):
            r2_full_text = "N/A"
        else:
            r2_full_text = f"{r2_full:.3f}"

        if pd.isna(r2_current):
            r2_current_text = "N/A"
        else:
            r2_current_text = f"{r2_current:.3f}"

        if platform == "chesscom":
            platform_label = "Chess.com"
        else:
            platform_label = platform.title()

        html = f"""
        <div class="performance-group">
            <div class="performance-group-title">
                {platform_label} — {perf_type.title()}
            </div>

            <div class="performance-grid">
                <div class="performance-card">
                    <div class="performance-card-header">
                        <div class="performance-card-title">
                            Full history model
                        </div>

                        <div class="performance-badge">
                            {n_full:,} games
                        </div>
                    </div>

                    <div class="performance-main">
                        <div class="performance-stat">
                            <div class="performance-stat-label">
                                Predicted rating
                            </div>

                            <div class="performance-stat-value">
                                {predicted_full_text}
                            </div>
                        </div>

                        <div class="performance-stat">
                            <div class="performance-stat-label">
                                Actual rating
                            </div>

                            <div class="performance-stat-value">
                                {actual_rating_text}
                            </div>
                        </div>
                    </div>

                    <div class="performance-footer">
                        <div>
                            <div class="performance-small-label">
                                Prediction gap
                            </div>

                            <div class="performance-small-value {gap_full_class}">
                                {gap_full_text}
                            </div>
                        </div>

                        <div>
                            <div class="performance-small-label">
                                Regression R²
                            </div>

                            <div class="performance-small-value">
                                {r2_full_text}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="performance-card current">
                    <div class="performance-card-header">
                        <div class="performance-card-title">
                            Current regime model
                        </div>

                        <div class="performance-badge">
                            {n_current:,} games
                        </div>
                    </div>

                    <div class="performance-main">
                        <div class="performance-stat">
                            <div class="performance-stat-label">
                                Predicted rating
                            </div>

                            <div class="performance-stat-value">
                                {predicted_current_text}
                            </div>
                        </div>

                        <div class="performance-stat">
                            <div class="performance-stat-label">
                                Actual rating
                            </div>

                            <div class="performance-stat-value">
                                {actual_rating_text}
                            </div>
                        </div>
                    </div>

                    <div class="performance-footer">
                        <div>
                            <div class="performance-small-label">
                                Prediction gap
                            </div>

                            <div class="performance-small-value {gap_current_class}">
                                {gap_current_text}
                            </div>
                        </div>

                        <div>
                            <div class="performance-small-label">
                                Regression R²
                            </div>

                            <div class="performance-small-value">
                                {r2_current_text}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="performance-context">
                Current quality: {current_quality:.2f}%
                win points lost per move
            </div>
        </div>
        """

        st.html(html)

        weak_full = (
            not pd.isna(r2_full)
            and r2_full < 0.10
        )

        weak_current = (
            not pd.isna(r2_current)
            and r2_current < 0.10
        )

        if weak_full or weak_current:
            st.html(
                """
                <div class="performance-warning">
                    ⚠ <strong>Weak regression fit.</strong>
                    The predicted ratings are useful for comparison,
                    but the relationship between move quality and rating
                    is too weak to treat these values as reliable rating estimates.
                </div>
                """
            )
        else:
            st.html(
                """
                <div class="performance-ok">
                    The regression fit is above the weak-fit threshold
                    for both models.
                </div>
                """
            )

    st.markdown(
        f"**Current-regime games kept:** "
        f"{len(games_current)} / {len(games)} total"
    )

    st.markdown(
        "### Full history vs current regime"
    )

    m1_full = m1_peak_by_move(
        moves,
        games,
    ).reset_index()

    m1_current = m1_peak_by_move(
        moves_current,
        games_current,
    ).reset_index()

    m2_full = m2_time_to_blunder(
        moves,
        games,
    )

    m2_current = m2_time_to_blunder(
        moves_current,
        games_current,
    )

    m3_full = m3_greed_conversion(
        moves,
        games,
    )

    m3_current = m3_greed_conversion(
        moves_current,
        games_current,
    )

    comparison_perf_types: list[str] = [
        perf_type
        for perf_type in m2_current.keys()
        if perf_type in m2_full
        and perf_type in m1_full["perf_group"].unique()
        and perf_type in m1_current["perf_group"].unique()
    ]

    perf_type_choice = st.selectbox(
        "Time control",
        comparison_perf_types,
        key="current_regime_time_control",
    )

    m1_full_filtered = m1_full[
        m1_full["perf_group"] == perf_type_choice
    ]

    m1_current_filtered = m1_current[
        m1_current["perf_group"] == perf_type_choice
    ]

    if perf_type_choice == "bullet":
        m3_full_filtered = m3_full[
            m3_full["perf_type"] == "bullet"
        ]

        m3_current_filtered = m3_current[
            m3_current["perf_type"] == "bullet"
        ]

    else:
        m3_full_filtered = m3_full[
            m3_full["perf_type"] != "bullet"
        ]

        m3_current_filtered = m3_current[
            m3_current["perf_type"] != "bullet"
        ]

    st.markdown(
        "#### M1 — Fatigue curve"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Full history — peak blunder rate",
        f"{m1_full_filtered['blunder_rate'].max():.1%}",
    )

    col2.metric(
        "Current regime — peak blunder rate",
        f"{m1_current_filtered['blunder_rate'].max():.1%}",
    )

    st.markdown(
        "#### M2 — Time to blunder"
    )

    stats_full = m2_full[
        perf_type_choice
    ]

    stats_current = m2_current[
        perf_type_choice
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "Full history — blunder risk multiplier",
        f"{stats_full['ratio']:.2f}×",
    )

    col2.metric(
        "Current regime — blunder risk multiplier",
        f"{stats_current['ratio']:.2f}×",
    )

    st.markdown(
        "#### M3 — Advantage conversion"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Full history — conversion success rate",
        f"{m3_full_filtered['did_convert'].mean():.1%}",
    )

    col2.metric(
        "Current regime — conversion success rate",
        f"{m3_current_filtered['did_convert'].mean():.1%}",
    )

    st.markdown(
        "### Why this page exists"
    )

    st.markdown(
        """
        Averaging "beginner me" and "current me" into one blunder rate describes
        neither. Rating floors per (platform, perf_type) are set in `config.toml`'s
        `[current_regime]` section, not hardcoded here — see `PROJECT_TRACKER.md`'s
        decisions log for how each floor was picked, and which groups (rapid, chess.com
        blitz) were excluded outright for not having a stable regime yet.
        """
    )


if __name__ == "__main__":
    main()