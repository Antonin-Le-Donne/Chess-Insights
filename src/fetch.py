"""Lichess game download (spec section 1, Source A).

Downloads the PGN export for a user and writes it, streamed, to
data/raw/lichess/games_{timestamp}.pgn. No parsing happens here.

Incremental by default (spec: "use `since` on next run"): only games started
after the most recent Lichess game already in games.parquet are requested.
Any overlap would be harmless anyway, because parse.py skips known game ids.
``--full`` downloads the whole history again.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import tomllib
from dotenv import load_dotenv

LICHESS_GAMES_URL = "https://lichess.org/api/games/user/{username}"
GAMES_PARQUET = Path("data/processed/games.parquet")

@dataclass
class FetchConfig:
    """Everything the fetch needs, assembled from .env + config.toml."""
    username: str
    chesscom_username: str
    perf_types: list[str]
    token: str | None


def load_config(config_path: Path = Path("config.toml")) -> FetchConfig:
    """Read username/perfTypes from config.toml and LICHESS_TOKEN from .env."""

    load_dotenv()
    token = os.getenv("LICHESS_TOKEN")
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    
    config = FetchConfig(username=data["username"],chesscom_username=data["chesscom_username"], perf_types=data["perf_types"], token=token)

    return config
    

def latest_lichess_game_ms(games_path: Path = GAMES_PARQUET) -> int | None:
    """First millisecond after the start of the newest parsed Lichess game.

    Derived from games.parquet rather than from a stored "last fetch"
    timestamp: a game that was still being played during the previous fetch
    started after the newest stored game, so it is still picked up.
    Stored start times have one-second resolution, so one second is added to
    exclude the newest game itself (one player cannot start two games in the
    same second). Returns None when there is nothing parsed yet (first run).
    """
    if not games_path.exists():
        return None
    games = pd.read_parquet(games_path, columns=["platform", "date_utc"])
    lichess_dates = games.loc[games["platform"] == "lichess", "date_utc"]
    if lichess_dates.empty:
        return None
    # date_utc is stored naive but is UTC (UTCDate/UTCTime headers).
    newest = pd.Timestamp(lichess_dates.max()).tz_localize("UTC")
    return int(newest.timestamp() * 1000) + 1000


def build_request(config: FetchConfig, since_ms: int | None = None) -> tuple[str, dict, dict]:
    """Build (url, params, headers) for the Lichess export endpoint."""

    url = LICHESS_GAMES_URL.format(username=config.username)
    params = {
    "clocks": "true",
    "evals": "true",
    "opening": "true",
    "rated": "true",
    "perfType": ",".join(config.perf_types),
    }
    if since_ms is not None:
        params["since"] = str(since_ms)

    headers = {"Accept": "application/x-chess-pgn"}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
 
    return url, params, headers


def stream_games_to_file(url: str, params: dict, headers: dict, out_path: Path) -> int:
    """Stream the PGN response to out_path, chunk by chunk."""
    
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, params=params, headers=headers, stream=True, timeout=30) as response:
        if response.status_code == 429:
            print("Rate limited by Lichess (429), wait 60 seconds before retrying")
            sys.exit(1)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                n += chunk.count(b"[Event ")
                print(f"~{n} games", end="\r", flush=True)
    return n


def main() -> None:
    """Wire it together: load config, build request, stream to a new file."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full",
        action="store_true",
        help="download the whole Lichess history instead of only new games",
    )
    args = parser.parse_args()

    since_ms = None if args.full else latest_lichess_game_ms()
    if since_ms is None:
        print("Fetching the full Lichess history.")
    else:
        since_text = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
        print(f"Fetching Lichess games started since {since_text:%Y-%m-%d %H:%M} UTC.")

    # Time in the name: two runs on the same day must not overwrite each other.
    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    config = load_config()
    url,params,headers = build_request(config, since_ms)
    out_path = Path("data/raw/lichess") / f"games_{stamp}.pgn"
    n = stream_games_to_file(url,params,headers,out_path)

    if n == 0:
        out_path.unlink(missing_ok=True)
        print("No new Lichess games.")
        return
    print(f"number of games: {n} (saved to {out_path})")

if __name__ == "__main__":
    main()
