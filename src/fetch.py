"""Lichess game download (spec section 1, Source A).

Downloads the full PGN export for a user and writes it, streamed, to
data/raw/lichess/games_{date}.pgn. No parsing happens here.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime


LICHESS_GAMES_URL = "https://lichess.org/api/games/user/{username}"


@dataclass
class FetchConfig:
    """Everything the fetch needs, assembled from .env + config.toml."""

    username: str
    perf_types: list[str]
    token: str | None
    # TODO: any other config.toml fields you decide you need (rated-only? since?)


def load_config(config_path: Path = Path("config.toml")) -> FetchConfig:
    """Read username/perfTypes from config.toml and LICHESS_TOKEN from .env."""


    load_dotenv()
    token = os.getenv("LICHESS_TOKEN")
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    
    config = FetchConfig(username=data["username"], perf_types=data["perf_types"], token=token)

    return config
    

def build_request(config: FetchConfig) -> tuple[str, dict, dict]:
    """Build (url, params, headers) for the Lichess export endpoint."""

    url = LICHESS_GAMES_URL.format(username=config.username)
    params = {
    "clocks": "true",
    "evals": "true",
    "opening": "true",
    "rated": "true",
    "perfType": ",".join(config.perf_types),
    # TODO: remove before real fetch
    "max": "2",
    }

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
                print(f"~{n} games", end="\r")
    return n


def main() -> None:
    """Wire it together: load config, build request, stream to today's file."""
    
    today = datetime.now().strftime("%Y-%m-%d")
    config = load_config()
    url,params,headers = build_request(config)
    out_path = Path("data/raw/lichess") / f"games_{today}.pgn"
    n = stream_games_to_file(url,params,headers,out_path)

    print(f"number of games: {n}")

if __name__ == "__main__":
    main()
