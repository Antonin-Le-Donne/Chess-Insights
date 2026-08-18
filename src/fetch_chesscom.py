"""chess.com game download (spec section 1, Source B)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.fetch import load_config

CHESSCOM_ARCHIVES_URL = "https://api.chess.com/pub/player/{username}/games/archives"
USER_AGENT = "ChessInsights/1.0 (contact: ledonneantoni@gmail.com)"


def list_archives(username: str, headers: dict) -> list[str]:
    """GET the archives endpoint, return the list of monthly archive URLs."""
    url = CHESSCOM_ARCHIVES_URL.format(username=username)
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()["archives"]

def month_cache_path(archive_url: str) -> Path:
    """Derive data/raw/chesscom/{YYYY}_{MM}.json from an archive URL."""

    parts = archive_url.rstrip("/").split("/")
    year = parts[-2]
    month = parts[-1]

    return Path("data/raw/chesscom") / f"{year}_{month}.json"


def is_current_month(archive_url: str) -> bool:
    """True if this archive's YYYY/MM matches today's UTC year/month --
    the one exception to "never re-fetch a cached month".
    """

    parts = archive_url.rstrip("/").split("/")
    year = int(parts[-2])
    month = int(parts[-1])

    now = datetime.now(tz=timezone.utc)

    return year == now.year and month == now.month


def fetch_month(archive_url: str, headers: dict) -> dict:
    """GET one monthly archive, return the parsed JSON (list of games
    under the "games" key). requests.get(archive_url, headers=headers).
    """

    response = requests.get(archive_url, headers=headers)
    response.raise_for_status()

    return response.json()


def main() -> None:
    config = load_config()
    headers = {"User-Agent": USER_AGENT}

    archive_urls = list_archives(config.chesscom_username, headers)

    for archive_url in archive_urls:
        cache_path = month_cache_path(archive_url)

        if cache_path.exists() and not is_current_month(archive_url):
            print(f"Skipping {cache_path.name} -- already cached")
            continue

        print(f"Fetching {cache_path.name}...")

        data = fetch_month(archive_url, headers)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {cache_path.name}")

if __name__ == "__main__":
    main()