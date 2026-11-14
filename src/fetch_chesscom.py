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


def month_end(archive_url: str) -> datetime:
    """First instant (UTC) after the archive's month: games can no longer be added."""

    parts = archive_url.rstrip("/").split("/")
    year = int(parts[-2])
    month = int(parts[-1])

    if month == 12:
        return datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(year, month + 1, 1, tzinfo=timezone.utc)


def cache_is_complete(cache_path: Path, archive_url: str) -> bool:
    """True if the cached file was written after its month ended.

    A month is immutable only once it is over (spec: "never re-fetch a
    completed past month"). A file cached DURING its month is partial: the
    games played after that fetch are missing, so it must be fetched again.
    Checking "is it the current month?" is not enough -- a month cached on
    the 17th and checked the following month would be skipped forever.
    """
    if not cache_path.exists():
        return False
    written_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    return written_at >= month_end(archive_url)


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

        if cache_is_complete(cache_path, archive_url):
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