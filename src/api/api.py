import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

load_dotenv(override=True)


@dataclass
class SpotifyData:
    """Container for Spotify API data: tracks, artists, and albums.

    Attributes:
        tracks: Mapping from track ID to track data.
        artists: Mapping from artist ID to artist data.
        albums: Mapping from album ID to album data.
    """

    tracks: dict[str, dict[str, Any]]
    artists: dict[str, dict[str, Any]]
    albums: dict[str, dict[str, Any]]


class SpotifyDataCollector:
    """Collects and caches Spotify tracks, artists, and albums data."""

    _TRACK_BATCH_SIZE: int = 50
    _ALBUM_BATCH_SIZE: int = 20
    _ARTIST_BATCH_SIZE: int = 20
    _RATE_LIMIT_DELAY: float = 0.2

    def __init__(
        self,
        client_id: str | None = os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret: str | None = os.getenv("SPOTIFY_CLIENT_SECRET"),
    ) -> None:
        """Initialize the collector with Spotify API credentials and cache dirs.

        Args:
            client_id: Spotify API client ID.
            client_secret: Spotify API client secret.
        """
        self.spotify_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        )
        self.data_dir: Path = Path(os.getenv("DATA_DIR"))  # type: ignore
        self.cache_dir: Path = Path("data/api/cache")
        self._setup_cache_directories()

    def _setup_cache_directories(self) -> None:
        """Create cache subdirectories for tracks, artists, and albums."""
        for category in ("tracks", "artists", "albums"):
            (self.cache_dir / category).mkdir(parents=True, exist_ok=True)

    def _get_cached_ids(self, category: str) -> set[str]:
        """Retrieve all cached item IDs for a given category.

        Args:
            category: One of 'tracks', 'artists', 'albums'.

        Returns:
            A set of cached item IDs (filename stems).
        """
        cache_path = self.cache_dir / category
        return {file.stem for file in cache_path.glob("*.json")}

    def _load_cache(self, category: str, item_id: str) -> dict[str, Any]:
        """Load a single cached item from disk.

        Args:
            category: Cache category.
            item_id: ID of the item to load.

        Returns:
            The cached item data as a dictionary.
        """
        path = self.cache_dir / category / f"{item_id}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self, category: str, items: list[dict[str, Any]]) -> None:
        """Save multiple items to the cache.

        Args:
            category: Cache category.
            items: List of item data dictionaries with 'id' keys.
        """
        for item in items:
            item_id = item["id"]
            path = self.cache_dir / category / f"{item_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(item, f, indent=4, sort_keys=True)

    def _chunk_list(self, items: list[str], size: int) -> list[list[str]]:
        """Split a list into smaller lists of a given size.

        Args:
            items: The list to split.
            size: Maximum size of each chunk.

        Returns:
            A list of list chunks.
        """
        return [items[i : i + size] for i in range(0, len(items), size)]

    def fetch_tracks(self, track_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch track data by IDs, using cache when available.

        Args:
            track_ids: List of Spotify track IDs.

        Returns:
            List of track data dictionaries.
        """
        cached = self._get_cached_ids("tracks")
        uncached = [tid for tid in track_ids if tid not in cached]

        # Load cached tracks
        tracks: list[dict[str, Any]] = [
            self._load_cache("tracks", tid) for tid in track_ids if tid in cached
        ]

        # Fetch remaining tracks in batches
        for batch in tqdm(
            self._chunk_list(uncached, self._TRACK_BATCH_SIZE),
            desc="Fetching track batches",
        ):
            response = self.spotify_client.tracks(batch)["tracks"]
            self._save_cache("tracks", response)
            tracks.extend(response)
            time.sleep(self._RATE_LIMIT_DELAY)

        return tracks

    def fetch_artists(self, artist_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch artist data by IDs, using cache when available.

        Args:
            artist_ids: List of Spotify artist IDs.

        Returns:
            List of artist data dictionaries.
        """
        cached = self._get_cached_ids("artists")
        uncached = [aid for aid in artist_ids if aid not in cached]

        artists: list[dict[str, Any]] = [
            self._load_cache("artists", aid) for aid in artist_ids if aid in cached
        ]

        for batch in tqdm(
            self._chunk_list(uncached, self._ARTIST_BATCH_SIZE),
            desc="Fetching artist batches",
        ):
            response = self.spotify_client.artists(batch)["artists"]
            self._save_cache("artists", response)
            artists.extend(response)
            time.sleep(self._RATE_LIMIT_DELAY)

        return artists

    def fetch_albums(self, album_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch album data by IDs, using cache when available.

        Args:
            album_ids: List of Spotify album IDs.

        Returns:
            List of album data dictionaries.
        """
        cached = self._get_cached_ids("albums")
        uncached = [aid for aid in album_ids if aid not in cached]

        albums: list[dict[str, Any]] = [
            self._load_cache("albums", aid) for aid in album_ids if aid in cached
        ]

        for batch in tqdm(
            self._chunk_list(uncached, self._ALBUM_BATCH_SIZE),
            desc="Fetching album batches",
        ):
            response = self.spotify_client.albums(batch)["albums"]
            self._save_cache("albums", response)
            albums.extend(response)
            time.sleep(self._RATE_LIMIT_DELAY)

        return albums


def _list_to_dict(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Convert a list of dicts into a dict keyed by a specified field.

    Args:
        items: List of dictionaries.
        key: The key to index dictionaries by.

    Returns:
        A dict mapping item[key] to the item dict.
    """
    return {item[key]: item for item in items}


def load_api_data() -> SpotifyData:
    """Load all Spotify data using ID files and the data collector.

    Reads track, artist, and album ID files, fetches their data, and returns
    a SpotifyData object containing dicts of the fetched data.
    """
    collector = SpotifyDataCollector()
    # Read ID lists from files
    track_file = collector.data_dir / "track_ids.txt"
    artist_file = collector.data_dir / "artist_ids.txt"
    album_file = collector.data_dir / "album_ids.txt"

    with open(track_file, encoding="utf-8") as f:
        track_ids = [line.strip() for line in f]
    with open(artist_file, encoding="utf-8") as f:
        artist_ids = [line.strip() for line in f]
    with open(album_file, encoding="utf-8") as f:
        album_ids = [line.strip() for line in f]

    tracks = collector.fetch_tracks(track_ids)
    artists = collector.fetch_artists(artist_ids)
    albums = collector.fetch_albums(album_ids)

    return SpotifyData(
        tracks=_list_to_dict(tracks, "id"),
        artists=_list_to_dict(artists, "id"),
        albums=_list_to_dict(albums, "id"),
    )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fetch Spotify data and cache it")
    parser.add_argument("--save", action="store_true", help="Save fetched data to CSV files")
    args = parser.parse_args()

    # Initialize collector
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    collector = SpotifyDataCollector(client_id, client_secret)

    # Load ID lists
    with open(collector.data_dir / "track_ids.txt", encoding="utf-8") as f:
        track_ids = [line.strip() for line in f]
    with open(collector.data_dir / "artist_ids.txt", encoding="utf-8") as f:
        artist_ids = [line.strip() for line in f]
    with open(collector.data_dir / "album_ids.txt", encoding="utf-8") as f:
        album_ids = [line.strip() for line in f]

    # Fetch data
    print("Fetching tracks...")
    tracks = collector.fetch_tracks(track_ids)
    print("Fetching artists...")
    artists = collector.fetch_artists(artist_ids)
    print("Fetching albums...")
    albums = collector.fetch_albums(album_ids)

    # Save data to CSV if requested
    if args.save:
        tracks_df = pd.DataFrame(tracks)
        tracks_df.to_csv(collector.data_dir / "tracks.csv", index=False)

        artists_df = pd.DataFrame(artists)
        artists_df.to_csv(collector.data_dir / "artists.csv", index=False)

        albums_df = pd.DataFrame(albums)
        albums_df.to_csv(collector.data_dir / "albums.csv", index=False)
