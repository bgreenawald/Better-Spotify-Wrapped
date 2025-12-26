import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.exceptions import SpotifyException
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
    _ARTIST_BATCH_SIZE: int = 50  # Increased from 20 to 50 (API max)
    _DEFAULT_RATE_LIMIT_DELAY: float = 0.5  # Increased to avoid rate limiting

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        rate_limit_delay: float | None = None,
        max_workers: int = 1,
    ) -> None:
        """Initialize the collector with Spotify API credentials and cache dirs.

        Args:
            client_id: Spotify API client ID.
            client_secret: Spotify API client secret.
            cache_dir: Optional cache directory override.
            rate_limit_delay: Delay between API requests in seconds (default: 0.5).
            max_workers: Maximum concurrent worker threads for API requests (default: 1).
                        Lower values are safer for rate limits, higher values are faster.
        """
        # Resolve credentials from args or env at runtime
        cid = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        csecret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        if not cid or not csecret:
            raise ValueError(
                "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET in the environment or pass them to SpotifyDataCollector()."
            )
        self.spotify_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=csecret)
        )
        data_dir_env = os.getenv("DATA_DIR")
        if not data_dir_env:
            raise ValueError(
                "DATA_DIR is not set. Create a .env with DATA_DIR=<path> "
                "(keys: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, DATA_DIR)."
            )
        self.data_dir = Path(data_dir_env)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Allow configurable cache directory via arg or env, defaulting to repo path
        cache_env = os.getenv("SPOTIFY_API_CACHE_DIR")
        self.cache_dir: Path = Path(cache_dir or cache_env or "data/api/cache")
        self._setup_cache_directories()

        # Configure rate limiting and concurrency
        self._rate_limit_delay = rate_limit_delay or float(
            os.getenv("SPOTIFY_API_RATE_DELAY", str(self._DEFAULT_RATE_LIMIT_DELAY))
        )
        self._max_workers = max_workers
        self._request_lock = threading.Lock()
        self._last_request_time = 0.0

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

    def _rate_limited_request(self) -> None:
        """Enforce rate limiting between API requests using a thread-safe lock."""
        with self._request_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
            self._last_request_time = time.time()

    def _fetch_with_retry(
        self, fetch_func: Any, batch: list[str], max_retries: int = 3
    ) -> list[dict[str, Any]]:
        """Execute an API fetch with retry logic and exponential backoff.

        Args:
            fetch_func: The Spotify API method to call.
            batch: The batch of IDs to fetch.
            max_retries: Maximum number of retry attempts.

        Returns:
            List of fetched items.

        Raises:
            SpotifyException: If all retry attempts are exhausted.
        """
        for attempt in range(max_retries):
            try:
                self._rate_limited_request()
                return fetch_func(batch)
            except SpotifyException as e:
                if e.http_status == 429:  # Rate limited
                    retry_after = int(e.headers.get("Retry-After", 2**attempt))
                    logging.warning(
                        f"Rate limited. Retrying after {retry_after}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_after)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logging.warning(f"Request failed: {e}. Retrying after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise
        raise SpotifyException(
            429, -1, f"Max retries ({max_retries}) exceeded for batch", headers={}
        )

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

        # Fetch remaining tracks in batches with concurrent processing
        batches = self._chunk_list(uncached, self._TRACK_BATCH_SIZE)
        if batches:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(
                        self._fetch_with_retry,
                        lambda _batch=batch: self.spotify_client.tracks(_batch)["tracks"],
                        batch,
                    ): batch
                    for batch in batches
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Fetching track batches"
                ):
                    try:
                        response = future.result()
                        self._save_cache("tracks", response)
                        tracks.extend(response)
                    except Exception as e:
                        logging.error(f"Failed to fetch track batch: {e}")

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

        # Fetch remaining artists in batches with concurrent processing
        batches = self._chunk_list(uncached, self._ARTIST_BATCH_SIZE)
        if batches:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(
                        self._fetch_with_retry,
                        lambda _batch=batch: self.spotify_client.artists(_batch)["artists"],
                        batch,
                    ): batch
                    for batch in batches
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Fetching artist batches"
                ):
                    try:
                        response = future.result()
                        self._save_cache("artists", response)
                        artists.extend(response)
                    except Exception as e:
                        logging.error(f"Failed to fetch artist batch: {e}")

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

        # Fetch remaining albums in batches with concurrent processing
        batches = self._chunk_list(uncached, self._ALBUM_BATCH_SIZE)
        if batches:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(
                        self._fetch_with_retry,
                        lambda _batch=batch: self.spotify_client.albums(_batch)["albums"],
                        batch,
                    ): batch
                    for batch in batches
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Fetching album batches"
                ):
                    try:
                        response = future.result()
                        self._save_cache("albums", response)
                        albums.extend(response)
                    except Exception as e:
                        logging.error(f"Failed to fetch album batch: {e}")

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
        track_ids = [s for line in f if (s := line.strip()) and not s.startswith("#")]
    with open(artist_file, encoding="utf-8") as f:
        artist_ids = [s for line in f if (s := line.strip()) and not s.startswith("#")]
    with open(album_file, encoding="utf-8") as f:
        album_ids = [s for line in f if (s := line.strip()) and not s.startswith("#")]

    tracks = collector.fetch_tracks(track_ids)
    artists = collector.fetch_artists(artist_ids)
    albums = collector.fetch_albums(album_ids)

    return SpotifyData(
        tracks=_list_to_dict(tracks, "id"),
        artists=_list_to_dict(artists, "id"),
        albums=_list_to_dict(albums, "id"),
    )


def populate_missing_track_isrcs(
    *,
    db_path: str | Path,
    client_id: str | None = None,
    client_secret: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> int:
    """Populate `dim_tracks.track_isrc` where NULL using Spotify metadata.

    Strategy:
    - Query `dim_tracks` for rows with NULL `track_isrc` and collect their Spotify IDs (`track_id`).
    - For those Spotify IDs, check the local cache (configurable dir). If not cached,
      fetch from the Spotify API (batched) and cache the JSON.
    - Extract `external_ids.isrc` and update `dim_tracks.track_isrc` via a bulk UPDATE.

    Args:
        db_path: Path to DuckDB database file.
        client_id: Spotify Client ID (falls back to env `SPOTIFY_CLIENT_ID`).
        client_secret: Spotify Client Secret (falls back to env `SPOTIFY_CLIENT_SECRET`).
        cache_dir: Optional cache directory override (env `SPOTIFY_API_CACHE_DIR` if not set).
        limit: Optional cap on number of tracks to process in this run (for testing).

    Returns:
        Count of rows updated in `dim_tracks`.
    """
    # 1) Discover targets from the DB
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT track_id
            FROM dim_tracks
            WHERE track_isrc IS NULL
            ORDER BY track_id
            """
        ).fetchall()
    finally:
        conn.close()

    spotify_ids: list[str] = [r[0] for r in rows if r and r[0]]
    if limit is not None:
        spotify_ids = spotify_ids[: max(0, int(limit))]
    if not spotify_ids:
        return 0

    # 2) Fetch metadata via cache-aware collector
    collector = SpotifyDataCollector(
        client_id=client_id, client_secret=client_secret, cache_dir=cache_dir
    )
    tracks = collector.fetch_tracks(spotify_ids)

    # 3) Build records for ISRC updates
    recs: list[tuple[str, str | None]] = []
    for t in tracks:
        tid = t.get("id")
        if not tid:
            continue
        isrc = (t.get("external_ids") or {}).get("isrc")
        if isrc and str(isrc).strip():
            recs.append((tid, str(isrc).strip()))

    if not recs:
        return 0

    # Deduplicate ISRCs to avoid UNIQUE constraint violations on dim_tracks.track_isrc
    df_isrc = pd.DataFrame(recs, columns=["track_id", "isrc"]).drop_duplicates(
        subset=["isrc"], keep="first"
    )

    # 4) Apply updates in a single transaction
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_isrc", df_isrc)

        # Pre-compute how many rows will be updated
        isrc_to_set = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_tracks d
            JOIN df_isrc u ON d.track_id = u.track_id
            WHERE d.track_isrc IS NULL AND u.isrc IS NOT NULL
            """
        ).fetchone()[0]

        # UPDATE with uniqueness check to avoid constraint violations
        conn.execute(
            """
            UPDATE dim_tracks AS d
            SET track_isrc = u.isrc
            FROM df_isrc AS u
            WHERE d.track_id = u.track_id
              AND d.track_isrc IS NULL
              AND u.isrc IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM dim_tracks d2
                WHERE d2.track_isrc = u.isrc AND d2.track_id <> d.track_id
              )
            """
        )

        conn.execute("COMMIT;")
        return int(isrc_to_set)
    except Exception as e:
        conn.execute("ROLLBACK;")
        logging.error("Error updating ISRCs: %s", e)
        raise
    finally:
        conn.close()


def populate_duration_and_explicit(
    *,
    db_path: str | Path,
    client_id: str | None = None,
    client_secret: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """Populate `dim_tracks.duration_ms` and `dim_tracks.explicit` when NULL.

    Fetches track metadata for tracks missing either value and updates them.

    Returns a tuple: (duration_rows_updated, explicit_rows_updated).
    """
    # 1) Discover targets from the DB
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT track_id
            FROM dim_tracks
            WHERE (duration_ms IS NULL OR explicit IS NULL)
            ORDER BY track_id
            """
        ).fetchall()
    finally:
        conn.close()

    spotify_ids: list[str] = [r[0] for r in rows if r and r[0]]
    if limit is not None:
        spotify_ids = spotify_ids[: max(0, int(limit))]
    if not spotify_ids:
        return (0, 0)

    # 2) Fetch metadata (uses cache when available)
    collector = SpotifyDataCollector(
        client_id=client_id, client_secret=client_secret, cache_dir=cache_dir
    )
    tracks = collector.fetch_tracks(spotify_ids)

    # 3) Build update mapping
    records: list[tuple[str, int | None, bool | None]] = []
    for t in tracks:
        tid = t.get("id")
        if not tid:
            continue
        dur = t.get("duration_ms")
        try:
            dur_int = int(dur) if dur is not None else None
        except Exception:
            dur_int = None
        explicit_val = t.get("explicit")
        explicit_bool = bool(explicit_val) if explicit_val is not None else None
        records.append((tid, dur_int, explicit_bool))

    if not records:
        return (0, 0)

    df_updates = pd.DataFrame(records, columns=["track_id", "duration_ms", "explicit"])
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_updates", df_updates)

        # Pre-compute how many rows will be newly populated
        duration_to_set = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_tracks d
            JOIN df_updates u ON d.track_id = u.track_id
            WHERE d.duration_ms IS NULL AND u.duration_ms IS NOT NULL
            """
        ).fetchone()[0]
        explicit_to_set = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_tracks d
            JOIN df_updates u ON d.track_id = u.track_id
            WHERE d.explicit IS NULL AND u.explicit IS NOT NULL
            """
        ).fetchone()[0]

        conn.execute(
            """
            UPDATE dim_tracks AS d
            SET
              duration_ms = COALESCE(d.duration_ms, u.duration_ms),
              explicit     = COALESCE(d.explicit, u.explicit)
            FROM df_updates AS u
            WHERE d.track_id = u.track_id
              AND (d.duration_ms IS NULL OR d.explicit IS NULL)
            """
        )
        conn.execute("COMMIT;")
        return int(duration_to_set), int(explicit_to_set)
    finally:
        conn.close()


def populate_track_albums(
    *,
    db_path: str | Path,
    client_id: str | None = None,
    client_secret: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Populate `dim_tracks.album_id` and upsert albums into `dim_albums`.

    Strategy:
    - Find tracks with NULL `album_id`.
    - Fetch track metadata, extract album IDs/names/years.
    - Insert missing albums into `dim_albums` (anti-join).
    - Update `dim_tracks.album_id` where NULL.

    Returns counts: {"albums_inserted": x, "tracks_updated": y}.
    """
    # 1) Discover targets
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT track_id
            FROM dim_tracks
            WHERE album_id IS NULL
            ORDER BY track_id
            """
        ).fetchall()
    finally:
        conn.close()

    track_ids: list[str] = [r[0] for r in rows if r and r[0]]
    if limit is not None:
        track_ids = track_ids[: max(0, int(limit))]
    if not track_ids:
        return {"albums_inserted": 0, "tracks_updated": 0}

    # 2) Fetch track metadata
    collector = SpotifyDataCollector(
        client_id=client_id, client_secret=client_secret, cache_dir=cache_dir
    )
    tracks = collector.fetch_tracks(track_ids)

    # 3) Build frames for albums and track->album mapping
    album_rows: list[tuple[str, str, int | None]] = []
    track_album_rows: list[tuple[str, str]] = []
    for t in tracks:
        tid = t.get("id")
        album = t.get("album") or {}
        aid = album.get("id")
        aname_raw = album.get("name")
        # Parse release year (YYYY prefix of release_date)
        release_date = album.get("release_date")
        try:
            release_year = int(str(release_date)[:4]) if release_date else None
        except Exception:
            release_year = None
        if aid:
            aname = (
                aname_raw
                if (aname_raw is not None and str(aname_raw).strip())
                else f"spotify:{aid}"
            )
            album_rows.append((aid, aname, release_year))
        if tid and aid:
            track_album_rows.append((tid, aid))

    if not track_album_rows:
        return {"albums_inserted": 0, "tracks_updated": 0}

    df_albums = pd.DataFrame(
        album_rows, columns=["album_id", "album_name", "release_year"]
    ).drop_duplicates(subset=["album_id"], keep="first")
    df_track_album = pd.DataFrame(
        track_album_rows, columns=["track_id", "album_id"]
    ).drop_duplicates(subset=["track_id"], keep="first")

    # Ensure proper data types to avoid DuckDB ART index issues
    df_albums = df_albums.astype({"album_id": "string", "album_name": "string"})
    df_track_album = df_track_album.astype({"track_id": "string", "album_id": "string"})

    # 4) Apply DB changes
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_albums", df_albums)
        conn.register("df_track_album", df_track_album)

        albums_to_insert = conn.execute(
            """
            SELECT COUNT(*)
            FROM df_albums a
            ANTI JOIN dim_albums d ON d.album_id = a.album_id
            """
        ).fetchone()[0]

        tracks_to_update = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_tracks d
            JOIN df_track_album u ON d.track_id = u.track_id
            WHERE d.album_id IS NULL AND u.album_id IS NOT NULL
            """
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO dim_albums (album_id, album_name, release_year, label)
            SELECT a.album_id, a.album_name, a.release_year, NULL
            FROM df_albums a
            ANTI JOIN dim_albums d ON d.album_id = a.album_id
            """
        )

        conn.execute(
            """
            UPDATE dim_tracks AS d
            SET album_id = COALESCE(d.album_id, u.album_id)
            FROM df_track_album AS u
            WHERE d.track_id = u.track_id AND d.album_id IS NULL
            """
        )

        conn.execute("COMMIT;")
        return {"albums_inserted": int(albums_to_insert), "tracks_updated": int(tracks_to_update)}
    finally:
        conn.close()


def populate_track_artists(
    *,
    db_path: str | Path,
    client_id: str | None = None,
    client_secret: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Populate `dim_artists` and `bridge_track_artists` from track metadata.

    Strategy:
    - Find tracks that currently have no artist bridge rows.
    - Fetch track metadata and extract artist IDs/names and roles.
    - Insert missing artists into `dim_artists` (anti-join).
    - Insert missing bridge rows into `bridge_track_artists` (anti-join).

    Returns counts: {"artists_inserted": x, "bridges_inserted": y}.
    """
    # 1) Identify tracks lacking any artist bridges
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT t.track_id
            FROM dim_tracks t
            LEFT JOIN bridge_track_artists b ON b.track_id = t.track_id
            WHERE b.track_id IS NULL
            ORDER BY t.track_id
            """
        ).fetchall()
    finally:
        conn.close()

    track_ids: list[str] = [r[0] for r in rows if r and r[0]]
    if limit is not None:
        track_ids = track_ids[: max(0, int(limit))]
    if not track_ids:
        return {"artists_inserted": 0, "bridges_inserted": 0}

    # 2) Fetch tracks metadata
    collector = SpotifyDataCollector(
        client_id=client_id, client_secret=client_secret, cache_dir=cache_dir
    )
    tracks = collector.fetch_tracks(track_ids)

    # 3) Build frames for artists and bridges
    artist_rows: list[tuple[str, str]] = []
    bridge_rows: list[tuple[str, str, str]] = []  # track_id, artist_id, role
    for t in tracks:
        tid = t.get("id")
        artists = t.get("artists") or []
        for idx, a in enumerate(artists):
            aid = a.get("id")
            aname_raw = a.get("name")
            if not aid:
                continue
            role = "primary" if idx == 0 else "feature"
            aname = (
                aname_raw
                if (aname_raw is not None and str(aname_raw).strip())
                else f"spotify:{aid}"
            )
            artist_rows.append((aid, aname))
            if tid:
                bridge_rows.append((tid, aid, role))

    if not bridge_rows:
        return {"artists_inserted": 0, "bridges_inserted": 0}

    # Deduplicate artist and bridge rows in Python
    seen_artists: set[str] = set()
    unique_artists: list[tuple[str, str]] = []
    for aid, aname in artist_rows:
        if aid not in seen_artists:
            seen_artists.add(aid)
            unique_artists.append((aid, aname))

    unique_bridges = list(set(bridge_rows))  # Remove duplicates

    # 4) Apply DB changes using batch inserts to avoid DataFrame registration issues
    conn = duckdb.connect(str(db_path))
    try:
        # Get existing data to compute what needs inserting
        existing_artists = {
            r[0] for r in conn.execute("SELECT artist_id FROM dim_artists").fetchall()
        }
        existing_tracks = {r[0] for r in conn.execute("SELECT track_id FROM dim_tracks").fetchall()}
        bridge_results = conn.execute(
            "SELECT track_id, artist_id, role FROM bridge_track_artists"
        ).fetchall()
        existing_bridges = {(r[0], r[1], r[2]) for r in bridge_results}

        # Filter to only new artists
        new_artists = [(aid, aname) for aid, aname in unique_artists if aid not in existing_artists]

        # Filter to only new bridges where track exists
        new_bridges = [
            (tid, aid, role)
            for tid, aid, role in unique_bridges
            if (tid, aid, role) not in existing_bridges and tid in existing_tracks
        ]

        artists_to_insert = len(new_artists)
        bridges_to_insert = len(new_bridges)

        # Insert artists using executemany in batches
        batch_size = 1000
        if new_artists:
            for i in range(0, len(new_artists), batch_size):
                batch = new_artists[i : i + batch_size]
                conn.executemany(
                    "INSERT INTO dim_artists (artist_id, artist_name) VALUES (?, ?)", batch
                )

        # For bridge_track_artists: drop indexes, insert, recreate indexes
        # This works around DuckDB ART index corruption during bulk inserts
        if new_bridges:
            # Drop indexes temporarily
            try:
                conn.execute("DROP INDEX IF EXISTS idx_bridge_track_role")
                conn.execute("DROP INDEX IF EXISTS idx_bridge_artist_role")
            except Exception as e:
                logging.warning(f"Failed to drop indexes (may not exist): {e}")

            # Insert in batches without indexes
            bridge_batch_size = 5000  # Can use larger batches without indexes
            for i in range(0, len(new_bridges), bridge_batch_size):
                batch = new_bridges[i : i + bridge_batch_size]
                conn.executemany(
                    "INSERT INTO bridge_track_artists (track_id, artist_id, role) VALUES (?, ?, ?)",
                    batch,
                )

            # Recreate indexes
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bridge_track_role ON bridge_track_artists(track_id, role)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bridge_artist_role ON bridge_track_artists(artist_id, role)"
                )
            except Exception as e:
                logging.error(f"Failed to recreate indexes: {e}")
                raise

        return {
            "artists_inserted": artists_to_insert,
            "bridges_inserted": bridges_to_insert,
        }
    except Exception as e:
        logging.error(f"Error in populate_track_artists: {e}")
        raise
    finally:
        conn.close()


def populate_track_metadata(
    *,
    db_path: str | Path,
    client_id: str | None = None,
    client_secret: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Populate ISRC, duration_ms, and explicit in one pass.

    Fetches track metadata once for all tracks missing any of the three fields,
    then applies targeted updates while respecting uniqueness of `track_isrc`.

    Returns counts: {"isrc": x, "duration_ms": y, "explicit": z}.
    """
    # 1) Identify targets
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT track_id
            FROM dim_tracks
            WHERE (track_isrc IS NULL OR duration_ms IS NULL OR explicit IS NULL)
            ORDER BY track_id
            """
        ).fetchall()
    finally:
        conn.close()

    spotify_ids: list[str] = [r[0] for r in rows if r and r[0]]
    if limit is not None:
        spotify_ids = spotify_ids[: max(0, int(limit))]
    if not spotify_ids:
        return {"isrc": 0, "duration_ms": 0, "explicit": 0}

    # 2) Fetch metadata via cache-aware collector
    collector = SpotifyDataCollector(
        client_id=client_id, client_secret=client_secret, cache_dir=cache_dir
    )
    tracks = collector.fetch_tracks(spotify_ids)

    # 3) Build frames
    recs: list[tuple[str, str | None, int | None, bool | None]] = []
    for t in tracks:
        tid = t.get("id")
        if not tid:
            continue
        isrc = (t.get("external_ids") or {}).get("isrc")
        dur_raw = t.get("duration_ms")
        try:
            duration_ms = int(dur_raw) if dur_raw is not None else None
        except Exception:
            duration_ms = None
        explicit_val = t.get("explicit")
        explicit_bool = bool(explicit_val) if explicit_val is not None else None
        recs.append((tid, isrc, duration_ms, explicit_bool))

    if not recs:
        return {"isrc": 0, "duration_ms": 0, "explicit": 0}

    df_all = pd.DataFrame(recs, columns=["track_id", "track_isrc", "duration_ms", "explicit"])

    # Deduplicate ISRCs to avoid UNIQUE constraint violations
    df_isrc = df_all.dropna(subset=["track_isrc"]).copy()
    if not df_isrc.empty:
        # Keep first occurrence of each ISRC
        df_isrc = df_isrc.sort_values("track_id").drop_duplicates(
            subset=["track_isrc"], keep="first"
        )

    # 4) Apply updates in a single transaction
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_all", df_all)
        if not df_isrc.empty:
            conn.register("df_isrc", df_isrc[["track_id", "track_isrc"]])

        # Counts to be set (pre-update)
        isrc_to_set = 0
        if not df_isrc.empty:
            isrc_to_set = conn.execute(
                """
                SELECT COUNT(*)
                FROM dim_tracks d
                JOIN df_isrc u ON d.track_id = u.track_id
                WHERE d.track_isrc IS NULL
                  AND u.track_isrc IS NOT NULL
                  AND NOT EXISTS (
                        SELECT 1 FROM dim_tracks d2 WHERE d2.track_isrc = u.track_isrc
                  )
                """
            ).fetchone()[0]

        duration_to_set = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_tracks d
            JOIN df_all u ON d.track_id = u.track_id
            WHERE d.duration_ms IS NULL AND u.duration_ms IS NOT NULL
            """
        ).fetchone()[0]
        explicit_to_set = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_tracks d
            JOIN df_all u ON d.track_id = u.track_id
            WHERE d.explicit IS NULL AND u.explicit IS NOT NULL
            """
        ).fetchone()[0]

        # Updates
        if not df_isrc.empty:
            conn.execute(
                """
                UPDATE dim_tracks AS d
                SET track_isrc = u.track_isrc
                FROM df_isrc AS u
                WHERE d.track_id = u.track_id
                  AND d.track_isrc IS NULL
                  AND NOT EXISTS (
                        SELECT 1 FROM dim_tracks d2 WHERE d2.track_isrc = u.track_isrc
                  )
                """
            )

        conn.execute(
            """
            UPDATE dim_tracks AS d
            SET
              duration_ms = COALESCE(d.duration_ms, u.duration_ms),
              explicit     = COALESCE(d.explicit, u.explicit)
            FROM df_all AS u
            WHERE d.track_id = u.track_id
              AND (d.duration_ms IS NULL OR d.explicit IS NULL)
            """
        )

        conn.execute("COMMIT;")
        return {
            "isrc": int(isrc_to_set),
            "duration_ms": int(duration_to_set),
            "explicit": int(explicit_to_set),
        }
    finally:
        conn.close()


def populate_tracks_from_cached_albums(
    *,
    db_path: str | Path,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Load tracks listed inside cached album JSONs into `dim_tracks`.

    - Scans `cache_dir/albums/*.json` for album payloads.
    - Extracts each track from `album["tracks"]["items"]` with basic metadata.
    - Inserts missing rows into `dim_tracks` and fills NULL metadata for existing rows.

    Returns counts: {"albums_scanned": a, "tracks_inserted": i, "tracks_updated": u}.
    """
    # Determine cache directory (default to collector default)
    cache_env = os.getenv("SPOTIFY_API_CACHE_DIR")
    cache_root = Path(cache_dir or cache_env or "data/api/cache")
    albums_dir = cache_root / "albums"
    if not albums_dir.exists():
        return {"albums_scanned": 0, "tracks_inserted": 0, "tracks_updated": 0}

    album_files = sorted(albums_dir.glob("*.json"))
    if limit is not None:
        album_files = album_files[: max(0, int(limit))]

    rows: list[tuple[str, str, str | None, int | None, bool | None]] = []
    albums_scanned = 0
    for path in album_files:
        try:
            with open(path, encoding="utf-8") as f:
                album = json.load(f)
        except Exception:
            continue
        albums_scanned += 1
        aid = album.get("id")
        tracks_obj = album.get("tracks") or {}
        items = tracks_obj.get("items") or []
        for t in items:
            tid = t.get("id")
            if not tid:
                continue
            name_raw = t.get("name")
            name = (
                name_raw if (name_raw is not None and str(name_raw).strip()) else f"spotify:{tid}"
            )
            dur = t.get("duration_ms")
            try:
                duration_ms = int(dur) if dur is not None else None
            except Exception:
                duration_ms = None
            explicit_val = t.get("explicit")
            explicit_bool = bool(explicit_val) if explicit_val is not None else None
            rows.append((tid, name, aid, duration_ms, explicit_bool))

    if not rows:
        return {"albums_scanned": albums_scanned, "tracks_inserted": 0, "tracks_updated": 0}

    df_tracks = pd.DataFrame(
        rows, columns=["track_id", "track_name", "album_id", "duration_ms", "explicit"]
    ).drop_duplicates(subset=["track_id"], keep="first")

    # Ensure proper data types to avoid DuckDB ART index issues
    df_tracks = df_tracks.astype(
        {"track_id": "string", "track_name": "string", "album_id": "string"}
    )

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_tracks", df_tracks)

        tracks_to_insert = conn.execute(
            """
            SELECT COUNT(*)
            FROM df_tracks t
            ANTI JOIN dim_tracks d ON d.track_id = t.track_id
            """
        ).fetchone()[0]

        tracks_to_update = conn.execute(
            """
            SELECT COUNT(*)
            FROM dim_tracks d
            JOIN df_tracks u ON d.track_id = u.track_id
            WHERE (d.album_id IS NULL AND u.album_id IS NOT NULL)
               OR (d.duration_ms IS NULL AND u.duration_ms IS NOT NULL)
               OR (d.explicit IS NULL AND u.explicit IS NOT NULL)
               OR (d.track_name IS NULL AND u.track_name IS NOT NULL)
            """
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO dim_tracks (track_id, track_name, album_id, duration_ms, explicit)
            SELECT t.track_id, t.track_name, t.album_id, t.duration_ms, t.explicit
            FROM df_tracks t
            ANTI JOIN dim_tracks d ON d.track_id = t.track_id
            """
        )

        conn.execute(
            """
            UPDATE dim_tracks AS d
            SET
              track_name  = COALESCE(d.track_name, u.track_name),
              album_id    = COALESCE(d.album_id, u.album_id),
              duration_ms = COALESCE(d.duration_ms, u.duration_ms),
              explicit    = COALESCE(d.explicit, u.explicit)
            FROM df_tracks AS u
            WHERE d.track_id = u.track_id
              AND (d.album_id IS NULL OR d.duration_ms IS NULL OR d.explicit IS NULL OR d.track_name IS NULL)
            """
        )

        conn.execute("COMMIT;")
        return {
            "albums_scanned": int(albums_scanned),
            "tracks_inserted": int(tracks_to_insert),
            "tracks_updated": int(tracks_to_update),
        }
    finally:
        conn.close()


def populate_artist_genre_evidence_from_cache(
    *,
    db_path: str | Path,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Ingest artist genres from cached Spotify artist JSONs into `tag_evidence`.

    - Scans `cache_dir/artists/*.json`.
    - For each artist JSON, reads the `genres` list.
    - Inserts rows into `tag_evidence` with:
        entity_type='artist', entity_key=<artist_id>, source='spotify',
        tag_raw=<genre>, tag_kind='genre', weight_raw=1.0, observed_at=NOW.

    Returns counts: {"artists_scanned": a, "rows_inserted": i}.
    """
    cache_env = os.getenv("SPOTIFY_API_CACHE_DIR")
    cache_root = Path(cache_dir or cache_env or "data/api/cache")
    artists_dir = cache_root / "artists"
    if not artists_dir.exists():
        return {"artists_scanned": 0, "rows_inserted": 0}

    files = sorted(artists_dir.glob("*.json"))
    if limit is not None:
        files = files[: max(0, int(limit))]

    now = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None, microsecond=0)
    rows: list[tuple[str, str, str, str, str, float, str]] = []
    artists_scanned = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        artists_scanned += 1
        aid = data.get("id")
        genres = data.get("genres") or []
        if not aid or not isinstance(genres, list):
            continue
        for g in genres:
            if not g or not str(g).strip():
                continue
            rows.append(
                (
                    "artist",
                    str(aid),
                    "spotify",
                    str(g),
                    "genre",
                    1.0,
                    now.isoformat(sep=" ", timespec="seconds"),
                )
            )

    if not rows:
        return {"artists_scanned": artists_scanned, "rows_inserted": 0}

    df = pd.DataFrame(
        rows,
        columns=[
            "entity_type",
            "entity_key",
            "source",
            "tag_raw",
            "tag_kind",
            "weight_raw",
            "observed_at",
        ],
    )

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_evidence", df)

        to_insert = conn.execute(
            """
            SELECT COUNT(*)
            FROM df_evidence e
            ANTI JOIN tag_evidence t
              ON t.entity_type = e.entity_type
             AND t.entity_key  = e.entity_key
             AND t.source      = e.source
             AND t.tag_raw     = e.tag_raw
             AND t.tag_kind    = e.tag_kind
            """
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO tag_evidence (
                entity_type, entity_key, source, tag_raw, tag_kind, weight_raw, observed_at
            )
            SELECT e.entity_type, e.entity_key, e.source, e.tag_raw, e.tag_kind, e.weight_raw, e.observed_at
            FROM df_evidence e
            ANTI JOIN tag_evidence t
              ON t.entity_type = e.entity_type
             AND t.entity_key  = e.entity_key
             AND t.source      = e.source
             AND t.tag_raw     = e.tag_raw
             AND t.tag_kind    = e.tag_kind
            """
        )

        conn.execute("COMMIT;")
        return {"artists_scanned": int(artists_scanned), "rows_inserted": int(to_insert)}
    finally:
        conn.close()


def populate_artist_genres_from_cache(
    *,
    db_path: str | Path,
    cache_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
    source: str = "spotify",
) -> dict[str, int]:
    """Link artists to canonical genres in `artist_genres` using cached artist JSONs.

    Strategy:
    - Scan `cache_dir/artists/*.json` for artist payloads.
    - For each, read `id`, optional `name`, and `genres` list.
    - Map each raw genre tag to `dim_genres.genre_id` via `map_genre` for the given source.
    - Insert distinct `(artist_id, genre_id)` rows into `artist_genres` via anti-join.
      Only insert for artists that exist in `dim_artists` to satisfy FK.

    Returns counts: {"artists_scanned": a, "links_inserted": i, "unmapped_tags": u, "artists_missing_dim": m}.
    """
    cache_env = os.getenv("SPOTIFY_API_CACHE_DIR")
    cache_root = Path(cache_dir or cache_env or "data/api/cache")
    artists_dir = cache_root / "artists"
    if not artists_dir.exists():
        return {
            "artists_scanned": 0,
            "links_inserted": 0,
            "unmapped_tags": 0,
            "artists_missing_dim": 0,
        }

    files = sorted(artists_dir.glob("*.json"))
    if limit is not None:
        files = files[: max(0, int(limit))]

    # Load mapping dict from map_genre (source, tag_raw lowercased) -> genre_id
    conn = duckdb.connect(str(db_path))
    try:
        map_rows = conn.execute(
            """
            SELECT lower(tag_raw) AS tag_lc, genre_id
            FROM map_genre
            WHERE source = ?
            """,
            [source],
        ).fetchall()
        tag_to_genre: dict[str, int] = {
            str(tag_lc): int(genre_id) for (tag_lc, genre_id) in map_rows
        }
        # Also load dim_genres for direct child matches by name/slug
        dg_rows = conn.execute(
            """
            SELECT lower(name) AS name_lc, lower(slug) AS slug_lc, genre_id, COALESCE(level, -1) AS level
            FROM dim_genres
            WHERE COALESCE(active, TRUE)
            """
        ).fetchall()
        name_to_id: dict[str, int] = {}
        slug_to_id: dict[str, int] = {}
        level_by_id: dict[int, int] = {}
        for name_lc, slug_lc, gid, lvl in dg_rows:
            if isinstance(name_lc, str):
                name_to_id[name_lc] = int(gid)
            if isinstance(slug_lc, str):
                slug_to_id[slug_lc] = int(gid)
            level_by_id[int(gid)] = int(lvl)

        # Fetch existing artist IDs to satisfy FK
        artist_rows = conn.execute("SELECT artist_id FROM dim_artists").fetchall()
        existing_artists: set[str] = {r[0] for r in artist_rows if r and r[0]}
    finally:
        conn.close()

    rows: list[tuple[str, int]] = []
    artists_scanned = 0
    unmapped_tags = 0
    artists_missing_dim = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        aid = data.get("id")
        genres = data.get("genres") or []
        if not aid or not isinstance(genres, list):
            continue
        artists_scanned += 1
        if aid not in existing_artists:
            artists_missing_dim += 1
            continue
        for g in genres:
            if not g or not str(g).strip():
                continue
            raw = str(g).strip()
            glc = raw.lower()
            gid_parent = tag_to_genre.get(glc)
            matched_any = False
            if gid_parent is not None:
                rows.append((str(aid), int(gid_parent)))
                matched_any = True

            # Try direct child match by dim_genres name or slug (e.g., 'indie pop')
            # Slug policy follows load_genre_taxonomy: replace spaces with underscores only
            slug = raw.replace(" ", "_").lower()
            gid_child = name_to_id.get(glc) or slug_to_id.get(slug)
            if gid_child is not None:
                # Update to use level_by_id: only create direct link for level 1,
                # or level 0 only if no parent mapping exists for this tag
                level = level_by_id.get(gid_child, -1)
                if level == 1 or (level == 0 and gid_parent is None):
                    rows.append((str(aid), int(gid_child)))
                    matched_any = True

            if not matched_any:
                unmapped_tags += 1

    if not rows:
        return {
            "artists_scanned": artists_scanned,
            "links_inserted": 0,
            "unmapped_tags": unmapped_tags,
            "artists_missing_dim": artists_missing_dim,
        }

    df = pd.DataFrame(rows, columns=["artist_id", "genre_id"]).drop_duplicates()

    # Ensure proper data types to avoid DuckDB ART index issues
    df = df.astype({"artist_id": "string"})

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_links", df)

        to_insert = conn.execute(
            """
            SELECT COUNT(*)
            FROM df_links l
            ANTI JOIN artist_genres ag
              ON ag.artist_id = l.artist_id AND ag.genre_id = l.genre_id
            """
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO artist_genres (artist_id, genre_id)
            SELECT l.artist_id, l.genre_id
            FROM df_links l
            -- Satisfy FKs explicitly, though we already filtered
            JOIN dim_artists a ON a.artist_id = l.artist_id
            JOIN dim_genres  g ON g.genre_id  = l.genre_id
            ANTI JOIN artist_genres ag
              ON ag.artist_id = l.artist_id AND ag.genre_id = l.genre_id
            """
        )

        conn.execute("COMMIT;")
        return {
            "artists_scanned": int(artists_scanned),
            "links_inserted": int(to_insert),
            "unmapped_tags": int(unmapped_tags),
            "artists_missing_dim": int(artists_missing_dim),
        }
    finally:
        conn.close()


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
        track_ids = [s for line in f if (s := line.strip()) and not s.startswith("#")]
    with open(collector.data_dir / "artist_ids.txt", encoding="utf-8") as f:
        artist_ids = [s for line in f if (s := line.strip()) and not s.startswith("#")]
    with open(collector.data_dir / "album_ids.txt", encoding="utf-8") as f:
        album_ids = [s for line in f if (s := line.strip()) and not s.startswith("#")]

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
