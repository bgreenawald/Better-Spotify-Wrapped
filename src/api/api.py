import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
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
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """Initialize the collector with Spotify API credentials and cache dirs.

        Args:
            client_id: Spotify API client ID.
            client_secret: Spotify API client secret.
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
        _ = conn.execute(
            """
            SELECT track_id
            FROM dim_tracks
            WHERE track_isrc IS NULL
            ORDER BY track_id
            """
        ).fetchall()
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

    df_artists = pd.DataFrame(artist_rows, columns=["artist_id", "artist_name"]).drop_duplicates(
        subset=["artist_id"], keep="first"
    )
    df_bridge = pd.DataFrame(
        bridge_rows, columns=["track_id", "artist_id", "role"]
    ).drop_duplicates()

    # 4) Apply DB changes
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.register("df_artists", df_artists)
        conn.register("df_bridge", df_bridge)

        artists_to_insert = conn.execute(
            """
            SELECT COUNT(*)
            FROM df_artists a
            ANTI JOIN dim_artists d ON d.artist_id = a.artist_id
            """
        ).fetchone()[0]

        bridges_to_insert = conn.execute(
            """
            SELECT COUNT(*)
            FROM df_bridge b
            ANTI JOIN bridge_track_artists d
              ON d.track_id = b.track_id AND d.artist_id = b.artist_id AND d.role = b.role
            """
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO dim_artists (artist_id, artist_name)
            SELECT a.artist_id, a.artist_name
            FROM df_artists a
            ANTI JOIN dim_artists d ON d.artist_id = a.artist_id
            """
        )

        conn.execute(
            """
            INSERT INTO bridge_track_artists (track_id, artist_id, role)
            SELECT b.track_id, b.artist_id, b.role
            FROM df_bridge b
            -- Ensure the referenced track exists to satisfy FK
            JOIN dim_tracks t ON t.track_id = b.track_id
            ANTI JOIN bridge_track_artists d
              ON d.track_id = b.track_id AND d.artist_id = b.artist_id AND d.role = b.role
            """
        )

        conn.execute("COMMIT;")
        return {
            "artists_inserted": int(artists_to_insert),
            "bridges_inserted": int(bridges_to_insert),
        }
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
