"""Database ingestion utilities for loading Spotify listening history.

This module provides a function to load listening history JSON records into the
normalized schema defined in DDL.sql, specifically inserting into `fact_plays`
and ensuring minimal supporting rows exist in `dim_users` and `dim_tracks`.

Assumptions/Notes:
- Tables are created ahead of time by applying `DDL.sql` to the DuckDB DB.
- We only ingest music track plays (skip podcast episodes).
- `track_id` uses the Spotify Track ID directly (PK), not a derived UUID.
- Minimal `dim_tracks` attributes are populated from the history export; fields
  like `album_id`, `duration_ms` of the track, and `explicit` may be unknown.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

# Local import kept optional to avoid hard dependency; function accepts either
# a directory path of JSON exports or a prebuilt DataFrame of records.
try:
    from src.io import load_spotify_history  # type: ignore
except Exception:  # pragma: no cover - defensive import
    load_spotify_history = None  # type: ignore


@dataclass
class IngestResult:
    inserted_plays: int
    deduped_plays: int
    inserted_tracks: int
    existing_tracks: int


SPOTIFY_TRACK_NAMESPACE = uuid.NAMESPACE_URL  # preserved for backward compat if needed


def _extract_spotify_track_id(uri: str | None) -> str | None:
    if not uri:
        return None
    # Accept both `spotify:track:ID` and `https://open.spotify.com/track/ID` forms
    if uri.startswith("spotify:track:"):
        return uri.split(":")[-1]
    if "open.spotify.com/track/" in uri:
        # Strip query params if present
        part = uri.split("open.spotify.com/track/")[-1]
        return part.split("?")[0]
    return None


def _to_timestamp(ts: object) -> str | None:
    """Normalize input timestamp to ISO 8601 string without timezone.

    Accepts strings or pandas Timestamps. Returns ISO-like string that SQLite
    can store as TEXT or NUMERIC and be compatible with `TIMESTAMP`.
    """
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)  # type: ignore[assignment]
        return ts.to_pydatetime().isoformat(sep=" ", timespec="seconds")
    if isinstance(ts, str):
        try:
            parsed = pd.to_datetime(ts, errors="coerce")
            if pd.isna(parsed):
                return None
            if parsed.tzinfo is not None:
                parsed = parsed.tz_convert(None)
            return parsed.to_pydatetime().isoformat(sep=" ", timespec="seconds")
        except Exception:
            return None
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return ts.isoformat(sep=" ", timespec="seconds")
    return None


def _ensure_user(conn: duckdb.DuckDBPyConnection, user_id: str) -> None:
    exists = conn.execute(
        "SELECT 1 FROM dim_users WHERE user_id = ? LIMIT 1",
        [user_id],
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO dim_users (user_id) VALUES (?)",
            [user_id],
        )


def _upsert_track(
    conn: duckdb.DuckDBPyConnection, *, track_spotify_id: str, track_name: str | None
) -> bool:
    """Insert a row into `dim_tracks` if it does not already exist.

    Returns True if a new row was inserted; False if it already existed.
    """
    # Spotify track ID is the primary key
    track_id = track_spotify_id
    # Ensure a non-null track_name to satisfy NOT NULL constraint
    safe_name = (
        track_name
        if (track_name is not None and str(track_name).strip())
        else f"spotify:{track_spotify_id}"
    )
    exists = conn.execute(
        "SELECT 1 FROM dim_tracks WHERE track_id = ? LIMIT 1",
        [track_id],
    ).fetchone()
    if exists:
        return False
    conn.execute(
        """
        INSERT INTO dim_tracks (
            track_id, track_name, album_id, duration_ms, explicit
        ) VALUES (?, ?, NULL, NULL, NULL)
        """,
        [track_id, safe_name],
    )
    return True


def _insert_play(
    conn: duckdb.DuckDBPyConnection,
    *,
    user_id: str,
    track_spotify_id: str,
    played_at_iso: str,
    device_type: str | None,
    duration_ms: int | None,
) -> bool:
    # Spotify ID is the track_id (no transformation)
    track_id = track_spotify_id
    play_id = str(uuid.uuid4())
    exists = conn.execute(
        "SELECT 1 FROM fact_plays WHERE user_id = ? AND track_id = ? AND played_at = ? LIMIT 1",
        [user_id, track_id, played_at_iso],
    ).fetchone()
    if exists:
        return False
    conn.execute(
        """
        INSERT INTO fact_plays (
            play_id,
            user_id,
            track_id,
            played_at,
            context_type,
            context_id,
            device_type,
            duration_ms,
            conn_country,
            ip_addr,
            reason_start,
            reason_end,
            shuffle,
            skipped,
            offline,
            offline_timestamp,
            incognito_mode
        ) VALUES (
            ?, ?, ?, ?,
            NULL, NULL,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            play_id,
            user_id,
            track_id,
            played_at_iso,
            device_type,
            duration_ms,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    )
    return True


def load_history_into_fact_plays(
    *,
    db_path: str | Path,
    user_id: str,
    history_dir: str | Path | None = None,
) -> IngestResult:
    """Load listening history into `fact_plays` and supporting dims.

    Provide either `history_dir` (a directory with Spotify JSON export files)
    or `history_df` (a preloaded DataFrame). Podcast plays (episode rows) are
    skipped; only music tracks with a Spotify track URI are ingested.

    Args:
        db_path: Path to SQLite DB (e.g., `data/db/music.db`).
        user_id: Application-level user identifier (UUID string).
        history_dir: Directory containing `*.json` history files.

    Returns:
        IngestResult: counts of inserted vs deduped records.
    """

    history_df = load_spotify_history(str(history_dir))  # type: ignore[arg-type]

    # Normalize and filter rows
    df = history_df.copy()

    # Deduplicate
    df = df.drop_duplicates(subset=["spotify_track_uri", "ts"], keep="first")

    # Ensure timestamp is string ISO
    df["played_at_iso"] = df["ts"].apply(_to_timestamp)

    # Extract Spotify track IDs
    df["spotify_track_id_only"] = df["spotify_track_uri"].apply(_extract_spotify_track_id)

    # Filter to track plays only
    mask_tracks = df["spotify_track_id_only"].notna() & df["played_at_iso"].notna()
    if "spotify_episode_uri" in df.columns:
        mask_tracks &= df["spotify_episode_uri"].isna()
    if "episode_name" in df.columns:
        mask_tracks &= df["episode_name"].isna()

    df_tracks = df.loc[mask_tracks].copy()

    # Prepare bulk frames
    # Tracks to consider
    tracks_cols = [
        "spotify_track_id_only",
        "master_metadata_track_name",
    ]
    tracks_df = (
        df_tracks[tracks_cols]
        .dropna(subset=["spotify_track_id_only"])  # keep only rows with track ids
        .drop_duplicates(subset=["spotify_track_id_only"])
        .rename(columns={"master_metadata_track_name": "track_name"})
        .reset_index(drop=True)
    )
    # Derive track_id (Spotify ID) and safe track_name
    if not tracks_df.empty:
        tracks_df["track_id"] = tracks_df["spotify_track_id_only"]
        tracks_df["track_name"] = tracks_df.apply(
            lambda r: r["track_name"]
            if (pd.notna(r["track_name"]) and str(r["track_name"]).strip())
            else f"spotify:{r['spotify_track_id_only']}",
            axis=1,
        )

    # Plays to consider
    def _clean_ms(v: object) -> int | None:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        try:
            return int(v)  # type: ignore[arg-type]
        except Exception:
            return None

    plays_df = pd.DataFrame()
    if not df_tracks.empty:
        plays_df = pd.DataFrame(
            {
                "play_id": [str(uuid.uuid4()) for _ in range(len(df_tracks))],
                "user_id": user_id,
                "track_id": df_tracks["spotify_track_id_only"],
                "played_at": df_tracks["played_at_iso"],
                "device_type": df_tracks.get("platform"),
                "duration_ms": df_tracks["ms_played"].apply(_clean_ms)
                if "ms_played" in df_tracks.columns
                else None,
                "conn_country": df_tracks.get("conn_country"),
                "ip_addr": df_tracks.get("ip_addr"),
                "reason_start": df_tracks.get("reason_start"),
                "reason_end": df_tracks.get("reason_end"),
                "shuffle": df_tracks.get("shuffle"),
                "skipped": df_tracks.get("skipped"),
                "offline": df_tracks.get("offline"),
                "offline_timestamp": df_tracks.get("offline_timestamp").apply(_to_timestamp)
                if "offline_timestamp" in df_tracks.columns
                else None,
                "incognito_mode": df_tracks.get("incognito_mode"),
            }
        )

    # Connect to DuckDB
    db_path = str(db_path)
    conn = duckdb.connect(db_path)
    try:
        conn.execute("BEGIN TRANSACTION;")
        # Ensure user exists
        _ensure_user(conn, user_id=user_id)

        inserted_tracks = 0
        existing_tracks = 0
        inserted_plays = 0
        deduped_plays = 0

        # Bulk upsert tracks using anti-join
        if not tracks_df.empty:
            conn.register("to_tracks", tracks_df[["track_id", "track_name"]])
            inserted_tracks = conn.execute(
                """
                SELECT COUNT(*) FROM to_tracks t
                ANTI JOIN dim_tracks d ON d.track_id = t.track_id
                """
            ).fetchone()[0]
            existing_tracks = int(len(tracks_df) - inserted_tracks)
            conn.execute(
                """
                INSERT INTO dim_tracks (track_id, track_name, album_id, duration_ms, explicit)
                SELECT t.track_id, t.track_name, NULL, NULL, NULL
                FROM to_tracks t
                ANTI JOIN dim_tracks d ON d.track_id = t.track_id
                """
            )

        # Bulk insert plays using anti-join on dedupe key
        if not plays_df.empty:
            conn.register("df_plays", plays_df)
            inserted_plays = conn.execute(
                """
                SELECT COUNT(*)
                FROM df_plays p
                ANTI JOIN fact_plays f
                  ON f.user_id = p.user_id AND f.track_id = p.track_id AND f.played_at = p.played_at
                """
            ).fetchone()[0]
            deduped_plays = int(len(plays_df) - inserted_plays)

            conn.execute(
                """
                INSERT INTO fact_plays (
                    play_id,
                    user_id,
                    track_id,
                    played_at,
                    context_type,
                    context_id,
                    device_type,
                    duration_ms,
                    conn_country,
                    ip_addr,
                    reason_start,
                    reason_end,
                    shuffle,
                    skipped,
                    offline,
                    offline_timestamp,
                    incognito_mode
                )
                SELECT
                    p.play_id,
                    p.user_id,
                    p.track_id,
                    p.played_at,
                    NULL,
                    NULL,
                    p.device_type,
                    p.duration_ms,
                    p.conn_country,
                    p.ip_addr,
                    p.reason_start,
                    p.reason_end,
                    p.shuffle,
                    p.skipped,
                    p.offline,
                    p.offline_timestamp,
                    p.incognito_mode
                FROM df_plays p
                ANTI JOIN fact_plays f
                  ON f.user_id = p.user_id AND f.track_id = p.track_id AND f.played_at = p.played_at
                """
            )

        conn.execute("COMMIT;")
        return IngestResult(
            inserted_plays=inserted_plays,
            deduped_plays=deduped_plays,
            inserted_tracks=inserted_tracks,
            existing_tracks=existing_tracks,
        )
    finally:
        conn.close()


__all__ = [
    "IngestResult",
    "load_history_into_fact_plays",
]
