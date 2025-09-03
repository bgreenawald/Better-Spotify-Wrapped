import contextlib
from pathlib import Path
from typing import Any

import pandas as pd


def get_most_played_tracks(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the most played tracks from a filtered Spotify listening history.

    Groups tracks by track name and artist, counts plays, and computes play
    percentages.

    Args:
        filtered_df (pd.DataFrame): DataFrame already filtered by filter_songs().

    Returns:
        pd.DataFrame: Tracks sorted by play count with columns:
            'track_artist', 'track_name', 'artist', 'artist_id',
            'artist_genres', 'play_count', 'percentage'.
    """
    df = filtered_df.copy()
    # Combine track name and artist for grouping
    df["track_artist"] = (
        df["master_metadata_track_name"] + " - " + df["master_metadata_album_artist_name"]
    )

    # Count plays per track
    grouped = (
        df.groupby(
            [
                "track_artist",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
                "artist_id",
                "artist_genres",
            ]
        )
        .size()
        .reset_index(name="play_count")
    )

    # Sort by descending play count
    sorted_df = grouped.sort_values("play_count", ascending=False)

    # Calculate percentage of total plays
    total_plays = sorted_df["play_count"].sum()
    sorted_df["percentage"] = sorted_df["play_count"] / total_plays

    # Rename columns for clarity
    sorted_df.rename(
        columns={
            "master_metadata_track_name": "track_name",
            "master_metadata_album_artist_name": "artist",
        },
        inplace=True,
    )

    return sorted_df


def get_top_albums(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
) -> pd.DataFrame:
    """Calculate top albums based on median song plays (with zero-play tracks included).

    Uses DuckDB exclusively: provide `con` or `db_path` to query the schema in DDL.sql.

    Args:
        filtered_df: DataFrame already filtered by `filter_songs()` or
            `get_filtered_plays()`.
        db_path: Optional DuckDB database path (if `con` not provided).
        con: Optional DuckDB connection to use.

    Notes:
        - Excludes short releases (albums with 5 or fewer tracks) to prevent
          singles/EPs from dominating via inflated medians.

    Returns:
        pd.DataFrame: Albums sorted by median play count with columns:
            'album_name', 'artist', 'median_plays', 'total_tracks',
            'tracks_played', and optionally 'release_year'.
    """
    # Fast exit
    if filtered_df.empty:
        return pd.DataFrame(
            columns=[
                "album_name",
                "artist",
                "median_plays",
                "total_tracks",
                "tracks_played",
                "release_year",
            ]
        )

    # Helper to extract track_id from spotify URI
    def _extract_track_id(uri: Any) -> str | None:
        if not isinstance(uri, str):
            return None
        if uri.startswith("spotify:track:"):
            return uri.split(":")[-1]
        if "open.spotify.com/track/" in uri:
            part = uri.split("open.spotify.com/track/")[-1]
            return part.split("?")[0]
        return None

    # Require DuckDB backend
    use_duckdb = (con is not None) or (db_path is not None)
    if use_duckdb:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError("DuckDB backend requested, but duckdb is not installed.") from exc
        # Build or use provided connection
        close_conn = False
        if con is None:
            con = duckdb.connect(str(db_path))
            close_conn = True
        try:
            # Derive play counts per track_id from filtered_df
            df = filtered_df.copy()
            if "track_id" not in df.columns:
                df["track_id"] = df.get("spotify_track_uri").apply(_extract_track_id)
            play_counts = (
                df.dropna(subset=["track_id"])
                .groupby("track_id")
                .size()
                .reset_index(name="play_count")
            )
            if play_counts.empty:
                return pd.DataFrame(
                    columns=[
                        "album_name",
                        "artist",
                        "median_plays",
                        "total_tracks",
                        "tracks_played",
                        "release_year",
                    ]
                )

            # Register plays as a temporary relation
            with contextlib.suppress(Exception):
                con.unregister("df_play_counts")  # type: ignore[attr-defined]
            con.register("df_play_counts", play_counts)

            # SQL plan:
            # 1) Identify album_ids present in plays via dim_tracks
            # 2) Expand to all tracks for those albums
            # 3) Left join play counts to include zero-play tracks
            # 4) Compute per-album aggregates + pick a primary artist (mode over primary role)
            sql = """
                WITH played_tracks AS (
                    SELECT t.album_id, t.track_id, p.play_count
                    FROM df_play_counts p
                    JOIN dim_tracks t ON t.track_id = p.track_id
                    WHERE t.album_id IS NOT NULL
                ),
                albums_in_scope AS (
                    SELECT DISTINCT album_id FROM played_tracks
                ),
                all_album_tracks AS (
                    SELECT t.album_id, t.track_id
                    FROM dim_tracks t
                    JOIN albums_in_scope s ON s.album_id = t.album_id
                ),
                album_track_counts AS (
                    SELECT a.album_id,
                        aat.track_id,
                        COALESCE(p.play_count, 0) AS play_count
                    FROM all_album_tracks aat
                    LEFT JOIN df_play_counts p ON p.track_id = aat.track_id
                    JOIN dim_albums a ON a.album_id = aat.album_id
                ),
                album_artist_counts AS (
                    SELECT t.album_id,
                        ar.artist_name,
                        COUNT(*) AS cnt
                    FROM dim_tracks t
                    JOIN albums_in_scope s ON s.album_id = t.album_id
                    JOIN bridge_track_artists b
                    ON b.track_id = t.track_id AND b."role" = 'primary'
                    JOIN dim_artists ar ON ar.artist_id = b.artist_id
                    GROUP BY 1, 2
                ),
                album_primary_artist AS (
                    SELECT album_id,
                        artist_name,
                        ROW_NUMBER() OVER (PARTITION BY album_id ORDER BY cnt DESC, artist_name) AS rn
                    FROM album_artist_counts
                )
                SELECT
                    al.album_name,
                    COALESCE(pa.artist_name, '') AS artist,
                    MEDIAN(atc.play_count) AS median_plays,
                    COUNT(*) AS total_tracks,
                    SUM(CASE WHEN atc.play_count > 0 THEN 1 ELSE 0 END) AS tracks_played,
                    al.release_year
                FROM album_track_counts atc
                JOIN dim_albums al ON al.album_id = atc.album_id
                LEFT JOIN album_primary_artist pa ON pa.album_id = atc.album_id AND pa.rn = 1
                GROUP BY 1, 2, 6
                HAVING COUNT(*) > 5
                ORDER BY median_plays DESC, total_tracks DESC, album_name;

            """
            res = con.execute(sql).df()
            # Ensure types and ordering
            if not res.empty:
                res["median_plays"] = res["median_plays"].astype(float)
                res = res.sort_values("median_plays", ascending=False)
            return res
        finally:
            if close_conn:
                con.close()
    # If we get here, neither `con` nor `db_path` was provided
    raise ValueError("get_top_albums requires a DuckDB connection or db_path.")


def get_top_artist_genres(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
    unique_tracks: bool = False,
    top_artists_per_genre: int = 2,
) -> pd.DataFrame:
    """Calculate the most common artist genres in the listening history.

    DuckDB-backed implementation using the normalized schema (see DDL.sql).

    Args:
        filtered_df: DataFrame already filtered by filter_songs(). Must contain
            a 'spotify_track_uri' column to extract Spotify track IDs from.
        unique_tracks: If True, consider each track ID only once regardless of play count.
        top_artists_per_genre: Number of top artists to include per genre in the summary.
        db_path: Optional path to a DuckDB database. If provided (or `con`), the DuckDB path is used.
        con: Optional DuckDB connection to use.

    Returns:
        pd.DataFrame: Genres sorted by frequency with columns:
            'genre', 'play_count', 'percentage', 'top_artists'.
    """

    # Helper to extract track_id from spotify URI
    def _extract_track_id(uri: Any) -> str | None:
        if not isinstance(uri, str):
            return None
        if uri.startswith("spotify:track:"):
            return uri.split(":")[-1]
        if "open.spotify.com/track/" in uri:
            part = uri.split("open.spotify.com/track/")[-1]
            return part.split("?")[0]
        # Fallback to last colon segment if present
        if ":" in uri:
            return uri.split(":")[-1]
        return None

    use_duckdb = (con is not None) or (db_path is not None)
    if use_duckdb:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError("DuckDB backend requested, but duckdb is not installed.") from exc

        close_conn = False
        if con is None:
            con = duckdb.connect(str(db_path))
            close_conn = True
        try:
            # Prepare play counts per track_id from filtered_df
            df = filtered_df.copy()
            if "track_id" not in df.columns:
                df["track_id"] = df.get("spotify_track_uri").apply(_extract_track_id)
            plays = df.dropna(subset=["track_id"]).copy()

            if unique_tracks:
                # Each unique track contributes exactly 1
                play_counts = plays.drop_duplicates(subset=["track_id"])[["track_id"]].assign(
                    play_count=1
                )
                total_tracked = int(play_counts.shape[0])
            else:
                # Aggregate counts per track_id
                play_counts = plays.groupby("track_id").size().reset_index(name="play_count")
                total_tracked = int(play_counts["play_count"].sum())

            if play_counts.empty or total_tracked == 0:
                return pd.DataFrame(columns=["genre", "play_count", "percentage", "top_artists"])

            # Register temp relation
            with contextlib.suppress(Exception):
                con.unregister("df_play_counts")  # type: ignore[attr-defined]
            con.register("df_play_counts", play_counts)

            # Total counts per genre (via primary artist for each track)
            sql_genre_counts = """
                WITH primary_artist AS (
                    SELECT track_id, artist_id
                    FROM (
                        SELECT b.track_id,
                               b.artist_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY b.track_id
                                   ORDER BY a.artist_name
                               ) AS rn
                        FROM bridge_track_artists b
                        JOIN dim_artists a ON a.artist_id = b.artist_id
                        WHERE b."role" = 'primary'
                    ) x
                    WHERE rn = 1
                )
                SELECT g.name AS genre,
                       SUM(p.play_count) AS play_count
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                GROUP BY 1
                ORDER BY play_count DESC, genre
            """
            genre_counts_df = con.execute(sql_genre_counts).df()

            if genre_counts_df.empty:
                return pd.DataFrame(columns=["genre", "play_count", "percentage", "top_artists"])

            # Per-genre, per-artist counts to compute top artists listing
            sql_artist_genre_counts = """
                WITH primary_artist AS (
                    SELECT track_id, artist_id
                    FROM (
                        SELECT b.track_id,
                               b.artist_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY b.track_id
                                   ORDER BY a.artist_name
                               ) AS rn
                        FROM bridge_track_artists b
                        JOIN dim_artists a ON a.artist_id = b.artist_id
                        WHERE b."role" = 'primary'
                    ) x
                    WHERE rn = 1
                )
                SELECT g.name AS genre,
                       a.artist_name AS artist,
                       SUM(p.play_count) AS play_count
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN dim_artists a ON a.artist_id = pa.artist_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                GROUP BY 1, 2
            """
            artist_genre_df = con.execute(sql_artist_genre_counts).df()

            # Build final frame: add percentage and formatted top artists per genre
            genre_counts_df["percentage"] = (
                (genre_counts_df["play_count"] / total_tracked) * 100
            ).round(2)

            # Format top artists
            def format_top_artists(genre: str) -> str:
                subset = artist_genre_df[artist_genre_df["genre"] == genre]
                if subset.empty:
                    return ""
                subset = subset.sort_values(["play_count", "artist"], ascending=[False, True])
                top_rows = subset.head(top_artists_per_genre)
                return ", ".join(
                    f"{row['artist']} ({int(row['play_count'])} plays)"
                    for _, row in top_rows.iterrows()
                )

            genre_counts_df["top_artists"] = genre_counts_df["genre"].apply(format_top_artists)
            genre_counts_df.rename(columns={"genre": "genre"}, inplace=True)
            # Standardize column order
            result = genre_counts_df[["genre", "play_count", "percentage", "top_artists"]]
            return result.sort_values(["play_count", "genre"], ascending=[False, True])
        finally:
            if close_conn:
                con.close()
    # If we reach here, neither `con` nor `db_path` was provided
    raise ValueError("get_top_artist_genres requires a DuckDB connection or db_path.")


def get_most_played_artists(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the most played artists from a filtered Spotify listening history.

    Counts total plays and unique tracks per artist, with play percentages.

    Args:
        filtered_df (pd.DataFrame): DataFrame filtered by filter_songs().

    Returns:
        pd.DataFrame: Artists sorted by play count with columns:
            'artist', 'artist_id', 'artist_genres', 'play_count',
            'unique_tracks', 'percentage'.
    """
    df = filtered_df.copy()

    # Total plays per artist
    plays = (
        df.groupby(
            [
                "master_metadata_album_artist_name",
                "artist_id",
                "artist_genres",
            ]
        )
        .size()
        .reset_index(name="play_count")
    )

    # Unique tracks per artist
    uniques = (
        df.groupby(
            [
                "master_metadata_album_artist_name",
                "artist_id",
                "artist_genres",
            ]
        )["master_metadata_track_name"]
        .nunique()
        .reset_index(name="unique_tracks")
    )

    # Merge play counts and unique track counts
    stats = pd.merge(
        plays,
        uniques,
        on=[
            "master_metadata_album_artist_name",
            "artist_id",
            "artist_genres",
        ],
    )

    # Sort and compute percentage
    stats = stats.sort_values("play_count", ascending=False)
    total_plays = stats["play_count"].sum()
    stats["percentage"] = (stats["play_count"] / total_plays * 100).round(2)

    # Rename columns for clarity
    stats.rename(
        columns={"master_metadata_album_artist_name": "artist"},
        inplace=True,
    )

    return stats


def get_playcount_by_day(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily play counts from a filtered Spotify listening history.

    Groups by date, track, and artist to count plays per day.

    Args:
        filtered_df (pd.DataFrame): DataFrame filtered by filter_songs().

    Returns:
        pd.DataFrame: Daily play counts with columns:
            'date', 'track', 'artist', 'play_count'.
    """
    df = filtered_df.copy()
    # Extract date from timestamp
    df["date"] = df["ts"].dt.date

    # Group by date, track name, and artist
    daily_counts = (
        df.groupby(
            [
                "date",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
            ]
        )
        .size()
        .reset_index(name="play_count")
    )

    # Rename columns for clarity
    daily_counts.rename(
        columns={
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
        },
        inplace=True,
    )

    # Sort by date and descending play count
    return daily_counts.sort_values(["date", "play_count"], ascending=[True, False])
