import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.utils import extract_track_id


def get_most_played_tracks(
    filtered_df: pd.DataFrame,
    *,
    con: Any | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Most played tracks using DuckDB when available.

    Falls back to pandas groupby if no connection provided.

    Returns columns: 'track_artist', 'track_name', 'artist', 'artist_id',
    'artist_genres', 'play_count', 'percentage' (fraction of total plays).
    """
    if filtered_df.empty:
        return pd.DataFrame(
            columns=[
                "track_artist",
                "track_name",
                "artist",
                "artist_id",
                "artist_genres",
                "play_count",
                "percentage",
            ]
        )

    if con is not None:
        # Work on a reduced view; exclude artist_genres here to avoid object dtype issues in DuckDB
        df = filtered_df[
            [
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
                "artist_id",
            ]
        ].copy()

        rel = "df_tracks_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        lim = f"LIMIT {int(limit)}" if (limit is not None and limit > 0) else ""
        sql = f"""
            WITH plays AS (
                SELECT
                    (master_metadata_track_name || ' - ' || master_metadata_album_artist_name) AS track_artist,
                    master_metadata_track_name AS track_name,
                    master_metadata_album_artist_name AS artist,
                    COALESCE(artist_id, '') AS artist_id,
                    COUNT(*) AS play_count
                FROM {rel}
                GROUP BY 1,2,3,4
            ), totals AS (
                SELECT SUM(play_count) AS total_plays FROM plays
            )
            SELECT p.track_artist, p.track_name, p.artist, p.artist_id,
                   p.play_count,
                   (p.play_count::DOUBLE / NULLIF(t.total_plays, 0)) AS percentage
            FROM plays p CROSS JOIN totals t
            ORDER BY p.play_count DESC, p.track_artist
            {lim}
        """
        res = con.execute(sql).df()
        # Bring back artist_genres via a lightweight mapping from the filtered_df
        try:
            if "artist_genres" in filtered_df.columns and not res.empty:
                m = (
                    filtered_df[
                        [
                            "master_metadata_track_name",
                            "master_metadata_album_artist_name",
                            "artist_id",
                            "artist_genres",
                        ]
                    ]
                    .dropna(
                        subset=["master_metadata_track_name", "master_metadata_album_artist_name"]
                    )  # type: ignore[list-item]
                    .drop_duplicates(
                        subset=[
                            "master_metadata_track_name",
                            "master_metadata_album_artist_name",
                            "artist_id",
                        ]
                    )
                )
                m = m.rename(
                    columns={
                        "master_metadata_track_name": "track_name",
                        "master_metadata_album_artist_name": "artist",
                    }
                )
                m["artist_id"] = m["artist_id"].fillna("")
                res = res.merge(
                    m,
                    on=["track_name", "artist", "artist_id"],
                    how="left",
                )
        except Exception:
            # If anything goes wrong, default to empty tuples
            res["artist_genres"] = [() for _ in range(len(res))]
        # Guarantee artist_genres exists and handle NaNs
        if not res.empty and "artist_genres" not in res.columns:
            res["artist_genres"] = [() for _ in range(len(res))]
        if not res.empty and "artist_genres" in res.columns:
            res["artist_genres"] = res["artist_genres"].fillna(())
        # Ensure stable types
        if not res.empty:
            res["percentage"] = res["percentage"].fillna(0.0)
        return res

    # Fallback to pandas path
    df = filtered_df.copy()
    df["track_artist"] = (
        df["master_metadata_track_name"] + " - " + df["master_metadata_album_artist_name"]
    )
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
    grouped = grouped.sort_values("play_count", ascending=False)
    total_plays = grouped["play_count"].sum() or 1
    if limit is not None and limit > 0:
        grouped = grouped.head(limit)
    grouped["percentage"] = grouped["play_count"] / total_plays
    grouped.rename(
        columns={
            "master_metadata_track_name": "track_name",
            "master_metadata_album_artist_name": "artist",
        },
        inplace=True,
    )
    return grouped


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
                df["track_id"] = df.get("spotify_track_uri").apply(extract_track_id)
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
            if hasattr(con, "unregister"):
                with contextlib.suppress(AttributeError, KeyError):
                    con.unregister("df_play_counts")
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
                df["track_id"] = df.get("spotify_track_uri").apply(extract_track_id)
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
            if hasattr(con, "unregister"):
                with contextlib.suppress(AttributeError, KeyError):
                    con.unregister("df_play_counts")
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


def get_most_played_artists(
    filtered_df: pd.DataFrame,
    *,
    con: Any | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Most played artists with unique track counts using DuckDB when available.

    Returns columns: 'artist', 'artist_id', 'artist_genres', 'play_count',
    'unique_tracks', 'percentage' (as percent with 2 decimals to match prior behavior).
    """
    if filtered_df.empty:
        return pd.DataFrame(
            columns=[
                "artist",
                "artist_id",
                "artist_genres",
                "play_count",
                "unique_tracks",
                "percentage",
            ]
        )

    if con is not None:
        df = filtered_df[
            [
                "master_metadata_album_artist_name",
                "master_metadata_track_name",
                "artist_id",
            ]
        ].copy()

        rel = "df_artist_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        lim = f"LIMIT {int(limit)}" if (limit is not None and limit > 0) else ""
        sql = f"""
            WITH stats AS (
                SELECT
                    master_metadata_album_artist_name AS artist,
                    COALESCE(artist_id, '') AS artist_id,
                    COUNT(*) AS play_count,
                    COUNT(DISTINCT master_metadata_track_name) AS unique_tracks
                FROM {rel}
                GROUP BY 1,2
            ), totals AS (
                SELECT SUM(play_count) AS total_plays FROM stats
            )
            SELECT s.artist, s.artist_id, s.play_count, s.unique_tracks,
                   ROUND((s.play_count::DOUBLE / NULLIF(t.total_plays, 0)) * 100, 2) AS percentage
            FROM stats s CROSS JOIN totals t
            ORDER BY s.play_count DESC, s.artist
            {lim}
        """
        res = con.execute(sql).df()
        # Reattach artist_genres from filtered_df
        try:
            if "artist_genres" in filtered_df.columns and not res.empty:
                # Coerce artist_id to non-null strings for consistent matching
                res["artist_id"] = res["artist_id"].fillna("").astype(str)
                m = (
                    filtered_df[
                        [
                            "master_metadata_album_artist_name",
                            "artist_id",
                            "artist_genres",
                        ]
                    ]
                    .drop_duplicates(subset=["master_metadata_album_artist_name", "artist_id"])
                    .rename(columns={"master_metadata_album_artist_name": "artist"})
                )
                m["artist_id"] = m["artist_id"].fillna("").astype(str)
                res = res.merge(m, on=["artist", "artist_id"], how="left")
                # Guarantee artist_genres exists with empty lists where missing
                if "artist_genres" not in res.columns:
                    res["artist_genres"] = [[] for _ in range(len(res))]
                else:
                    res["artist_genres"] = res["artist_genres"].fillna([])
        except Exception:
            if "artist_genres" not in res.columns:
                res["artist_genres"] = [[] for _ in range(len(res))]
            else:
                res["artist_genres"] = res["artist_genres"].fillna([])
        if not res.empty:
            res["percentage"] = res["percentage"].fillna(0.0)
        return res

    # Fallback to pandas
    df = filtered_df.copy()
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
    stats = pd.merge(
        plays,
        uniques,
        on=["master_metadata_album_artist_name", "artist_id", "artist_genres"],
    )
    stats = stats.sort_values("play_count", ascending=False)
    total_plays = stats["play_count"].sum() or 1
    if limit is not None and limit > 0:
        stats = stats.head(limit)
    stats["percentage"] = (stats["play_count"] / total_plays * 100).round(2)
    stats.rename(columns={"master_metadata_album_artist_name": "artist"}, inplace=True)
    return stats


def get_playcount_by_day(
    filtered_df: pd.DataFrame,
    *,
    con: Any | None = None,
    top_only: bool = True,
) -> pd.DataFrame:
    """Daily play counts (optionally top track per day) using DuckDB.

    - When `top_only=True`, returns exactly one row per day: the top track by plays.
    - Falls back to pandas grouping when no connection is provided.
    """
    if filtered_df.empty:
        return pd.DataFrame(columns=["date", "track", "artist", "play_count"])

    if con is not None:
        df = filtered_df[
            ["ts", "master_metadata_track_name", "master_metadata_album_artist_name"]
        ].copy()
        rel = "df_daily_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        if top_only:
            sql = f"""
                WITH daily AS (
                    SELECT CAST(ts AS DATE) AS date,
                           master_metadata_track_name AS track,
                           master_metadata_album_artist_name AS artist,
                           COUNT(*) AS play_count
                    FROM {rel}
                    GROUP BY 1,2,3
                ), ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY date
                               ORDER BY play_count DESC, track, artist
                           ) AS rn
                    FROM daily
                )
                SELECT date, track, artist, play_count
                FROM ranked
                WHERE rn = 1
                ORDER BY date ASC
            """
        else:
            sql = f"""
                SELECT CAST(ts AS DATE) AS date,
                       master_metadata_track_name AS track,
                       master_metadata_album_artist_name AS artist,
                       COUNT(*) AS play_count
                FROM {rel}
                GROUP BY 1,2,3
                ORDER BY date ASC, play_count DESC
            """
        return con.execute(sql).df()

    # Fallback to pandas path
    df = filtered_df.copy()
    df["date"] = df["ts"].dt.date
    daily_counts = (
        df.groupby(["date", "master_metadata_track_name", "master_metadata_album_artist_name"])
        .size()
        .reset_index(name="play_count")
    )
    daily_counts.rename(
        columns={
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
        },
        inplace=True,
    )
    daily_counts = daily_counts.sort_values(["date", "play_count"], ascending=[True, False])
    if top_only:
        # pick the first row per date (highest play_count after sorting)
        daily_counts = daily_counts.groupby("date").head(1)
    return daily_counts
