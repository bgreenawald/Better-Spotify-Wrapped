import contextlib
from pathlib import Path
from typing import Any

import pandas as pd


def get_listening_time_by_month(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total listening time by month.

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'ms_played', 'master_metadata_track_name', and
            'master_metadata_album_artist_name' columns.

    Returns:
        pd.DataFrame: Monthly listening statistics with columns:
            - month (str): Year-month in 'YYYY-MM' format
            - unique_tracks (int): Number of unique tracks
            - unique_artists (int): Number of unique artists
            - total_hours (float): Total listening time in hours
            - avg_hours_per_day (float): Average listening time per day
    """
    # Group by month string and compute total playtime and unique counts
    monthly = (
        filtered_df.groupby(filtered_df["ts"].dt.strftime("%Y-%m"))
        .agg(
            {
                "ms_played": "sum",
                "master_metadata_track_name": "nunique",
                "master_metadata_album_artist_name": "nunique",
            }
        )
        .reset_index()
    )

    # Convert milliseconds to hours
    monthly["total_hours"] = monthly["ms_played"] / (1000 * 60 * 60)

    # Rename for clarity
    monthly.columns = [
        "month",
        "ms_played",
        "unique_tracks",
        "unique_artists",
        "total_hours",
    ]

    # Compute days in each month and average hours per day
    monthly["days_in_month"] = monthly["month"].apply(lambda m: pd.Period(m).days_in_month)
    monthly["avg_hours_per_day"] = monthly["total_hours"] / monthly["days_in_month"]

    # Round numeric columns
    monthly["total_hours"] = monthly["total_hours"].round(2)
    monthly["avg_hours_per_day"] = monthly["avg_hours_per_day"].round(2)

    # Drop intermediates, sort, and return
    result = monthly.drop(["ms_played", "days_in_month"], axis=1)
    result = result.sort_values("month").reset_index(drop=True)
    return result


def get_genre_trends(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
) -> pd.DataFrame:
    """Calculate genre listening trends over time.

    DuckDB-backed implementation using the normalized schema (see DDL.sql).

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must contain
            'ts' and 'spotify_track_uri' columns.
        db_path: Optional DuckDB database path (if `con` not provided).
        con: Optional DuckDB connection to use.

    Returns:
        pd.DataFrame: Genre trends with columns:
            - month (str): 'YYYY-MM'
            - genre (str)
            - play_count (int)
            - percentage (float)
            - top_artists (str)
            - rank (float): Rank of genre by plays within month
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
        if ":" in uri:
            return uri.split(":")[-1]
        return None

    use_duckdb = (con is not None) or (db_path is not None)
    if not use_duckdb:
        raise ValueError("get_genre_trends requires a DuckDB connection or db_path.")

    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("DuckDB backend requested, but duckdb is not installed.") from exc

    close_conn = False
    if con is None:
        con = duckdb.connect(str(db_path))
        close_conn = True

    try:
        df = filtered_df.copy()
        df["month"] = df["ts"].dt.strftime("%Y-%m")
        df["track_id"] = df.get("spotify_track_uri").apply(_extract_track_id)
        plays = (
            df.dropna(subset=["track_id"])  # type: ignore[arg-type]
            .groupby(["month", "track_id"])  # type: ignore[list-item]
            .size()
            .reset_index(name="play_count")
        )

        if plays.empty:
            return pd.DataFrame(
                columns=["month", "genre", "play_count", "percentage", "top_artists", "rank"]
            )

        with contextlib.suppress(Exception):
            con.unregister("df_plays")  # type: ignore[attr-defined]
        con.register("df_plays", plays)

        # Aggregate per-genre per-month plays via primary artist
        sql_genre_month = """
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
            ),
            genre_artist_month AS (
                SELECT p.month,
                       g.name AS genre,
                       a.artist_name AS artist,
                       SUM(p.play_count) AS artist_plays
                FROM df_plays p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN dim_artists a ON a.artist_id = pa.artist_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                GROUP BY 1, 2, 3
            ),
            genre_month AS (
                SELECT month, genre, SUM(artist_plays) AS play_count
                FROM genre_artist_month
                GROUP BY 1, 2
            ),
            monthly_totals AS (
                SELECT month, SUM(play_count) AS total_plays
                FROM genre_month
                GROUP BY 1
            )
            SELECT gm.month,
                   gm.genre,
                   gm.play_count,
                   mt.total_plays
            FROM genre_month gm
            JOIN monthly_totals mt USING(month)
            ORDER BY gm.month, gm.play_count DESC, gm.genre
        """
        genre_month_df = con.execute(sql_genre_month).df()

        if genre_month_df.empty:
            return pd.DataFrame(
                columns=["month", "genre", "play_count", "percentage", "top_artists", "rank"]
            )

        # Top artists per month-genre
        sql_genre_artist = """
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
            ),
            genre_artist_month AS (
                SELECT p.month,
                       g.name AS genre,
                       a.artist_name AS artist,
                       SUM(p.play_count) AS artist_plays
                FROM df_plays p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN dim_artists a ON a.artist_id = pa.artist_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                GROUP BY 1, 2, 3
            )
            SELECT month, genre, artist, artist_plays
            FROM genre_artist_month
        """
        genre_artist_df = con.execute(sql_genre_artist).df()

        out = genre_month_df.copy()
        out["percentage"] = (out["play_count"] / out["total_plays"] * 100).round(2)
        out = out.drop(columns=["total_plays"])

        def format_top(month: str, genre: str) -> str:
            sub = genre_artist_df[
                (genre_artist_df["month"] == month) & (genre_artist_df["genre"] == genre)
            ]
            if sub.empty:
                return ""
            sub = sub.sort_values(["artist_plays", "artist"], ascending=[False, True]).head(2)
            return ", ".join(f"{r.artist} ({int(r.artist_plays)} plays)" for _, r in sub.iterrows())

        out["top_artists"] = [
            format_top(m, g) for m, g in zip(out["month"], out["genre"], strict=False)
        ]
        out = out.sort_values(["month", "play_count"], ascending=[True, False])
        out["rank"] = out.groupby("month")["play_count"].rank(method="dense", ascending=False)
        return out.reset_index(drop=True)
    finally:
        if close_conn:
            con.close()


def get_artist_trends(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate artist listening trends over time.

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'master_metadata_album_artist_name',
            'master_metadata_track_name', and 'ms_played' columns.

    Returns:
        pd.DataFrame: Artist trends with columns:
            - month (str)
            - artist (str)
            - play_count (int)
            - unique_tracks (int)
            - percentage (float)
            - avg_duration_min (float)
            - top_tracks (str)
            - rank (float)
    """
    df = filtered_df.copy()
    df["month"] = df["ts"].dt.strftime("%Y-%m")

    # Count plays per track per artist-month
    track_counts = (
        df.groupby(["month", "master_metadata_album_artist_name", "master_metadata_track_name"])
        .size()
        .reset_index(name="track_plays")
    )

    # Rank and select top 2 tracks per artist-month
    track_counts["track_rank"] = track_counts.groupby(
        ["month", "master_metadata_album_artist_name"]
    )["track_plays"].rank(method="dense", ascending=False)
    top_tracks_df = track_counts[track_counts["track_rank"] <= 2]

    # Aggregate top tracks strings
    top_tracks_by_artist = (
        top_tracks_df.sort_values("track_plays", ascending=False)
        .groupby(["month", "master_metadata_album_artist_name"])
        .apply(
            lambda grp: ", ".join(
                f"{row['master_metadata_track_name']} ({row['track_plays']} plays)"
                for _, row in grp.head(2).iterrows()
            )
        )
        .reset_index(name="top_tracks")
    )

    # Compute artist metrics: play count, unique tracks, avg duration
    metrics = df.groupby(["month", "master_metadata_album_artist_name"]).agg(
        {
            "master_metadata_track_name": ["count", "nunique"],
            "ms_played": "mean",
        }
    )
    metrics.columns = ["play_count", "unique_tracks", "avg_duration_ms"]
    metrics = metrics.reset_index()

    # Compute monthly totals for percentage
    monthly_totals = metrics.groupby("month")["play_count"].sum().reset_index(name="total_plays")
    metrics = metrics.merge(monthly_totals, on="month")
    metrics["percentage"] = (metrics["play_count"] / metrics["total_plays"] * 100).round(2)

    # Convert duration to minutes
    metrics["avg_duration_min"] = (metrics["avg_duration_ms"] / (1000 * 60)).round(2)

    # Combine with top tracks
    result = metrics.merge(
        top_tracks_by_artist,
        on=["month", "master_metadata_album_artist_name"],
        how="left",
    )

    # Clean up and rename
    result = result.drop(["avg_duration_ms", "total_plays"], axis=1)
    result = result.rename(columns={"master_metadata_album_artist_name": "artist"})

    # Sort and rank
    result = result.sort_values(["month", "play_count"], ascending=[True, False])
    result["rank"] = result.groupby("month")["play_count"].rank(method="dense", ascending=False)

    return result.reset_index(drop=True)


def get_track_trends(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate track listening trends over time.

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'master_metadata_track_name',
            'master_metadata_album_artist_name', and 'ms_played' columns.

    Returns:
        pd.DataFrame: Track trends with columns:
            - month (str)
            - track (str)
            - artist (str)
            - track_artist (str)
            - play_count (int)
            - percentage (float)
            - avg_duration_min (float)
            - rank (float)
    """
    df = filtered_df.copy()
    df["month"] = df["ts"].dt.strftime("%Y-%m")

    # Combine track name and artist
    df["track_artist"] = (
        df["master_metadata_track_name"] + " - " + df["master_metadata_album_artist_name"]
    )

    # Compute play count and avg duration per track-artist-month
    track_metrics = df.groupby(
        [
            "month",
            "track_artist",
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
        ]
    ).agg(
        {
            "track_artist": "count",
            "ms_played": "mean",
        }
    )
    track_metrics.columns = ["play_count", "avg_duration_ms"]
    track_metrics = track_metrics.reset_index()

    # Monthly totals for percentage
    monthly_totals = (
        track_metrics.groupby("month")["play_count"].sum().reset_index(name="total_plays")
    )
    track_metrics = track_metrics.merge(monthly_totals, on="month")
    track_metrics["percentage"] = (
        track_metrics["play_count"] / track_metrics["total_plays"] * 100
    ).round(2)

    # Convert duration to minutes
    track_metrics["avg_duration_min"] = (track_metrics["avg_duration_ms"] / (1000 * 60)).round(2)

    # Clean up and rename columns
    result = track_metrics.drop(["avg_duration_ms", "total_plays"], axis=1)
    result = result.rename(
        columns={
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
        }
    )

    # Sort and rank
    result = result.sort_values(["month", "play_count"], ascending=[True, False])
    result["rank"] = result.groupby("month")["play_count"].rank(method="dense", ascending=False)

    return result.reset_index(drop=True)
