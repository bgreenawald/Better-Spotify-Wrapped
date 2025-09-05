import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import extract_track_id


def get_listening_time_by_month(
    filtered_df: pd.DataFrame,
    *,
    con: Any | None = None,
) -> pd.DataFrame:
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
    if con is not None and not filtered_df.empty:
        df = filtered_df[
            [
                "ts",
                "ms_played",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
            ]
        ].copy()
        rel = "df_monthly_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        sql = f"""
            WITH base AS (
                SELECT strftime(ts, '%Y-%m') AS month,
                       ms_played,
                       master_metadata_track_name AS track,
                       master_metadata_album_artist_name AS artist
                FROM {rel}
            ), monthly AS (
                SELECT month,
                       SUM(ms_played) AS ms_played,
                       COUNT(DISTINCT track) AS unique_tracks,
                       COUNT(DISTINCT artist) AS unique_artists
                FROM base
                GROUP BY 1
            ), days AS (
                SELECT month,
                       CAST(EXTRACT(day FROM (date_trunc('month', strptime(month || '-01', '%Y-%m-%d'))
                              + INTERVAL 1 MONTH - INTERVAL 1 DAY)) AS INTEGER) AS days_in_month
                FROM monthly
            )
            SELECT m.month,
                   m.unique_tracks,
                   m.unique_artists,
                   ROUND(m.ms_played / (1000.0 * 60.0 * 60.0), 2) AS total_hours,
                   ROUND((m.ms_played / (1000.0 * 60.0 * 60.0)) / NULLIF(d.days_in_month, 0), 2) AS avg_hours_per_day
            FROM monthly m
            JOIN days d USING (month)
            ORDER BY m.month
        """
        return con.execute(sql).df().reset_index(drop=True)

    # Fallback to pandas
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
    monthly["total_hours"] = monthly["ms_played"] / (1000 * 60 * 60)
    monthly.columns = [
        "month",
        "ms_played",
        "unique_tracks",
        "unique_artists",
        "total_hours",
    ]
    monthly["days_in_month"] = monthly["month"].apply(lambda m: pd.Period(m).days_in_month)
    monthly["avg_hours_per_day"] = monthly["total_hours"] / monthly["days_in_month"]
    monthly["total_hours"] = monthly["total_hours"].round(2)
    monthly["avg_hours_per_day"] = monthly["avg_hours_per_day"].round(2)
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
        # Prefer existing track_id to avoid repeated URI parsing
        if "track_id" in df.columns:
            tid = df["track_id"].astype("string")
        elif "spotify_track_uri" in df.columns:
            tid = df.get("spotify_track_uri").apply(extract_track_id).astype("string")
        else:
            tid = pd.Series([pd.NA] * len(df), dtype="string")
        df["track_id"] = tid
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
        # One-pass SQL: compute per-month genre plays, percentage, rank, and top artists
        sql = """
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
            ),
            artist_ranked AS (
                SELECT month, genre, artist, artist_plays,
                       ROW_NUMBER() OVER (PARTITION BY month, genre ORDER BY artist_plays DESC, artist) AS rn
                FROM genre_artist_month
            ),
            top2 AS (
                SELECT month, genre,
                       STRING_AGG(artist || ' (' || CAST(artist_plays AS BIGINT) || ' plays)', ', ' ORDER BY rn) AS top_artists
                FROM artist_ranked
                WHERE rn <= 2
                GROUP BY 1,2
            )
            SELECT gm.month,
                   gm.genre,
                   CAST(gm.play_count AS BIGINT) AS play_count,
                   ROUND((gm.play_count::DOUBLE / NULLIF(mt.total_plays, 0)) * 100, 2) AS percentage,
                   COALESCE(t2.top_artists, '') AS top_artists,
                   DENSE_RANK() OVER (PARTITION BY gm.month ORDER BY gm.play_count DESC, gm.genre) AS rank
            FROM genre_month gm
            JOIN monthly_totals mt USING (month)
            LEFT JOIN top2 t2 ON t2.month = gm.month AND t2.genre = gm.genre
            ORDER BY gm.month, gm.play_count DESC, gm.genre
        """
        out = con.execute(sql).df()
        if out.empty:
            return pd.DataFrame(
                columns=["month", "genre", "play_count", "percentage", "top_artists", "rank"]
            )
        return out.reset_index(drop=True)
    finally:
        if close_conn:
            con.close()


def get_artist_trends(
    filtered_df: pd.DataFrame,
    *,
    con: Any | None = None,
) -> pd.DataFrame:
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
    if con is not None and not filtered_df.empty:
        df = filtered_df[
            [
                "ts",
                "master_metadata_album_artist_name",
                "master_metadata_track_name",
                "ms_played",
            ]
        ].copy()

        rel = "df_artist_trends_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        sql_metrics = f"""
            WITH base AS (
                SELECT strftime(ts, '%Y-%m') AS month,
                       master_metadata_album_artist_name AS artist,
                       master_metadata_track_name AS track,
                       ms_played
                FROM {rel}
            ), metrics AS (
                SELECT month,
                       artist,
                       COUNT(*) AS play_count,
                       COUNT(DISTINCT track) AS unique_tracks,
                       AVG(ms_played) AS avg_duration_ms
                FROM base
                GROUP BY 1,2
            ), monthly_totals AS (
                SELECT month, SUM(play_count) AS total_plays FROM metrics GROUP BY 1
            )
            SELECT m.month,
                   m.artist,
                   m.play_count,
                   m.unique_tracks,
                   ROUND((m.play_count::DOUBLE / NULLIF(t.total_plays, 0)) * 100, 2) AS percentage,
                   (m.avg_duration_ms / (1000.0 * 60.0)) AS avg_duration_min
            FROM metrics m
            JOIN monthly_totals t USING (month)
            ORDER BY m.month, m.play_count DESC, m.artist
        """
        metrics_df = con.execute(sql_metrics).df()

        sql_top2 = f"""
            WITH base AS (
                SELECT strftime(ts, '%Y-%m') AS month,
                       master_metadata_album_artist_name AS artist,
                       master_metadata_track_name AS track
                FROM {rel}
            ), track_counts AS (
                SELECT month, artist, track, COUNT(*) AS track_plays
                FROM base
                GROUP BY 1,2,3
            ), ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY month, artist
                           ORDER BY track_plays DESC, track
                       ) AS rn
                FROM track_counts
            )
            SELECT month, artist, track, track_plays
            FROM ranked
            WHERE rn <= 2
            ORDER BY month, artist, track_plays DESC, track
        """
        top2_df = con.execute(sql_top2).df()

        if not top2_df.empty:
            top_tracks = (
                top2_df.groupby(["month", "artist"])
                .apply(
                    lambda grp: ", ".join(
                        f"{row['track']} ({int(row['track_plays'])} plays)"
                        for _, row in grp.iterrows()
                    )
                )
                .reset_index(name="top_tracks")
            )
            out = metrics_df.merge(top_tracks, on=["month", "artist"], how="left")
        else:
            out = metrics_df.copy()
            out["top_tracks"] = ""

        out["rank"] = out.groupby("month")["play_count"].rank(method="dense", ascending=False)
        return out.reset_index(drop=True)

    # Fallback to pandas implementation
    df = filtered_df.copy()
    df["month"] = df["ts"].dt.strftime("%Y-%m")
    track_counts = (
        df.groupby(["month", "master_metadata_album_artist_name", "master_metadata_track_name"])
        .size()
        .reset_index(name="track_plays")
    )
    track_counts["track_rank"] = track_counts.groupby(
        ["month", "master_metadata_album_artist_name"]
    )["track_plays"].rank(method="dense", ascending=False)
    top_tracks_df = track_counts[track_counts["track_rank"] <= 2]
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
    metrics = df.groupby(["month", "master_metadata_album_artist_name"]).agg(
        {"master_metadata_track_name": ["count", "nunique"], "ms_played": "mean"}
    )
    metrics.columns = ["play_count", "unique_tracks", "avg_duration_ms"]
    metrics = metrics.reset_index()
    monthly_totals = metrics.groupby("month")["play_count"].sum().reset_index(name="total_plays")
    metrics = metrics.merge(monthly_totals, on="month")
    metrics["percentage"] = (metrics["play_count"] / metrics["total_plays"] * 100).round(2)
    metrics["avg_duration_min"] = (metrics["avg_duration_ms"] / (1000 * 60)).round(2)
    result = metrics.merge(
        top_tracks_by_artist, on=["month", "master_metadata_album_artist_name"], how="left"
    )
    result = result.drop(["avg_duration_ms", "total_plays"], axis=1)
    result = result.rename(columns={"master_metadata_album_artist_name": "artist"})
    result = result.sort_values(["month", "play_count"], ascending=[True, False])
    result["rank"] = result.groupby("month")["play_count"].rank(method="dense", ascending=False)
    return result.reset_index(drop=True)


def get_track_trends(
    filtered_df: pd.DataFrame,
    *,
    con: Any | None = None,
) -> pd.DataFrame:
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
    if con is not None and not filtered_df.empty:
        df = filtered_df[
            [
                "ts",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
                "ms_played",
            ]
        ].copy()
        rel = "df_track_trends_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        sql = f"""
            WITH base AS (
                SELECT strftime(ts, '%Y-%m') AS month,
                       master_metadata_track_name AS track,
                       master_metadata_album_artist_name AS artist,
                       (master_metadata_track_name || ' - ' || master_metadata_album_artist_name) AS track_artist,
                       ms_played
                FROM {rel}
            ), metrics AS (
                SELECT month, track, artist, track_artist,
                       COUNT(*) AS play_count,
                       AVG(ms_played) AS avg_duration_ms
                FROM base
                GROUP BY 1,2,3,4
            ), totals AS (
                SELECT month, SUM(play_count) AS total_plays FROM metrics GROUP BY 1
            )
            SELECT m.month, m.track, m.artist, m.track_artist,
                   m.play_count,
                   ROUND((m.play_count::DOUBLE / NULLIF(t.total_plays, 0)) * 100, 2) AS percentage,
                   (m.avg_duration_ms / (1000.0 * 60.0)) AS avg_duration_min,
                   DENSE_RANK() OVER (PARTITION BY m.month ORDER BY m.play_count DESC, m.track_artist) AS rank
            FROM metrics m
            JOIN totals t USING (month)
            ORDER BY m.month, m.play_count DESC, m.track_artist
        """
        return con.execute(sql).df().reset_index(drop=True)

    # Fallback to pandas implementation
    df = filtered_df.copy()
    df["month"] = df["ts"].dt.strftime("%Y-%m")
    df["track_artist"] = (
        df["master_metadata_track_name"] + " - " + df["master_metadata_album_artist_name"]
    )
    track_metrics = df.groupby(
        [
            "month",
            "track_artist",
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
        ]
    ).agg(
        play_count=pd.NamedAgg(column="track_artist", aggfunc="size"),
        avg_duration_ms=pd.NamedAgg(column="ms_played", aggfunc="mean"),
    )
    track_metrics = track_metrics.reset_index()
    monthly_totals = (
        track_metrics.groupby("month")["play_count"].sum().reset_index(name="total_plays")
    )
    track_metrics = track_metrics.merge(monthly_totals, on="month")
    track_metrics["percentage"] = (
        track_metrics["play_count"] / track_metrics["total_plays"] * 100
    ).round(2)
    track_metrics["avg_duration_min"] = (track_metrics["avg_duration_ms"] / (1000 * 60)).round(2)
    result = track_metrics.drop(["avg_duration_ms", "total_plays"], axis=1)
    result = result.rename(
        columns={
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
        }
    )
    result = result.sort_values(["month", "play_count"], ascending=[True, False])
    result["rank"] = result.groupby("month")["play_count"].rank(method="dense", ascending=False)
    return result.reset_index(drop=True)
