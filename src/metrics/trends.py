import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.albums import gini_coefficient, normalized_entropy, shannon_entropy
from src.metrics.utils import column_exists

from .utils import extract_track_id


def get_listening_time_by_month(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
) -> pd.DataFrame:
    """Calculate total listening time by month (DuckDB-only).

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'ms_played', 'master_metadata_track_name', and
            'master_metadata_album_artist_name' columns.
        db_path (str | Path | None): Optional DuckDB database path if `con` not provided.
        con (Any | None): DuckDB connection to use.

    Returns:
        pd.DataFrame: Monthly listening statistics with columns:
            - month (str): Year-month in 'YYYY-MM' format
            - unique_tracks (int): Number of unique tracks
            - unique_artists (int): Number of unique artists
            - total_hours (float): Total listening time in hours
            - avg_hours_per_day (float): Average listening time per day
    """
    # Empty input → empty output with stable columns
    if filtered_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "unique_tracks",
                "unique_artists",
                "total_hours",
                "avg_hours_per_day",
            ]
        )

    # Require DuckDB backend
    if con is None and db_path is None:
        raise ValueError("get_listening_time_by_month requires a DuckDB connection or db_path.")

    close_conn = False
    if con is None:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path))
        close_conn = True

    try:
        mcols = [
            "ts",
            "ms_played",
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
        ]
        for extra in ("track_id", "artist_id"):
            if extra in filtered_df.columns:
                mcols.append(extra)
        df = filtered_df[mcols].copy()
        for extra in ("track_id", "artist_id"):
            if extra not in df.columns:
                df[extra] = None
        rel = "df_monthly_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        has_tracks = column_exists(con, "dim_tracks", "track_isrc")
        isrc_join = "LEFT JOIN dim_tracks dt ON dt.track_id = p.track_id" if has_tracks else ""
        recording_key = (
            "COALESCE(NULLIF(TRIM(dt.track_isrc), ''), p.track_id, p.master_metadata_track_name)"
            if has_tracks
            else "COALESCE(p.track_id, p.master_metadata_track_name)"
        )
        # dim_tracks.duration_ms is the track's length; fact_plays.duration_ms
        # (aliased ms_played here) is how much of it was actually played.
        track_length = "dt.duration_ms" if has_tracks else "NULL"
        sql = f"""
            WITH base AS (
                -- Counted by identity, not by display name: distinct
                -- recordings (ISRC, falling back to track_id) and distinct
                -- artist ids. Counting distinct names merged two different
                -- songs sharing a title and split an artist who was renamed.
                SELECT strftime(p.ts, '%Y-%m') AS month,
                       p.ms_played,
                       {recording_key} AS recording_key,
                       {track_length} AS track_length_ms,
                       COALESCE(NULLIF(TRIM(p.artist_id), ''),
                                p.master_metadata_album_artist_name) AS artist_key
                FROM {rel} p
                {isrc_join}
            ), monthly AS (
                SELECT month,
                       SUM(ms_played) AS ms_played,
                       COUNT(DISTINCT recording_key) AS unique_tracks,
                       COUNT(DISTINCT artist_key) AS unique_artists,
                       -- Plays are unweighted by duration everywhere else, so
                       -- report how complete they were: listened ms over the
                       -- track's own length. A month of half-finished tracks
                       -- otherwise looks the same as a month of full ones.
                       AVG(
                           CASE
                               WHEN COALESCE(track_length_ms, 0) > 0
                               THEN LEAST(ms_played::DOUBLE / track_length_ms, 1.0)
                           END
                       ) AS avg_completion,
                       COUNT(*) AS plays
                FROM base
                GROUP BY 1
            ), bounds AS (
                SELECT MIN(CAST(ts AS DATE)) AS first_day,
                       MAX(CAST(ts AS DATE)) AS last_day
                FROM {rel}
            ), days AS (
                -- Days of this month that actually fall inside the observed
                -- range. Dividing by the full calendar month made the first
                -- and current months read low, since only part of them is
                -- covered by the data.
                SELECT m.month,
                       CAST(DATE_DIFF('day',
                           GREATEST(
                               CAST(strptime(m.month || '-01', '%Y-%m-%d') AS DATE),
                               b.first_day
                           ),
                           LEAST(
                               CAST(strptime(m.month || '-01', '%Y-%m-%d') AS DATE)
                                   + INTERVAL 1 MONTH - INTERVAL 1 DAY,
                               b.last_day
                           )
                       ) + 1 AS INTEGER) AS days_observed
                FROM monthly m CROSS JOIN bounds b
            )
            SELECT m.month,
                   m.unique_tracks,
                   m.unique_artists,
                   m.plays,
                   ROUND(100.0 * m.avg_completion, 1) AS avg_completion_pct,
                   ROUND(m.ms_played / (1000.0 * 60.0 * 60.0), 2) AS total_hours,
                   ROUND((m.ms_played / (1000.0 * 60.0 * 60.0)) / NULLIF(d.days_observed, 0), 2) AS avg_hours_per_day
            FROM monthly m
            JOIN days d USING (month)
            ORDER BY m.month
        """
        return con.execute(sql).df().reset_index(drop=True)
    finally:
        if close_conn:
            con.close()


def get_genre_trends(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
    min_month_plays: int = 0,
) -> pd.DataFrame:
    """Calculate genre listening trends over time (DuckDB-only).

    `min_month_plays` suppresses months too thin to rank meaningfully: with no
    threshold a month holding three plays can crown a genre #1 alongside a
    month of six hundred. The returned `month_plays` column carries each
    month's total so a caller can mark thin months instead of dropping them.

    Uses the normalized schema (see DDL.sql).

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
                columns=[
                    "month",
                    "genre",
                    "month_plays",
                    "play_count",
                    "percentage",
                    "top_artists",
                    "rank",
                ]
            )

        with contextlib.suppress(Exception):
            con.unregister("df_plays")  # type: ignore[attr-defined]
        con.register("df_plays", plays)
        # One-pass SQL: compute per-month genre plays, percentage, rank, and top artists
        sql = """
            WITH primary_artist AS (
                SELECT track_id, artist_id
                FROM v_primary_artist_per_track
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
                   CAST(mt.total_plays AS BIGINT) AS month_plays,
                   CAST(gm.play_count AS BIGINT) AS play_count,
                   ROUND((gm.play_count::DOUBLE / NULLIF(mt.total_plays, 0)) * 100, 2) AS percentage,
                   COALESCE(t2.top_artists, '') AS top_artists,
                   -- Dense rank with no tiebreaker in the window, so genuinely
                   -- tied entities share a rank. Adding a unique tiebreaker here
                   -- made DENSE_RANK behave exactly like ROW_NUMBER, so ties
                   -- never shared a rank and the order was alphabetical.
                   DENSE_RANK() OVER (PARTITION BY gm.month ORDER BY gm.play_count DESC) AS rank
            FROM genre_month gm
            JOIN monthly_totals mt USING (month)
            LEFT JOIN top2 t2 ON t2.month = gm.month AND t2.genre = gm.genre
            ORDER BY gm.month, gm.play_count DESC, gm.genre
        """
        out = con.execute(sql).df()
        if not out.empty and min_month_plays > 0:
            out = out[out["month_plays"] >= int(min_month_plays)].reset_index(drop=True)
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "month",
                    "genre",
                    "month_plays",
                    "play_count",
                    "percentage",
                    "top_artists",
                    "rank",
                ]
            )
        return out.reset_index(drop=True)
    finally:
        if close_conn:
            con.close()


def get_artist_trends(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
) -> pd.DataFrame:
    """Calculate artist listening trends over time (DuckDB-only).

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'master_metadata_album_artist_name',
            'master_metadata_track_name', and 'ms_played' columns.
        db_path (str | Path | None): Optional DuckDB database path if `con` not provided.
        con (Any | None): DuckDB connection to use.

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
    # Empty input → empty output
    if filtered_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "artist",
                "play_count",
                "unique_tracks",
                "percentage",
                "avg_duration_min",
                "top_tracks",
                "rank",
            ]
        )

    if con is None and db_path is None:
        raise ValueError("get_artist_trends requires a DuckDB connection or db_path.")

    close_conn = False
    if con is None:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path))
        close_conn = True

    try:
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
                    ),
                    include_groups=False,
                )
                .reset_index(name="top_tracks")
            )
            out = metrics_df.merge(top_tracks, on=["month", "artist"], how="left")
        else:
            out = metrics_df.copy()
            out["top_tracks"] = ""

        out["rank"] = out.groupby("month")["play_count"].rank(method="dense", ascending=False)
        return out.reset_index(drop=True)
    finally:
        if close_conn:
            con.close()


def get_track_trends(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
) -> pd.DataFrame:
    """Calculate track listening trends over time (DuckDB-only).

    Args:
        filtered_df (pd.DataFrame): Filtered Spotify listening history. Must
            contain 'ts', 'master_metadata_track_name',
            'master_metadata_album_artist_name', and 'ms_played' columns.
        db_path (str | Path | None): Optional DuckDB database path if `con` not provided.
        con (Any | None): DuckDB connection to use.

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
    if filtered_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "track",
                "artist",
                "track_artist",
                "play_count",
                "percentage",
                "avg_duration_min",
                "rank",
            ]
        )

    if con is None and db_path is None:
        raise ValueError("get_track_trends requires a DuckDB connection or db_path.")

    close_conn = False
    if con is None:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path))
        close_conn = True

    try:
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
                   -- Dense rank with no tiebreaker in the window, so genuinely
                   -- tied entities share a rank. Adding a unique tiebreaker here
                   -- made DENSE_RANK behave exactly like ROW_NUMBER, so ties
                   -- never shared a rank and the order was alphabetical.
                   DENSE_RANK() OVER (PARTITION BY m.month ORDER BY m.play_count DESC) AS rank
            FROM metrics m
            JOIN totals t USING (month)
            ORDER BY m.month, m.play_count DESC, m.track_artist
        """
        return con.execute(sql).df().reset_index(drop=True)
    finally:
        if close_conn:
            con.close()


def get_listening_diversity_by_month(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Per-month diversity of listening across artists.

    Answers "did I range widely this month, or replay the same few artists?",
    which no existing metric reported. Entropy is in bits; `entropy_norm`
    divides by log2(n) so months with different catalogue sizes are
    comparable; `gini` is the concentration view, where high means a few
    artists took most of the plays; `top_artist_share` is the single largest
    artist's share of the month.

    Computed in pandas from the plays frame alone, so it needs no database
    connection.

    Args:
        filtered_df: Plays, as returned by `get_filtered_plays`.

    Returns:
        One row per month with columns 'month', 'unique_artists', 'plays',
        'entropy', 'entropy_norm', 'gini' and 'top_artist_share' (0-100).
    """
    columns = [
        "month",
        "unique_artists",
        "plays",
        "entropy",
        "entropy_norm",
        "gini",
        "top_artist_share",
    ]
    if filtered_df is None or filtered_df.empty:
        return pd.DataFrame(columns=columns)

    needed = {"ts", "master_metadata_album_artist_name"}
    if not needed.issubset(filtered_df.columns):
        return pd.DataFrame(columns=columns)

    df = filtered_df[["ts", "master_metadata_album_artist_name"]].copy()
    if "artist_id" in filtered_df.columns:
        df["artist_id"] = filtered_df["artist_id"]
    else:
        df["artist_id"] = None

    # Key by artist_id where available, matching how artists are counted
    # elsewhere, so a renamed artist is not two entities here either.
    df["artist_key"] = (
        df["artist_id"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA})
        .fillna(df["master_metadata_album_artist_name"].astype("string"))
    )
    df = df.dropna(subset=["ts", "artist_key"])
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["month"] = pd.to_datetime(df["ts"], errors="coerce").dt.strftime("%Y-%m")
    df = df.dropna(subset=["month"])

    rows: list[dict[str, Any]] = []
    for month, group in df.groupby("month", sort=True):
        counts = group["artist_key"].value_counts().tolist()
        plays = int(sum(counts))
        rows.append(
            {
                "month": str(month),
                "unique_artists": len(counts),
                "plays": plays,
                "entropy": round(shannon_entropy(counts), 4),
                "entropy_norm": round(normalized_entropy(counts), 4),
                "gini": round(gini_coefficient(counts), 4),
                "top_artist_share": round(100.0 * max(counts) / plays, 2) if plays else 0.0,
            }
        )

    return pd.DataFrame(rows, columns=columns)
