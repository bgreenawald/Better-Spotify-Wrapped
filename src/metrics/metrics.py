import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.utils import extract_track_id


def get_most_played_tracks(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Most played tracks using DuckDB.

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

    # Require DuckDB backend (either connection or db_path)
    if con is None and db_path is None:
        raise ValueError("get_most_played_tracks requires a DuckDB connection or db_path.")

    # Use DuckDB path
    close_conn = False
    if con is None:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path))
        close_conn = True

    try:
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
            # Cannot use fillna with non-scalar tuple/list; coerce per-row instead
            res["artist_genres"] = res["artist_genres"].apply(
                lambda v: v if isinstance(v, list | tuple) else ()
            )
        # Ensure stable types
        if not res.empty:
            res["percentage"] = res["percentage"].fillna(0.0)
        return res
    finally:
        if close_conn:
            con.close()


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

    # Use shared extractor from utils to normalize URIs

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

            # SQL plan (uses convenience views where helpful):
            # 1) Map played tracks to album_ids via v_plays_enriched
            # 2) Expand to all tracks for those albums (dim_tracks)
            # 3) Left join play counts to include zero-play tracks
            # 4) Compute per-album aggregates + pick a primary artist (mode over primary role)
            sql = """
                WITH played_tracks AS (
                    SELECT ve.album_id, p.track_id, p.play_count
                    FROM df_play_counts p
                    JOIN v_plays_enriched ve ON ve.track_id = p.track_id
                    WHERE ve.album_id IS NOT NULL
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
            # Prefer existing track_id to avoid repeated URI parsing
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

            # Compute per-genre totals and top artists string in a single SQL pass
            sql = f"""
            WITH primary_artist AS (
                SELECT track_id, artist_id
                FROM v_primary_artist_per_track
            ),
            genre_counts AS (
                SELECT g.name AS genre,
                       SUM(p.play_count) AS play_count
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                GROUP BY 1
            ),
            artist_genre_counts AS (
                SELECT g.name AS genre,
                       a.artist_name AS artist,
                       SUM(p.play_count) AS play_count
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN dim_artists a ON a.artist_id = pa.artist_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                GROUP BY 1, 2
                ),
                ranked AS (
                    SELECT genre, artist, play_count,
                           ROW_NUMBER() OVER (PARTITION BY genre ORDER BY play_count DESC, artist) AS rn
                    FROM artist_genre_counts
                ),
                top_str AS (
                    SELECT genre,
                           STRING_AGG(artist || ' (' || CAST(play_count AS BIGINT) || ' plays)', ', ' ORDER BY rn) AS top_artists
                    FROM ranked
                    WHERE rn <= {int(top_artists_per_genre)}
                    GROUP BY 1
                )
                SELECT gc.genre,
                       CAST(gc.play_count AS BIGINT) AS play_count,
                       COALESCE(ts.top_artists, '') AS top_artists
                FROM genre_counts gc
                LEFT JOIN top_str ts ON ts.genre = gc.genre
                ORDER BY gc.play_count DESC, gc.genre
            """
            out = con.execute(sql).df()

            if out.empty:
                return pd.DataFrame(columns=["genre", "play_count", "percentage", "top_artists"])

            # Add percentage column based on total_tracked (preserves original behavior)
            out["percentage"] = (out["play_count"] / total_tracked * 100).round(2)
            result = out[["genre", "play_count", "percentage", "top_artists"]]
            return result.sort_values(["play_count", "genre"], ascending=[False, True])
        finally:
            if close_conn:
                con.close()
    # If we reach here, neither `con` nor `db_path` was provided
    raise ValueError("get_top_artist_genres requires a DuckDB connection or db_path.")


def get_genre_sunburst_rows(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
) -> pd.DataFrame:
    """Return rows suitable for a two-level genre sunburst with top artists per wedge.

    Columns: 'parent' (level-0 name), 'child' (level-1 name or '<Parent> (direct)'),
    'value' (plays), 'top_artists' (top 2 artists as "Name (plays)" formatted).
    """
    if filtered_df.empty:
        return pd.DataFrame(columns=["parent", "child", "value", "top_artists"])

    use_duckdb = (con is not None) or (db_path is not None)
    if not use_duckdb:
        return pd.DataFrame(columns=["parent", "child", "value", "top_artists"])

    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("DuckDB backend requested, but duckdb is not installed.") from exc

    close_conn = False
    if con is None:
        con = duckdb.connect(str(db_path))
        close_conn = True
    try:
        df = filtered_df.copy()
        if "track_id" not in df.columns:
            df["track_id"] = df.get("spotify_track_uri").apply(extract_track_id)
        play_counts = (
            df.dropna(subset=["track_id"]).groupby("track_id").size().reset_index(name="play_count")
        )
        if play_counts.empty:
            return pd.DataFrame(columns=["parent", "child", "value", "top_artists"])

        if hasattr(con, "unregister"):
            with contextlib.suppress(AttributeError, KeyError):
                con.unregister("df_play_counts")
        con.register("df_play_counts", play_counts)

        sql = """
            WITH primary_artist AS (
                SELECT track_id, artist_id
                FROM v_primary_artist_per_track
            ),
            child_totals AS (
                SELECT cg.name AS child, SUM(p.play_count) AS value
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN genre_hierarchy gh ON gh.child_genre_id = ag.genre_id
                JOIN dim_genres cg ON cg.genre_id = gh.child_genre_id
                WHERE cg.level = 1
                GROUP BY 1
            ),
            child_artists AS (
                SELECT cg.name AS child, a.artist_name, SUM(p.play_count) AS play_count
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_artists a ON a.artist_id = pa.artist_id
                JOIN genre_hierarchy gh ON gh.child_genre_id = ag.genre_id
                JOIN dim_genres cg ON cg.genre_id = gh.child_genre_id
                WHERE cg.level = 1
                GROUP BY 1,2
            ),
            child_ranked AS (
                SELECT child, artist_name, play_count,
                       ROW_NUMBER() OVER (PARTITION BY child ORDER BY play_count DESC, artist_name) AS rn
                FROM child_artists
            ),
            child_top AS (
                SELECT child,
                       STRING_AGG(artist_name || ' (' || CAST(play_count AS BIGINT) || ' plays)', ', ' ORDER BY rn) AS top_artists
                FROM child_ranked
                WHERE rn <= 2
                GROUP BY 1
            ),
            sub AS (
                SELECT pg.name AS parent, cg.name AS child, t.value,
                       COALESCE(ct.top_artists, '') AS top_artists
                FROM genre_hierarchy gh
                JOIN dim_genres cg ON cg.genre_id = gh.child_genre_id
                JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id
                JOIN child_totals t ON t.child = cg.name
                LEFT JOIN child_top ct ON ct.child = cg.name
                WHERE cg.level = 1
            ),
            direct_totals AS (
                SELECT g.name AS parent, SUM(p.play_count) AS value
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                WHERE g.level = 0
                GROUP BY 1
            ),
            direct_artists AS (
                SELECT g.name AS parent, a.artist_name, SUM(p.play_count) AS play_count
                FROM df_play_counts p
                JOIN primary_artist pa ON pa.track_id = p.track_id
                JOIN artist_genres ag ON ag.artist_id = pa.artist_id
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                JOIN dim_artists a ON a.artist_id = pa.artist_id
                WHERE g.level = 0
                GROUP BY 1,2
            ),
            direct_ranked AS (
                SELECT parent, artist_name, play_count,
                       ROW_NUMBER() OVER (PARTITION BY parent ORDER BY play_count DESC, artist_name) AS rn
                FROM direct_artists
            ),
            direct_top AS (
                SELECT parent,
                       STRING_AGG(artist_name || ' (' || CAST(play_count AS BIGINT) || ' plays)', ', ' ORDER BY rn) AS top_artists
                FROM direct_ranked
                WHERE rn <= 2
                GROUP BY 1
            ),
            direct AS (
                SELECT d.parent, (d.parent || ' (direct)') AS child, d.value,
                       COALESCE(dt.top_artists, '') AS top_artists
                FROM direct_totals d
                LEFT JOIN direct_top dt ON dt.parent = d.parent
            )
            SELECT parent, child, value, top_artists FROM sub
            UNION ALL
            SELECT parent, child, value, top_artists FROM direct
        """
        out = con.execute(sql).df()
        # Fallback using track_genres if no children
        try:
            has_real_children = False
            if not out.empty:
                has_real_children = any(not str(c).endswith(" (direct)") for c in out["child"])
            if not has_real_children:
                alt_sql = """
                    WITH tg_counts AS (
                        SELECT g.genre_id, g.name AS genre_name, g.level, a.artist_name,
                               SUM(p.play_count * tg.score) AS play_count
                        FROM df_play_counts p
                        JOIN dim_tracks t ON t.track_id = p.track_id
                        JOIN track_genres tg ON tg.track_id = t.track_id
                        JOIN dim_genres g ON g.genre_id = tg.genre_id
                        JOIN bridge_track_artists b ON b.track_id = t.track_id AND b."role"='primary'
                        JOIN dim_artists a ON a.artist_id = b.artist_id
                        GROUP BY 1,2,3,4
                    ),
                    child_totals AS (
                        SELECT cg.name AS child, SUM(c.play_count) AS value
                        FROM tg_counts c
                        JOIN genre_hierarchy gh ON gh.child_genre_id = c.genre_id
                        JOIN dim_genres cg ON cg.genre_id = gh.child_genre_id
                        WHERE cg.level = 1
                        GROUP BY 1
                    ),
                    child_artists AS (
                        SELECT cg.name AS child, c.artist_name, SUM(c.play_count) AS play_count
                        FROM tg_counts c
                        JOIN genre_hierarchy gh ON gh.child_genre_id = c.genre_id
                        JOIN dim_genres cg ON cg.genre_id = gh.child_genre_id
                        WHERE cg.level = 1
                        GROUP BY 1,2
                    ),
                    child_ranked AS (
                        SELECT child, artist_name, play_count,
                               ROW_NUMBER() OVER (PARTITION BY child ORDER BY play_count DESC, artist_name) AS rn
                        FROM child_artists
                    ),
                    child_top AS (
                        SELECT child,
                               STRING_AGG(artist_name || ' (' || CAST(play_count AS BIGINT) || ' plays)', ', ' ORDER BY rn) AS top_artists
                        FROM child_ranked
                        WHERE rn <= 2
                        GROUP BY 1
                    ),
                    sub AS (
                        SELECT pg.name AS parent, cg.name AS child, t.value,
                               COALESCE(ct.top_artists, '') AS top_artists
                        FROM genre_hierarchy gh
                        JOIN dim_genres cg ON cg.genre_id = gh.child_genre_id
                        JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id
                        JOIN child_totals t ON t.child = cg.name
                        LEFT JOIN child_top ct ON ct.child = cg.name
                        WHERE cg.level = 1
                    ),
                    direct AS (
                        SELECT g.name AS parent, (g.name || ' (direct)') AS child, SUM(c.play_count) AS value,
                               '' AS top_artists
                        FROM tg_counts c
                        JOIN dim_genres g ON g.genre_id = c.genre_id
                        WHERE g.level = 0
                        GROUP BY 1,2
                    )
                    SELECT parent, child, value, top_artists FROM sub
                    UNION ALL
                    SELECT parent, child, value, top_artists FROM direct
                """
                alt = con.execute(alt_sql).df()
                if not alt.empty:
                    out = alt
        except Exception:
            pass
        return out
    finally:
        if close_conn:
            con.close()


def get_most_played_artists(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Most played artists with unique track counts using DuckDB.

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

    # Require DuckDB backend
    if con is None and db_path is None:
        raise ValueError("get_most_played_artists requires a DuckDB connection or db_path.")

    close_conn = False
    if con is None:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path))
        close_conn = True

    try:
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
                    res["artist_genres"] = res["artist_genres"].apply(
                        lambda v: v if isinstance(v, list | tuple) else []
                    )
        except Exception:
            if "artist_genres" not in res.columns:
                res["artist_genres"] = [[] for _ in range(len(res))]
            else:
                res["artist_genres"] = res["artist_genres"].apply(
                    lambda v: v if isinstance(v, list | tuple) else []
                )
        if not res.empty:
            res["percentage"] = res["percentage"].fillna(0.0)
        return res
    finally:
        if close_conn:
            con.close()


def get_playcount_by_day(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
    top_only: bool = True,
) -> pd.DataFrame:
    """Daily play counts (optionally top track per day) using DuckDB.

    - When `top_only=True`, returns exactly one row per day: the top track by plays.
    """
    if filtered_df.empty:
        return pd.DataFrame(columns=["date", "track", "artist", "play_count"])

    # Require DuckDB backend
    if con is None and db_path is None:
        raise ValueError("get_playcount_by_day requires a DuckDB connection or db_path.")

    close_conn = False
    if con is None:
        import duckdb  # type: ignore

        con = duckdb.connect(str(db_path))
        close_conn = True

    try:
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
    finally:
        if close_conn:
            con.close()
