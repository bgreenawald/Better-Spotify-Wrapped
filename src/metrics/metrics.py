import contextlib
import math
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

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
        # end query
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


def _deduplicate_albums_fuzzy(
    albums: list[dict[str, Any]], similarity_threshold: float = 90.0
) -> list[dict[str, Any]]:
    """Deduplicate albums using fuzzy matching on album names with exact artist matching.

    Args:
        albums: List of album dictionaries with 'album_name', 'artist', 'album_score', etc.
        similarity_threshold: Minimum similarity score (0-100) to consider albums as duplicates.

    Returns:
        Deduplicated list of albums, keeping the highest-scoring variant for each match.
    """
    if not albums:
        return albums

    # Group albums by exact artist name for efficient comparison
    artist_groups: dict[str, list[dict[str, Any]]] = {}
    for album in albums:
        artist = album.get("artist", "")
        if artist not in artist_groups:
            artist_groups[artist] = []
        artist_groups[artist].append(album)

    deduplicated = []

    # Process each artist group separately
    for _artist, artist_albums in artist_groups.items():
        if len(artist_albums) == 1:
            # No duplicates possible if only one album by this artist
            deduplicated.extend(artist_albums)
            continue

        # Track which albums have been merged
        used_indices = set()

        for i, album_a in enumerate(artist_albums):
            if i in used_indices:
                continue

            # Find all albums similar to album_a
            similar_group = [album_a]
            album_a_name = str(album_a.get("album_name", ""))

            for j in range(i + 1, len(artist_albums)):
                if j in used_indices:
                    continue

                album_b = artist_albums[j]
                album_b_name = str(album_b.get("album_name", ""))

                # Use token_set_ratio for fuzzy matching (handles editions, remasters, etc.)
                similarity = fuzz.token_set_ratio(album_a_name.lower(), album_b_name.lower())

                if similarity >= similarity_threshold:
                    similar_group.append(album_b)
                    used_indices.add(j)

            # Keep the album with the highest album_score from the similar group
            best_album = max(similar_group, key=lambda x: x.get("album_score", 0.0))
            deduplicated.append(best_album)
            used_indices.add(i)

    return deduplicated


def get_top_albums(
    filtered_df: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    con: Any | None = None,
) -> pd.DataFrame:
    """Calculate top albums based on a robust album score metric.

    Uses DuckDB exclusively: provide `con` or `db_path` to query the schema in DDL.sql.

    The album score is computed as:
    1. MQPC: Mean of middle 50% of track play counts (sorted ascending)
    2. t_eff: Number of tracks with play_count >= 0.5 * MQPC, capped at 15
    3. size_factor: 1 + 0.3 * (t_eff_capped / 15)
    4. album_score: MQPC * size_factor

    Args:
        filtered_df: DataFrame already filtered by `filter_songs()` or
            `get_filtered_plays()`.
        db_path: Optional DuckDB database path (if `con` not provided).
        con: Optional DuckDB connection to use.

    Notes:
        - Excludes short releases (albums with 5 or fewer tracks) to prevent
          singles/EPs from dominating.

    Returns:
        pd.DataFrame: Albums sorted by album_score with columns:
            'album_name', 'artist', 'album_score', 'mqpc', 't_eff_capped',
            'total_tracks', 'tracks_played', 'release_year', and 'track_details'
            (a list of dicts with 'track_name' and 'play_count' for each track).
    """
    # Fast exit
    if filtered_df.empty:
        return pd.DataFrame(
            columns=[
                "album_name",
                "artist",
                "album_score",
                "mqpc",
                "t_eff_capped",
                "total_tracks",
                "tracks_played",
                "release_year",
                "track_details",
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
                        "album_score",
                        "mqpc",
                        "t_eff_capped",
                        "total_tracks",
                        "tracks_played",
                        "release_year",
                        "track_details",
                    ]
                )

            # Register plays as a temporary relation
            if hasattr(con, "unregister"):
                with contextlib.suppress(AttributeError, KeyError):
                    con.unregister("df_play_counts")
            con.register("df_play_counts", play_counts)

            # SQL plan (uses convenience views where helpful):
            # 1) Get ALL played track_ids (including singles) with their play counts
            # 2) Find which albums contain tracks matching those track names + artists
            # 3) Sum play counts by (track_name, artist) across ALL versions (single + album)
            # 4) Map aggregated plays to album tracks
            # 5) Pick a primary artist (mode over primary role)
            sql = """
                WITH all_plays_with_metadata AS (
                    SELECT p.track_id, p.play_count, t.track_name, t.album_id
                    FROM df_play_counts p
                    JOIN dim_tracks t ON t.track_id = p.track_id
                ),
                played_track_artists AS (
                    SELECT ap.track_id,
                        COALESCE(a.artist_name, '') AS primary_artist,
                        ROW_NUMBER() OVER (PARTITION BY ap.track_id ORDER BY a.artist_name) AS rn
                    FROM all_plays_with_metadata ap
                    LEFT JOIN bridge_track_artists b ON b.track_id = ap.track_id AND b.role = 'primary'
                    LEFT JOIN dim_artists a ON a.artist_id = b.artist_id
                ),
                played_track_artists_deduped AS (
                    SELECT track_id, primary_artist
                    FROM played_track_artists
                    WHERE rn = 1 OR rn IS NULL
                ),
                plays_with_artists AS (
                    SELECT ap.track_name,
                        pta.primary_artist,
                        SUM(ap.play_count) AS total_play_count
                    FROM all_plays_with_metadata ap
                    LEFT JOIN played_track_artists_deduped pta ON pta.track_id = ap.track_id
                    GROUP BY 1, 2
                ),
                albums_in_scope AS (
                    SELECT DISTINCT t.album_id
                    FROM dim_tracks t
                    WHERE EXISTS (
                        SELECT 1 FROM all_plays_with_metadata ap
                        WHERE ap.track_id = t.track_id AND t.album_id IS NOT NULL
                    )
                ),
                all_album_tracks AS (
                    SELECT t.album_id, t.track_id, t.track_name
                    FROM dim_tracks t
                    JOIN albums_in_scope s ON s.album_id = t.album_id
                ),
                track_primary_artists AS (
                    SELECT b.track_id,
                        COALESCE(a.artist_name, '') AS primary_artist,
                        ROW_NUMBER() OVER (PARTITION BY b.track_id ORDER BY a.artist_name) AS rn
                    FROM dim_tracks t
                    JOIN albums_in_scope s ON s.album_id = t.album_id
                    LEFT JOIN bridge_track_artists b ON b.track_id = t.track_id AND b.role = 'primary'
                    LEFT JOIN dim_artists a ON a.artist_id = b.artist_id
                ),
                track_primary_artists_deduped AS (
                    SELECT track_id, primary_artist
                    FROM track_primary_artists
                    WHERE rn = 1 OR rn IS NULL
                ),
                album_track_counts AS (
                    SELECT a.album_id,
                        aat.track_id,
                        aat.track_name,
                        tpa.primary_artist,
                        COALESCE(pwa.total_play_count, 0) AS play_count
                    FROM all_album_tracks aat
                    JOIN dim_albums a ON a.album_id = aat.album_id
                    LEFT JOIN track_primary_artists_deduped tpa ON tpa.track_id = aat.track_id
                    LEFT JOIN plays_with_artists pwa
                        ON pwa.track_name = aat.track_name
                        AND pwa.primary_artist = tpa.primary_artist
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
                ),
                album_metadata AS (
                    SELECT
                        al.album_id,
                        al.album_name,
                        COALESCE(pa.artist_name, '') AS artist,
                        COUNT(*) AS total_tracks,
                        SUM(CASE WHEN atc.play_count > 0 THEN 1 ELSE 0 END) AS tracks_played,
                        al.release_year
                    FROM album_track_counts atc
                    JOIN dim_albums al ON al.album_id = atc.album_id
                    LEFT JOIN album_primary_artist pa ON pa.album_id = atc.album_id AND pa.rn = 1
                    GROUP BY 1, 2, 3, 6
                    HAVING COUNT(*) > 5
                )
                SELECT
                    am.album_id,
                    am.album_name,
                    am.artist,
                    am.total_tracks,
                    am.tracks_played,
                    am.release_year,
                    atc.track_id,
                    atc.track_name,
                    atc.primary_artist,
                    atc.play_count
                FROM album_metadata am
                JOIN album_track_counts atc ON atc.album_id = am.album_id
                ORDER BY am.album_id, atc.track_name;

            """
            res = con.execute(sql).df()

            # Process in Python to compute album_score
            if res.empty:
                return pd.DataFrame(
                    columns=[
                        "album_name",
                        "artist",
                        "album_score",
                        "mqpc",
                        "t_eff_capped",
                        "total_tracks",
                        "tracks_played",
                        "release_year",
                        "track_details",
                    ]
                )

            # Group by album and compute the metric
            albums = []
            for _album_id, group in res.groupby("album_id"):
                # Get metadata (same for all rows in group)
                album_name = group["album_name"].iloc[0]
                artist = group["artist"].iloc[0]
                release_year = (
                    group["release_year"].iloc[0]
                    if pd.notna(group["release_year"].iloc[0])
                    else None
                )

                # Deduplicate tracks by (track_name, primary_artist) - same song, different IDs
                # This handles cases where Spotify has the same song as both single and album track
                dedupe_cols = ["track_name", "primary_artist"]
                deduped_group = group.groupby(dedupe_cols, as_index=False, dropna=False).agg(
                    {"play_count": "sum"}
                )

                # Recalculate total_tracks and tracks_played based on deduplicated tracks
                total_tracks = len(deduped_group)
                tracks_played = int((deduped_group["play_count"] > 0).sum())

                # Get sorted play counts (ascending) from deduplicated tracks
                play_counts = sorted(deduped_group["play_count"].tolist())
                n = len(play_counts)

                # Compute MQPC: mean of middle 50% range
                # Range: from index ceil(0.25 * n) to floor(0.75 * n) (1-indexed, so subtract 1 for 0-indexed)
                start_idx = math.ceil(0.25 * n) - 1  # Convert to 0-indexed
                end_idx = math.floor(0.75 * n) - 1  # Convert to 0-indexed
                # Ensure valid indices
                start_idx = max(0, start_idx)
                end_idx = min(n - 1, end_idx)

                if start_idx <= end_idx:
                    middle_range = play_counts[start_idx : end_idx + 1]
                    mqpc = sum(middle_range) / len(middle_range) if middle_range else 0.0
                else:
                    mqpc = 0.0

                # Compute t_eff: tracks with play_count >= 0.5 * MQPC
                threshold = 0.5 * mqpc
                t_eff = sum(1 for pc in play_counts if pc >= threshold)
                t_eff_capped = min(t_eff, 15)

                # Compute size_factor
                size_factor = 1.0 + 0.3 * (t_eff_capped / 15.0)

                # Compute album_score
                album_score = mqpc * size_factor

                # Build track details from deduplicated data
                track_details = []
                if "track_name" in deduped_group.columns:
                    # Sort by play_count descending to show most played tracks first
                    sorted_deduped = deduped_group.sort_values("play_count", ascending=False)

                    track_details = [
                        {
                            "track_name": str(row["track_name"])
                            if pd.notna(row["track_name"])
                            else f"Track {i + 1}",
                            "play_count": int(row["play_count"]),
                        }
                        for i, (_, row) in enumerate(sorted_deduped.iterrows())
                    ]
                else:
                    # Fallback: use play_counts list without names
                    track_details = [
                        {"track_name": f"Track {i + 1}", "play_count": int(pc)}
                        for i, pc in enumerate(sorted(play_counts, reverse=True))
                    ]

                albums.append(
                    {
                        "album_name": album_name,
                        "artist": artist,
                        "album_score": album_score,
                        "mqpc": mqpc,
                        "t_eff_capped": t_eff_capped,
                        "total_tracks": total_tracks,
                        "tracks_played": tracks_played,
                        "release_year": release_year,
                        "track_details": track_details,
                    }
                )

            # Deduplicate albums using fuzzy matching on album names (exact artist match)
            albums = _deduplicate_albums_fuzzy(albums, similarity_threshold=70.0)

            result_df = pd.DataFrame(albums)
            if not result_df.empty:
                result_df = result_df.sort_values("album_score", ascending=False)
            return result_df
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
        # Attach artist_genres directly from DB taxonomy to avoid reliance on input frame
        try:
            if not res.empty:
                # Prepare a small relation of artist_ids present
                aid = res[["artist_id"]].assign(
                    artist_id=lambda d: d["artist_id"].fillna("").astype(str).str.strip(),
                )
                aid = aid[aid["artist_id"] != ""].drop_duplicates()
                if not aid.empty:
                    rel_ids = "rel_artist_ids"
                    with contextlib.suppress(Exception):
                        con.unregister(rel_ids)
                    con.register(rel_ids, aid)
                    sql_gen = f"""
                        WITH parent_map AS (
                            SELECT gh.child_genre_id,
                                   STRING_AGG(DISTINCT pg.name, ', ' ORDER BY pg.name) AS parent_names
                            FROM genre_hierarchy gh
                            JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id
                            WHERE COALESCE(pg.active, TRUE)
                            GROUP BY gh.child_genre_id
                        ), labels AS (
                            SELECT DISTINCT REGEXP_REPLACE(TRIM(ag.artist_id), '.*:', '') AS artist_id,
                                   CASE
                                       WHEN COALESCE(pm.parent_names, '') = '' THEN g.name
                                       ELSE (g.name || ' (' || pm.parent_names || ')')
                                   END AS label
                            FROM artist_genres ag
                            JOIN dim_genres g ON g.genre_id = ag.genre_id
                            LEFT JOIN parent_map pm ON pm.child_genre_id = g.genre_id
                            JOIN {rel_ids} r ON r.artist_id = REGEXP_REPLACE(TRIM(ag.artist_id), '.*:', '')
                            WHERE COALESCE(g.active, TRUE)
                        )
                        SELECT artist_id, list(label) AS artist_genres
                        FROM labels
                        GROUP BY artist_id
                    """
                    genres_map = con.execute(sql_gen).df()
                    if not genres_map.empty:
                        res = res.merge(genres_map, on="artist_id", how="left")
        except Exception:
            pass
        # Fallback to input-mapped genres if still missing
        if "artist_genres" not in res.columns:
            try:
                if "artist_genres" in filtered_df.columns and not res.empty:
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
            except Exception:
                pass
        # Normalize column to list for downstream formatting
        if "artist_genres" not in res.columns:
            res["artist_genres"] = [[] for _ in range(len(res))]
        else:

            def _to_list(v):
                try:
                    if v is None:
                        return []
                    if isinstance(v, list | tuple | set):
                        return list(v)
                    if hasattr(v, "__iter__") and not isinstance(v, str | bytes | dict):
                        return list(v)
                except Exception:
                    pass
                return []

            res["artist_genres"] = res["artist_genres"].apply(_to_list)
        # end normalization
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
