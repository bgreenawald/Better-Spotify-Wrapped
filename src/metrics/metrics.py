import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.albums import (
    canonical_album_name,
    cluster_album_variants,
    compute_album_score,
    is_album_sized,
)
from src.metrics.utils import column_exists, extract_track_id


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
        cols = [
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
            "artist_id",
        ]
        if "track_id" in filtered_df.columns:
            cols.append("track_id")
        df = filtered_df[cols].copy()
        if "track_id" not in df.columns:
            df["track_id"] = filtered_df.get("spotify_track_uri", pd.Series(dtype=str)).apply(
                extract_track_id
            )

        rel = "df_tracks_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        lim = f"LIMIT {int(limit)}" if (limit is not None and limit > 0) else ""
        # Tracks are keyed by recording, not by display name. ISRC identifies a
        # recording across releases, so a song issued as a single and on an
        # album is one entity, while two different songs that share a name and
        # an artist stay separate. On the live data 14,560 ISRCs span more than
        # one track_id (up to 9 releases of one recording), and 10,892
        # (name, artist) pairs cover more than one recording -- so keying by
        # either track_id or name alone is wrong in a different direction.
        # The 9% of tracks without an ISRC fall back to their track_id.
        # ISRC lives in dim_tracks. When the schema is not attached (a bare
        # connection holding only the registered frame), fall back to track_id
        # and then to the title, so the function still works.
        has_tracks = column_exists(con, "dim_tracks", "track_isrc")
        isrc_join = "LEFT JOIN dim_tracks dt ON dt.track_id = p.track_id" if has_tracks else ""
        recording_key = (
            "COALESCE(NULLIF(TRIM(dt.track_isrc), ''), p.track_id, p.master_metadata_track_name)"
            if has_tracks
            else "COALESCE(p.track_id, p.master_metadata_track_name)"
        )
        sql = f"""
            WITH keyed AS (
                SELECT
                    {recording_key} AS recording_key,
                    p.master_metadata_track_name AS track_name,
                    p.master_metadata_album_artist_name AS artist,
                    COALESCE(p.artist_id, '') AS artist_id
                FROM {rel} p
                {isrc_join}
            ), plays AS (
                SELECT
                    recording_key,
                    COUNT(*) AS play_count,
                    -- A recording can carry slightly different titles across
                    -- releases; show the most frequent spelling.
                    MODE(track_name) AS track_name,
                    MODE(artist) AS artist,
                    MODE(artist_id) AS artist_id
                FROM keyed
                WHERE recording_key IS NOT NULL
                GROUP BY 1
            ), totals AS (
                SELECT SUM(play_count) AS total_plays FROM plays
            )
            SELECT (p.track_name || ' - ' || p.artist) AS track_artist,
                   p.track_name, p.artist, p.artist_id,
                   p.play_count,
                   -- 0-100, matching every other metric that returns a
                   -- 'percentage' column (artists, genres, trends).
                   ROUND((p.play_count::DOUBLE / NULLIF(t.total_plays, 0)) * 100, 2) AS percentage,
                   -- Dense, so a tie at the LIMIT boundary is visible rather
                   -- than silently cut.
                   DENSE_RANK() OVER (ORDER BY p.play_count DESC) AS rank
            FROM plays p CROSS JOIN totals t
            ORDER BY p.play_count DESC, track_artist
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

    The arithmetic and the name matching live in `src/metrics/albums.py` as
    pure functions; this function supplies the data and formats the result.

    Notes:
        - Plays are attributed by `track_id`. A track with the same name on
          another release (a compilation, a Greatest Hits) is a different
          track and is credited separately.
        - Editions of one album (Deluxe, Remastered, "Taylor's Version") are
          merged before scoring: their tracklists are unioned by track name
          and their play counts summed. The plain title is displayed. Matching
          is deliberately conservative, since merging discards the distinction
          between two releases -- a variant whose title differs beyond an
          edition suffix stays its own row.
        - Excludes short releases (5 or fewer tracks) to prevent singles/EPs
          from dominating. Applied *after* edition merging, so a release is
          judged on its merged tracklist.
        - An album where most tracks went unplayed scores 0. That is the
          metric reporting that the album was sampled, not listened to.

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

            # Plays are attributed strictly by track_id. The previous version
            # matched on (track_name, primary_artist), so every album holding a
            # same-named track by that artist received the full play count --
            # compilations, Greatest Hits and each deluxe edition all credited
            # at once. Editions are instead merged in Python below, by name.
            sql = """
                WITH played AS (
                    SELECT p.track_id, p.play_count, t.album_id
                    FROM df_play_counts p
                    JOIN dim_tracks t ON t.track_id = p.track_id
                    WHERE t.album_id IS NOT NULL
                ),
                albums_in_scope AS (
                    SELECT DISTINCT album_id FROM played
                ),
                album_tracks AS (
                    SELECT t.album_id, t.track_id, t.track_name
                    FROM dim_tracks t
                    JOIN albums_in_scope s ON s.album_id = t.album_id
                ),
                album_artist_counts AS (
                    SELECT t.album_id, ar.artist_name, COUNT(*) AS cnt
                    FROM dim_tracks t
                    JOIN albums_in_scope s ON s.album_id = t.album_id
                    JOIN bridge_track_artists b
                      ON b.track_id = t.track_id AND b."role" = 'primary'
                    JOIN dim_artists ar ON ar.artist_id = b.artist_id
                    GROUP BY 1, 2
                ),
                album_primary_artist AS (
                    SELECT album_id, artist_name,
                           ROW_NUMBER() OVER (
                               PARTITION BY album_id ORDER BY cnt DESC, artist_name
                           ) AS rn
                    FROM album_artist_counts
                )
                SELECT
                    atr.album_id,
                    al.album_name,
                    COALESCE(pa.artist_name, '') AS artist,
                    al.release_year,
                    atr.track_id,
                    atr.track_name,
                    COALESCE(pl.play_count, 0) AS play_count
                FROM album_tracks atr
                JOIN dim_albums al ON al.album_id = atr.album_id
                LEFT JOIN played pl ON pl.track_id = atr.track_id
                LEFT JOIN album_primary_artist pa
                    ON pa.album_id = atr.album_id AND pa.rn = 1
                ORDER BY atr.album_id, atr.track_name
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

            # One record per album_id, carrying its tracklist.
            per_album: list[dict[str, Any]] = []
            for album_id, group in res.groupby("album_id"):
                # Distinct tracks within this album_id, summing any duplicate
                # track names (a song can appear under more than one id).
                tracks = group.groupby("track_name", as_index=False, dropna=False).agg(
                    {"play_count": "sum"}
                )
                release_year = (
                    group["release_year"].iloc[0]
                    if pd.notna(group["release_year"].iloc[0])
                    else None
                )
                per_album.append(
                    {
                        "album_id": album_id,
                        "album_name": str(group["album_name"].iloc[0] or ""),
                        "artist": str(group["artist"].iloc[0] or ""),
                        "release_year": release_year,
                        "tracks": tracks,
                    }
                )

            # Merge editions of the same album (Deluxe, Remastered, Taylor's
            # Version, ...) before scoring or applying the length threshold.
            clusters = cluster_album_variants(
                [a["album_name"] for a in per_album],
                [a["artist"] for a in per_album],
            )

            albums = []
            for cluster in clusters:
                members = [per_album[i] for i in cluster]

                # Union the tracklists by name; a song appearing on the standard
                # and the deluxe edition is one song, and its plays are the sum
                # of what each edition's track id received.
                merged = (
                    pd.concat([m["tracks"] for m in members], ignore_index=True)
                    .groupby("track_name", as_index=False, dropna=False)
                    .agg({"play_count": "sum"})
                )

                total_tracks = len(merged)
                # Applied here, after merging, rather than in SQL before it.
                if not is_album_sized(total_tracks):
                    continue

                play_counts = [int(x) for x in merged["play_count"].tolist()]
                score = compute_album_score(play_counts)

                years = [m["release_year"] for m in members if m["release_year"] is not None]
                sorted_tracks = merged.sort_values("play_count", ascending=False)

                albums.append(
                    {
                        "album_name": canonical_album_name([m["album_name"] for m in members]),
                        "artist": members[0]["artist"],
                        "album_score": score.album_score,
                        "mqpc": score.mqpc,
                        "t_eff_capped": score.t_eff_capped,
                        "total_tracks": total_tracks,
                        "tracks_played": int((merged["play_count"] > 0).sum()),
                        "release_year": min(years) if years else None,
                        "track_details": [
                            {
                                "track_name": str(row["track_name"])
                                if pd.notna(row["track_name"])
                                else f"Track {i + 1}",
                                "play_count": int(row["play_count"]),
                            }
                            for i, (_, row) in enumerate(sorted_tracks.iterrows())
                        ],
                    }
                )

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

        `play_count` counts a play once for every genre tag on its primary
        artist, so an artist tagged with three genres contributes to three
        rows. The column therefore sums to more than the number of plays.
        `percentage` is a share of that genre-tagged total -- not of plays --
        so it sums to 100%.
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

            # A play is counted once per genre tag on its artist, so the genre
            # counts deliberately sum to more than the number of plays. Dividing
            # them by the raw play total mixed two different universes and made
            # the percentages sum to ~244%. Use the genre-count total instead, so
            # each value is a share of all genre tag attributions and the column
            # sums to 100%.
            genre_total = int(out["play_count"].sum())
            out["percentage"] = (
                (out["play_count"] / genre_total * 100).round(2) if genre_total else 0.0
            )
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
        acols = [
            "master_metadata_album_artist_name",
            "master_metadata_track_name",
            "artist_id",
        ]
        if "track_id" in filtered_df.columns:
            acols.append("track_id")
        df = filtered_df[acols].copy()
        if "track_id" not in df.columns:
            df["track_id"] = filtered_df.get("spotify_track_uri", pd.Series(dtype=str)).apply(
                extract_track_id
            )

        rel = "df_artist_in"
        with contextlib.suppress(Exception):
            con.unregister(rel)
        con.register(rel, df)

        lim = f"LIMIT {int(limit)}" if (limit is not None and limit > 0) else ""
        # Keyed by artist_id, not by display name: an artist who changed name
        # used to appear as two rows, and two artists sharing a name as one.
        # Rows with no artist_id fall back to the name so they are still
        # counted. unique_tracks counts distinct recordings (ISRC) rather than
        # distinct titles, to match how tracks are keyed.
        has_tracks_a = column_exists(con, "dim_tracks", "track_isrc")
        isrc_join_a = "LEFT JOIN dim_tracks dt ON dt.track_id = p.track_id" if has_tracks_a else ""
        recording_key_a = (
            "COALESCE(NULLIF(TRIM(dt.track_isrc), ''), p.track_id, p.master_metadata_track_name)"
            if has_tracks_a
            else "COALESCE(p.track_id, p.master_metadata_track_name)"
        )
        sql = f"""
            WITH keyed AS (
                SELECT
                    CASE
                        WHEN COALESCE(TRIM(p.artist_id), '') <> ''
                            THEN 'id:' || TRIM(p.artist_id)
                        ELSE 'name:' || COALESCE(p.master_metadata_album_artist_name, '')
                    END AS artist_key,
                    p.master_metadata_album_artist_name AS artist,
                    COALESCE(p.artist_id, '') AS artist_id,
                    {recording_key_a} AS recording_key
                FROM {rel} p
                {isrc_join_a}
            ), stats AS (
                SELECT
                    artist_key,
                    MODE(artist) AS artist,
                    MODE(artist_id) AS artist_id,
                    COUNT(*) AS play_count,
                    COUNT(DISTINCT recording_key) AS unique_tracks
                FROM keyed
                GROUP BY 1
            ), totals AS (
                SELECT SUM(play_count) AS total_plays FROM stats
            )
            SELECT s.artist, s.artist_id, s.play_count, s.unique_tracks,
                   ROUND((s.play_count::DOUBLE / NULLIF(t.total_plays, 0)) * 100, 2) AS percentage,
                   -- Dense, so a tie at the LIMIT boundary is visible.
                   DENSE_RANK() OVER (ORDER BY s.play_count DESC) AS rank
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
