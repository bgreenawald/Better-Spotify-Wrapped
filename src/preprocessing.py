from __future__ import annotations

import contextlib
import datetime as dt
from collections.abc import Iterable, Sequence

import duckdb
import pandas as pd

from src.api.api import SpotifyData


def add_api_data(
    history_df: pd.DataFrame,
    api_data: SpotifyData,
) -> pd.DataFrame:
    """Add Spotify track, album, artist IDs, and genres to listening history.

    Extracts track ID from the Spotify URI, maps it to album and artist IDs
    from the API data, and adds the artist's genres.

    Args:
        history_df (pd.DataFrame): Listening history with a
            'spotify_track_uri' column.
        api_data (SpotifyData): Spotify API data containing 'tracks' and
            'artists' mappings.

    Returns:
        pd.DataFrame: Updated DataFrame with 'track_id', 'album_id',
            'artist_id', and 'artist_genres' columns added.
    """
    # Extract track ID from the Spotify URI (everything after the last colon)
    history_df["track_id"] = history_df["spotify_track_uri"].str.rsplit(":", n=1).str[-1]

    # Build mapping from track ID to album and artist IDs
    track_to_album = {track_id: info["album"]["id"] for track_id, info in api_data.tracks.items()}
    track_to_artist = {
        track_id: info["artists"][0]["id"] for track_id, info in api_data.tracks.items()
    }

    # Map album_id and artist_id columns
    history_df["album_id"] = history_df["track_id"].map(track_to_album)
    history_df["artist_id"] = history_df["track_id"].map(track_to_artist)

    # Build mapping from artist ID to genres
    artist_to_genres = {
        artist_id: tuple(data["genres"]) for artist_id, data in api_data.artists.items()
    }
    history_df["artist_genres"] = history_df["artist_id"].map(artist_to_genres)

    return history_df


def filter_songs(
    history_df: pd.DataFrame,
    user_id: str | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    exclude_december: bool = True,
    remove_incognito: bool = True,
    excluded_tracks: list[str] | None = None,
    excluded_artists: list[str] | None = None,
    excluded_albums: list[str] | None = None,
    excluded_genres: list[str] | None = None,
    allow_missing_user_column: bool = False,
) -> pd.DataFrame:
    """Filter listening history based on various criteria.

    Applies filters to exclude podcasts, out-of-range dates, skipped tracks,
    zero playtime, unknown reasons, incognito mode, and optional exclusions
    of specific tracks, artists, albums, or genres.

    Args:
        history_df (pd.DataFrame): Spotify listening history.
        start_date (Optional[pd.Timestamp]): Minimum timestamp (inclusive).
        end_date (Optional[pd.Timestamp]): Maximum timestamp (inclusive).
        exclude_december (bool): Exclude plays from December if True.
        remove_incognito (bool): Exclude incognito-mode plays if True.
        excluded_tracks (Optional[List[str]]): Track names to exclude.
        excluded_artists (Optional[List[str]]): Artist names to exclude.
        excluded_albums (Optional[List[str]]): Album names to exclude.
        excluded_genres (Optional[List[str]]): Genres to exclude.

    Returns:
        pd.DataFrame: Filtered DataFrame containing plays that match criteria.
    """

    history_df = history_df.copy()
    # Ensure 'ts' is datetime (coerce invalid values to NaT)
    if not pd.api.types.is_datetime64_any_dtype(history_df["ts"]):
        history_df["ts"] = pd.to_datetime(history_df["ts"], errors="coerce")
    # Always provide explicit year/month/day columns for consistent grouping/filtering
    history_df["year"] = history_df["ts"].dt.year
    history_df["month"] = history_df["ts"].dt.month
    history_df["day"] = history_df["ts"].dt.day

    mask = pd.Series(True, index=history_df.index)

    # Optional user filter
    if user_id is not None:
        if "user_id" not in history_df.columns:
            if not allow_missing_user_column:
                raise ValueError(
                    f"user_id filtering requested ({repr(user_id)}), but 'user_id' column is missing from history_df"
                )
        else:
            mask &= history_df["user_id"].eq(user_id)

    # Exclude episodes (podcasts) when column exists and filter to rows with track URI if present
    if "episode_name" in history_df.columns:
        mask &= history_df["episode_name"].isna()
    if "spotify_track_uri" in history_df.columns:
        mask &= history_df["spotify_track_uri"].notna()

    # Apply date range filters
    if start_date is not None:
        mask &= history_df["ts"] >= start_date
    if end_date is not None:
        mask &= history_df["ts"] <= end_date

    # Optionally exclude plays in December
    if exclude_december:
        mask &= history_df["ts"].dt.month < 12

    # Remove skipped tracks and zero-playtime sessions (default-safe if columns absent)
    if "skipped" in history_df.columns:
        mask &= ~history_df["skipped"].fillna(False)
    ms_series = history_df.get("ms_played")
    if ms_series is not None:
        ms_series_coerced = pd.to_numeric(ms_series, errors="coerce").fillna(0)
        mask &= ms_series_coerced > 0

    # Exclude plays with unknown start or end reasons
    rs = history_df.get("reason_start")
    if rs is not None:
        mask &= rs.ne("unknown").fillna(True)
    re = history_df.get("reason_end")
    if re is not None:
        mask &= re.ne("unknown").fillna(True)

    # Optionally remove incognito-mode plays
    if remove_incognito and "incognito_mode" in history_df.columns:
        mask &= ~history_df["incognito_mode"].fillna(False)

    # Optionally exclude specific tracks, artists, albums
    if excluded_tracks and "master_metadata_track_name" in history_df.columns:
        mask &= ~history_df["master_metadata_track_name"].isin(excluded_tracks)
    if excluded_artists and "master_metadata_album_artist_name" in history_df.columns:
        mask &= ~history_df["master_metadata_album_artist_name"].isin(excluded_artists)
    if excluded_albums and "master_metadata_album_album_name" in history_df.columns:
        mask &= ~history_df["master_metadata_album_album_name"].isin(excluded_albums)

    # Optionally exclude plays by genre
    if excluded_genres and "artist_genres" in history_df.columns:
        excluded_genres_set = {g.lower() for g in excluded_genres}
        mask &= ~history_df["artist_genres"].apply(
            lambda genres: (
                any(str(g).lower() in excluded_genres_set for g in genres)
                if hasattr(genres, "__iter__") and not isinstance(genres, str)
                else (
                    str(genres).lower() in excluded_genres_set
                    if genres is not None and not pd.isna(genres)
                    else False
                )
            )
        )

    return history_df.loc[mask].copy()


# --- DuckDB-backed filtering (migration path) ---


def _to_date(v: object) -> dt.datetime | None:
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _register_list_param(
    con: duckdb.DuckDBPyConnection, name: str, values: Iterable[str] | None
) -> str:
    """Register a small list as a temp relation and return its table name.

    DuckDB does not support array parameters in the same way as some drivers.
    For optional exclusion lists, we register a tiny one-column table that can
    be left-joined or used in `IN (SELECT ...)` clauses.
    """
    if not values:
        return ""
    df = pd.DataFrame({"val": list(values)})
    relname = f"tmp_{name}"
    # Overwrite if already present in this connection
    with contextlib.suppress(Exception):
        con.unregister(relname)
    con.register(relname, df)
    return relname


def get_filtered_plays(
    con: duckdb.DuckDBPyConnection,
    *,
    user_id: str,
    start_date: pd.Timestamp | dt.datetime | str | None = None,
    end_date: pd.Timestamp | dt.datetime | str | None = None,
    exclude_december: bool = True,
    remove_incognito: bool = True,
    excluded_tracks: Sequence[str] | None = None,
    excluded_artists: Sequence[str] | None = None,
    excluded_albums: Sequence[str] | None = None,
    excluded_genres: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fetch filtered plays for a user from DuckDB, returning a pandas DataFrame.

    This function is the DB-backed analog to `filter_songs` and is designed to
    evolve into the primary code path. It returns columns compatible with the
    current pandas pipeline where possible:

    - `ts` (TIMESTAMP): play time
    - `ms_played` (INT): duration_ms
    - `spotify_track_uri` (TEXT): synthesized `spotify:track:{track_id}`
    - `master_metadata_track_name` (TEXT): from dim_tracks.track_name
    - `master_metadata_album_artist_name` (TEXT, nullable): primary artist name
    - `master_metadata_album_album_name` (TEXT, nullable): album name
    - `artist_id` (TEXT, nullable)
    - `artist_genres` (ARRAY/TUPLE placeholder): currently empty tuple; replace when genre dims populated
    - `reason_start`, `reason_end`, `skipped`, `incognito_mode`

    Notes:
    - Excluding by artists/albums requires populated `bridge_track_artists` and
      `dim_albums`. If those tables are empty, the filters are effectively ignored.
    - Excluding by genres uses `track_genres` + `dim_genres.name`. If unpopulated,
      no rows are excluded.
    """
    s_date = _to_date(start_date)
    e_date = _to_date(end_date)

    # Register small exclusion lists as temp relations
    rel_tracks = _register_list_param(con, "excluded_tracks", excluded_tracks)
    rel_artists = _register_list_param(con, "excluded_artists", excluded_artists)
    rel_albums = _register_list_param(con, "excluded_albums", excluded_albums)
    rel_genres = _register_list_param(con, "excluded_genres", excluded_genres)

    def _table_exists(name: str) -> bool:
        try:
            if "." in name:
                schema_part, table_part = name.split(".", 1)
                row = con.execute(
                    "SELECT 1 FROM duckdb_tables() WHERE LOWER(schema_name) = LOWER(?) AND LOWER(table_name) = LOWER(?)",
                    [schema_part, table_part],
                ).fetchone()
                return row is not None
            else:
                df = con.execute(
                    "SELECT LOWER(table_name) as ln_table, LOWER(schema_name) as ln_schema FROM duckdb_tables()"
                ).df()
                lower_name = name.lower()
                return any(lower_name == row.ln_table for row in df.itertuples())
        except Exception:
            return False

    filters = ["p.user_id = ?"]
    params: list[object] = [user_id]
    if s_date is not None:
        filters.append("p.played_at >= ?")
        params.append(s_date)
    if e_date is not None:
        filters.append("p.played_at <= ?")
        params.append(e_date)
    if exclude_december:
        filters.append("EXTRACT(month FROM p.played_at) <> 12")
    # Require duration_ms > 0, excluding NULLs
    filters.append("COALESCE(p.duration_ms, 0) > 0")
    filters.append("COALESCE(p.reason_start, 'unknown') <> 'unknown'")
    filters.append("COALESCE(p.reason_end, 'unknown') <> 'unknown'")
    if remove_incognito:
        filters.append("COALESCE(p.incognito_mode, false) = false")
    # Optional name-based exclusions
    extra_where = []
    if rel_tracks:
        extra_where.append(f"COALESCE(t.track_name, '') NOT IN (SELECT val FROM {rel_tracks})")
    if rel_artists:
        extra_where.append(f"COALESCE(ar.artist_name, '') NOT IN (SELECT val FROM {rel_artists})")
    if rel_albums:
        extra_where.append(f"COALESCE(al.album_name, '') NOT IN (SELECT val FROM {rel_albums})")
    # Global genre exclusion using both track_genres and artist_genres, honoring parent mappings
    if rel_genres and _table_exists("dim_genres"):
        has_tg = _table_exists("track_genres")
        has_ag = _table_exists("artist_genres") and _table_exists("bridge_track_artists")
        clauses: list[str] = []
        # Normalize comparison to lower-case to avoid case mismatches
        has_hier = _table_exists("genre_hierarchy")
        if has_tg:
            # Exclude tracks tagged with an excluded child genre name
            clauses.append(
                "SELECT tg.track_id FROM track_genres tg "
                "JOIN dim_genres g ON g.genre_id = tg.genre_id "
                f"WHERE lower(g.name) IN (SELECT lower(val) FROM {rel_genres})"
            )
            # Exclude tracks whose child maps to an excluded parent genre
            if has_hier:
                clauses.append(
                    "SELECT tg.track_id FROM track_genres tg "
                    "JOIN genre_hierarchy gh ON gh.child_genre_id = tg.genre_id "
                    "JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id "
                    f"WHERE lower(pg.name) IN (SELECT lower(val) FROM {rel_genres})"
                )
        if has_ag:
            # Exclude tracks whose primary artist has an excluded child or parent genre
            if has_hier:
                clauses.append(
                    "SELECT b.track_id FROM bridge_track_artists b "
                    "JOIN artist_genres ag ON ag.artist_id = b.artist_id "
                    "JOIN dim_genres g ON g.genre_id = ag.genre_id "
                    "WHERE b.role = 'primary' AND ("
                    f"lower(g.name) IN (SELECT lower(val) FROM {rel_genres})"
                    " OR EXISTS ("
                    "   SELECT 1 FROM genre_hierarchy gh "
                    "   JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id "
                    "   WHERE gh.child_genre_id = ag.genre_id "
                    f"     AND lower(pg.name) IN (SELECT lower(val) FROM {rel_genres})"
                    " ) )"
                )
            else:
                clauses.append(
                    "SELECT b.track_id FROM bridge_track_artists b "
                    "JOIN artist_genres ag ON ag.artist_id = b.artist_id "
                    "JOIN dim_genres g ON g.genre_id = ag.genre_id "
                    "WHERE b.role = 'primary' AND "
                    f"lower(g.name) IN (SELECT lower(val) FROM {rel_genres})"
                )
        if clauses:
            union_subq = " UNION ".join(clauses)
            extra_where.append("p.track_id NOT IN (" + union_subq + ")")

    where_clause = " AND \n            ".join(filters + extra_where)

    sql = f"""
        WITH plays AS (
            SELECT
                p.played_at AS ts,
                p.duration_ms AS ms_played,
                p.reason_start,
                p.reason_end,
                p.skipped,
                p.incognito_mode,
                p.track_id,
                t.track_name,
                t.album_id,
                -- Normalize artist_id: trim and strip any URI prefix like 'spotify:artist:'
                REGEXP_REPLACE(TRIM(ar.artist_id), '.*:', '') AS artist_id,
                ar.artist_name,
                al.album_name
            FROM fact_plays p
            LEFT JOIN dim_tracks t ON t.track_id = p.track_id
            LEFT JOIN bridge_track_artists b
                ON b.track_id = p.track_id AND b.role = 'primary'
            LEFT JOIN dim_artists ar ON ar.artist_id = b.artist_id
            LEFT JOIN dim_albums al ON al.album_id = t.album_id
            WHERE {where_clause}
        ), artist_genres_agg AS (
            -- Build formatted artist genres as "Child (Parent1, Parent2)" when parents exist;
            -- otherwise just the genre name. De-duplicate labels per artist.
            WITH parent_map AS (
                SELECT gh.child_genre_id,
                       STRING_AGG(DISTINCT pg.name, ', ' ORDER BY pg.name) AS parent_names
                FROM genre_hierarchy gh
                JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id
                WHERE COALESCE(pg.active, TRUE)
                GROUP BY gh.child_genre_id
            ), labels AS (
                SELECT DISTINCT TRIM(ag.artist_id) AS artist_id,
                       CASE
                           WHEN COALESCE(pm.parent_names, '') = '' THEN g.name
                           ELSE (g.name || ' (' || pm.parent_names || ')')
                       END AS label
                FROM artist_genres ag
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                LEFT JOIN parent_map pm ON pm.child_genre_id = g.genre_id
                WHERE COALESCE(g.active, TRUE)
            )
            SELECT artist_id, list(label) AS artist_genres
            FROM labels
            GROUP BY artist_id
        )
        SELECT
            p.ts,
            p.ms_played,
            p.reason_start,
            p.reason_end,
            p.skipped,
            p.incognito_mode,
            p.track_id,
            p.track_name,
            p.artist_id,
            p.artist_name,
            p.album_name,
            aga.artist_genres
        FROM plays p
        LEFT JOIN artist_genres_agg aga ON aga.artist_id = p.artist_id
    """

    df = con.execute(sql, params).df()

    # Synthesize legacy-compatible columns for current UI/metrics
    if not df.empty:
        df["spotify_track_uri"] = df["track_id"].apply(
            lambda x: f"spotify:track:{x}" if pd.notna(x) else None
        )
        df.rename(
            columns={
                "track_name": "master_metadata_track_name",
                "artist_name": "master_metadata_album_artist_name",
                "album_name": "master_metadata_album_album_name",
            },
            inplace=True,
        )
        # Normalize list/array to tuples for downstream stability
        if "artist_genres" not in df.columns:
            df["artist_genres"] = [() for _ in range(len(df))]
        else:

            def _to_tuple(v):
                try:
                    if v is None:
                        return ()
                    if isinstance(v, list | tuple | set):
                        return tuple(v)
                    if hasattr(v, "__iter__") and not isinstance(v, str | bytes | dict):
                        return tuple(v)
                except Exception:
                    pass
                return ()

            df["artist_genres"] = df["artist_genres"].apply(_to_tuple)

    # Ensure dtypes are pandas-friendly
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    return df
