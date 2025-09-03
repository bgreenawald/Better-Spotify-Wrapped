"""Simple CLI for Better Spotify Wrapped tasks.

Provides an `ingest-history` command to load Spotify listening history into
the normalized DuckDB schema (see DDL.sql).
"""

from __future__ import annotations

from pathlib import Path

import click
import duckdb

from .api.api import (
    populate_artist_genre_evidence_from_cache,
    populate_artist_genres_from_cache,
    populate_duration_and_explicit,
    populate_missing_track_isrcs,
    populate_track_albums,
    populate_track_artists,
    populate_track_metadata,
    populate_tracks_from_cached_albums,
)
from .db_ingest import IngestResult, load_history_into_fact_plays


def _maybe_apply_ddl(db_path: Path, ddl_path: Path) -> None:
    if not ddl_path.exists():
        raise click.ClickException(f"DDL file not found: {ddl_path}")
    conn = duckdb.connect(str(db_path))
    try:
        with ddl_path.open("r", encoding="utf-8") as f:
            sql = f.read()
        # Naively split on semicolons to execute multiple statements
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            conn.execute(stmt)
    finally:
        conn.close()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Better Spotify Wrapped CLI."""


@main.command("ingest-history")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--user-id",
    type=str,
    required=True,
    help="User UUID (required).",
)
@click.option(
    "--history-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Directory containing Spotify JSON exports (listening_history).",
)
@click.option(
    "--apply-ddl/--no-apply-ddl",
    default=False,
    show_default=True,
    help="Apply DDL.sql to the database before ingest.",
)
@click.option(
    "--ddl",
    "ddl_path",
    type=click.Path(path_type=Path),
    default=Path("DDL.sql"),
    show_default=True,
    help="Path to DDL file (used with --apply-ddl).",
)
def ingest_history(
    db_path: Path,
    user_id: str,
    history_dir: Path,
    apply_ddl: bool,
    ddl_path: Path,
) -> None:
    """Load listening history JSON files into `fact_plays`.

    Example:
      bsw ingest-history --user-id egreenawald \
        --db data/db/music.db \
        --history-dir data/egreenawald/listening_history \
        --apply-ddl
    """
    if apply_ddl:
        click.echo(f"Applying DDL from {ddl_path} to {db_path}...")
        _maybe_apply_ddl(db_path, ddl_path)

    click.echo(f"Ingesting history from {history_dir} into {db_path} for user {user_id}")

    res: IngestResult = load_history_into_fact_plays(
        db_path=db_path,
        user_id=user_id,
        history_dir=history_dir,
    )

    click.echo(
        " | ".join(
            [
                f"inserted_plays={res.inserted_plays}",
                f"deduped_plays={res.deduped_plays}",
                f"inserted_tracks={res.inserted_tracks}",
                f"existing_tracks={res.existing_tracks}",
            ]
        )
    )


@main.command("ingest-isrcs")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of tracks to process.",
)
def ingest_isrcs(db_path: Path, limit: int | None) -> None:
    """Populate missing dim_tracks.track_isrc using cache + Spotify API."""
    click.echo(
        f"Populating missing track ISRCs in {db_path}" + (f" (limit={limit})" if limit else "")
    )

    updated = populate_missing_track_isrcs(
        db_path=db_path,
        limit=limit,
    )

    click.echo(f"Updated rows: {updated}")


@main.command("ingest-duration-explicit")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of tracks to process.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
def ingest_duration_explicit(db_path: Path, limit: int | None, cache_dir: Path | None) -> None:
    """Populate missing dim_tracks.duration_ms and dim_tracks.explicit."""
    click.echo(
        f"Populating duration_ms and explicit in {db_path}" + (f" (limit={limit})" if limit else "")
    )

    dur_count, exp_count = populate_duration_and_explicit(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
    )

    click.echo(f"Updated duration_ms: {dur_count} | explicit: {exp_count}")


@main.command("ingest-track-metadata")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of tracks to process.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
def ingest_track_metadata(db_path: Path, limit: int | None, cache_dir: Path | None) -> None:
    """Populate ISRC, duration_ms, and explicit in one batch."""
    click.echo(
        f"Populating track metadata (isrc, duration, explicit) in {db_path}"
        + (f" (limit={limit})" if limit else "")
    )

    counts = populate_track_metadata(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
    )

    click.echo(
        f"Updated isrc: {counts['isrc']} | duration_ms: {counts['duration_ms']} | explicit: {counts['explicit']}"
    )


@main.command("ingest-track-albums")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of tracks to process.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
def ingest_track_albums(db_path: Path, limit: int | None, cache_dir: Path | None) -> None:
    """Populate dim_tracks.album_id and upsert dim_albums."""
    click.echo(
        f"Populating album_id and dim_albums in {db_path}" + (f" (limit={limit})" if limit else "")
    )

    counts = populate_track_albums(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
    )

    click.echo(
        f"Inserted albums: {counts['albums_inserted']} | Tracks updated: {counts['tracks_updated']}"
    )


@main.command("ingest-track-artists")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of tracks to process.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
def ingest_track_artists(db_path: Path, limit: int | None, cache_dir: Path | None) -> None:
    """Populate dim_artists and bridge_track_artists from track metadata."""
    click.echo(
        f"Populating artists and track-artist bridges in {db_path}"
        + (f" (limit={limit})" if limit else "")
    )

    counts = populate_track_artists(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
    )

    click.echo(
        f"Inserted artists: {counts['artists_inserted']} | Bridges inserted: {counts['bridges_inserted']}"
    )


@main.command("ingest-artist-genres")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of artist JSON files to scan.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
def ingest_artist_genres(db_path: Path, limit: int | None, cache_dir: Path | None) -> None:
    """Load artist genres from cached artist JSONs into tag_evidence."""
    counts = populate_artist_genre_evidence_from_cache(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
    )
    click.echo(
        f"Artists scanned: {counts['artists_scanned']} | Rows inserted: {counts['rows_inserted']}"
    )


@main.command("link-artist-genres")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of artist JSON files to scan.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
def link_artist_genres(db_path: Path, limit: int | None, cache_dir: Path | None) -> None:
    """Link artists to canonical genres using cached artist JSONs and map_genre."""
    counts = populate_artist_genres_from_cache(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
    )
    click.echo(
        " | ".join(
            [
                f"Artists scanned: {counts['artists_scanned']}",
                f"Links inserted: {counts['links_inserted']}",
                f"Unmapped tags: {counts['unmapped_tags']}",
                f"Artists missing in dim_artists: {counts['artists_missing_dim']}",
            ]
        )
    )


@main.command("ingest-tracks-from-albums")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of album JSON files to scan.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
def ingest_tracks_from_albums(db_path: Path, limit: int | None, cache_dir: Path | None) -> None:
    """Load tracks from cached album JSONs into dim_tracks."""
    counts = populate_tracks_from_cached_albums(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
    )
    click.echo(
        f"Albums scanned: {counts['albums_scanned']} | Tracks inserted: {counts['tracks_inserted']} | Tracks updated: {counts['tracks_updated']}"
    )


if __name__ == "__main__":
    main()
