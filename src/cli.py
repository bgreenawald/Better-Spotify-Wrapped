"""Simple CLI for Slopify Wrapped tasks.

Provides an `ingest-history` command to load Spotify listening history into
the normalized DuckDB schema (see DDL.sql).
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import duckdb

from .api.api import (
    fetch_album_cache,
    fetch_artist_cache,
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
from .status import collect_status


def _maybe_apply_ddl(db_path: Path, ddl_path: Path) -> None:
    if not ddl_path.exists():
        raise click.ClickException(f"DDL file not found: {ddl_path}")
    sql = ddl_path.read_text(encoding="utf-8")
    conn = duckdb.connect(str(db_path))
    try:
        # DuckDB accepts multi-statement strings, so hand it the whole file.
        # Splitting on ";" breaks on semicolons inside comments and string
        # literals. The transaction keeps a failure from leaving a partial
        # schema behind.
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(sql)
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
)
def main(verbose: int) -> None:
    """Slopify Wrapped CLI."""
    # Without this, every logging.info/warning in the codebase is discarded,
    # so a failed batch or a skipped row leaves no trace anywhere.
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command("init-db")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file to initialize.",
)
@click.option(
    "--ddl",
    "ddl_path",
    type=click.Path(path_type=Path),
    default=Path("DDL.sql"),
    show_default=True,
    help="Path to DDL file with schema definitions.",
)
def init_db(db_path: Path, ddl_path: Path) -> None:
    """Initialize a new database with the schema from DDL.sql.

    Creates all tables, indexes, and views defined in DDL.sql.
    Safe to run on existing databases (uses IF NOT EXISTS for most objects).
    """
    click.echo(f"Initializing database {db_path} with schema from {ddl_path}...")

    # Create parent directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _maybe_apply_ddl(db_path, ddl_path)

    # Verify initialization
    conn = duckdb.connect(str(db_path))
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_count = len(tables)
        click.echo(f"✓ Database initialized successfully with {table_count} tables")
        if table_count > 0:
            click.echo(
                f"  Tables: {', '.join(t[0] for t in tables[:5])}"
                + (f" and {table_count - 5} more..." if table_count > 5 else "")
            )
    except Exception as e:
        click.echo(f"✗ Verification failed: {e}", err=True)
        raise
    finally:
        conn.close()


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
    help="User id (required), e.g. ben.greenawald.",
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
      slopify ingest-history --user-id egreenawald \
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
    if res.skipped_episodes:
        click.echo(f"Skipped {res.skipped_episodes} podcast episode row(s) (not modelled).")
    if res.skipped_rows:
        # Loud, because these rows are simply gone from the database.
        click.echo(
            f"WARNING: dropped {res.skipped_rows} row(s) with an unparseable timestamp "
            "or no Spotify track URI. Re-run with -v for details.",
            err=True,
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


@main.command("fetch-album-cache")
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
    help="Optional cap on number of albums to fetch.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Concurrent worker threads for API requests.",
)
def fetch_album_cache_cmd(
    db_path: Path, limit: int | None, cache_dir: Path | None, workers: int
) -> None:
    """Fetch album JSONs from Spotify into the local cache.

    Run this after `ingest-track-albums` and before
    `ingest-tracks-from-albums`, which reads the cache this command fills.
    """
    counts = fetch_album_cache(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
        max_workers=workers,
    )
    click.echo(
        f"Albums in DB: {counts['albums_in_db']} | "
        f"Already cached: {counts['already_cached']} | Fetched: {counts['fetched']}"
    )


@main.command("fetch-artist-cache")
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
    help="Optional cap on number of artists to fetch.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cache directory (defaults env SPOTIFY_API_CACHE_DIR or data/api/cache).",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Concurrent worker threads for API requests.",
)
def fetch_artist_cache_cmd(
    db_path: Path, limit: int | None, cache_dir: Path | None, workers: int
) -> None:
    """Fetch artist JSONs from Spotify into the local cache.

    Run this after `ingest-track-artists` and before `ingest-artist-genres`,
    which reads the cache this command fills.
    """
    counts = fetch_artist_cache(
        db_path=db_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        limit=limit,
        max_workers=workers,
    )
    click.echo(
        f"Artists in DB: {counts['artists_in_db']} | "
        f"Already cached: {counts['already_cached']} | Fetched: {counts['fetched']}"
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


@main.command("status")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=Path("data/db/music.db"),
    show_default=True,
    help="Path to DuckDB database file.",
)
def status_cmd(db_path: Path) -> None:
    """Report what is populated in the database and what is missing.

    Exits non-zero when something needs attention, so it can gate a pipeline.
    """
    try:
        report = collect_status(db_path)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    width = max(len(c.name) for c in report.checks)
    click.echo(f"Database: {report.db_path}\n")
    for check in report.checks:
        mark = "ok  " if check.ok else "WARN"
        click.echo(f"  [{mark}] {check.name.ljust(width)}  {check.value}")

    if report.ok:
        click.echo("\nAll checks passed.")
        return

    click.echo("\nNeeds attention:")
    for check in report.problems:
        click.echo(f"  - {check.name}: {check.hint or check.value}")
    raise SystemExit(1)
