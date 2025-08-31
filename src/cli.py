"""Simple CLI for Better Spotify Wrapped tasks.

Provides an `ingest-history` command to load Spotify listening history into
the normalized DuckDB schema (see DDL.sql).
"""

from __future__ import annotations

from pathlib import Path

import click
import duckdb

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
    help="User UUID. If omitted, derived deterministically from --user-name.",
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
      bsw ingest-history --user-name egreenawald \
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


if __name__ == "__main__":
    main()
