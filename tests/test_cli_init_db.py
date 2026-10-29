"""Tests for `slopify init-db` schema application."""

from __future__ import annotations

from pathlib import Path

import duckdb
from click.testing import CliRunner

from src.cli import main

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


def test_init_db_creates_schema(tmp_path: Path) -> None:
    """A fresh database gets every object declared in DDL.sql."""
    db = tmp_path / "fresh.db"
    result = CliRunner().invoke(main, ["init-db", "--db", str(db), "--ddl", str(DDL_PATH)])

    assert result.exit_code == 0, result.output
    assert db.exists()

    conn = duckdb.connect(str(db))
    try:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    finally:
        conn.close()

    # The semicolon-split bug used to abort partway through the file, so assert
    # on tables declared near the start, middle and end of DDL.sql.
    for expected in ("dim_users", "fact_plays", "dim_genres", "tag_evidence"):
        assert expected in tables


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    """Re-running init-db on an existing database succeeds unchanged."""
    db = tmp_path / "fresh.db"
    runner = CliRunner()
    args = ["init-db", "--db", str(db), "--ddl", str(DDL_PATH)]

    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.output

    second = runner.invoke(main, args)
    assert second.exit_code == 0, second.output


def test_init_db_rejects_missing_ddl(tmp_path: Path) -> None:
    """A missing DDL file is reported rather than silently ignored."""
    result = CliRunner().invoke(
        main,
        ["init-db", "--db", str(tmp_path / "x.db"), "--ddl", str(tmp_path / "nope.sql")],
    )
    assert result.exit_code != 0


# The schema a fresh init-db produces. Every name here is written and read by
# code; the mood tables, the agg_* rollups and track_genres were removed in
# 2026-10 as unreachable.
EXPECTED_OBJECTS = {
    "artist_genres",
    "bridge_track_artists",
    "dim_albums",
    "dim_artists",
    "dim_genres",
    "dim_tracks",
    "dim_users",
    "fact_plays",
    "genre_hierarchy",
    "map_genre",
    "tag_evidence",
    "v_plays_enriched",
    "v_primary_artist_per_track",
}


def test_schema_contains_exactly_the_live_objects(tmp_path: Path) -> None:
    """DDL.sql declares no table that code does not use, and none is missing.

    This is the guard against the schema drifting back into describing a
    system that does not exist. If you add a table, add it here too -- and if
    that feels wrong because nothing reads it yet, that is the point.
    """
    db = tmp_path / "fresh.db"
    result = CliRunner().invoke(main, ["init-db", "--db", str(db), "--ddl", str(DDL_PATH)])
    assert result.exit_code == 0, result.output

    conn = duckdb.connect(str(db))
    try:
        actual = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    finally:
        conn.close()

    assert actual == EXPECTED_OBJECTS


def test_no_mood_or_rollup_tables_came_back(tmp_path: Path) -> None:
    """The deleted tables stay deleted unless someone also writes to them."""
    db = tmp_path / "fresh.db"
    CliRunner().invoke(main, ["init-db", "--db", str(db), "--ddl", str(DDL_PATH)])

    conn = duckdb.connect(str(db))
    try:
        actual = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    finally:
        conn.close()

    removed = {
        "dim_moods",
        "map_mood",
        "track_moods",
        "track_genres",
        "agg_user_genre_daily",
        "agg_user_genre_monthly",
        "agg_user_mood_daily",
        "agg_user_mood_monthly",
    }
    assert not (actual & removed)
