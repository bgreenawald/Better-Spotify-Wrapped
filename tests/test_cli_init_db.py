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
    for expected in ("dim_users", "fact_plays", "dim_genres", "agg_user_mood_monthly"):
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
