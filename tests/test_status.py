"""Tests for `slopify status` health reporting."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from src.cli import main
from src.status import collect_status

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


def _init(path: Path) -> duckdb.DuckDBPyConnection:
    """Create a database with the real schema and return an open connection."""
    c = duckdb.connect(str(path))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))
    return c


def test_missing_database_is_reported_clearly(tmp_path: Path) -> None:
    """A path that does not exist names the command that creates it."""
    with pytest.raises(FileNotFoundError, match="init-db"):
        collect_status(tmp_path / "nope.db")


def test_empty_database_is_not_reported_as_healthy(tmp_path: Path) -> None:
    """The point of the command: an ingest that did nothing must not pass."""
    db = tmp_path / "e.db"
    _init(db).close()

    report = collect_status(db)

    assert not report.ok
    names = {c.name for c in report.problems}
    assert {"Users", "Plays"} <= names
    # Every problem tells the user what to run.
    assert all(c.hint for c in report.problems)


def test_populated_database_passes(tmp_path: Path) -> None:
    """A database with history, metadata and genres reports healthy."""
    db = tmp_path / "p.db"
    c = _init(db)
    c.execute("INSERT INTO dim_users (user_id, display_name, country) VALUES ('u1','U','US')")
    c.execute("INSERT INTO dim_artists VALUES ('ar1','A')")
    c.execute(
        "INSERT INTO dim_albums (album_id, album_name, release_year) VALUES ('al1','Alb',2020)"
    )
    c.execute(
        "INSERT INTO dim_tracks (track_id, track_name, album_id, duration_ms, explicit, "
        "track_isrc) VALUES ('t1','S','al1',200000,false,'ISRC1')"
    )
    c.execute("INSERT INTO bridge_track_artists VALUES ('t1','ar1','primary')")
    c.execute(
        "INSERT INTO dim_genres (genre_id, slug, name, level, active) VALUES (1,'pop','Pop',1,true)"
    )
    c.execute("INSERT INTO artist_genres VALUES ('ar1',1)")
    c.execute(
        "INSERT INTO tag_evidence VALUES "
        "('artist','ar1','spotify','pop','genre',1.0,'2024-01-01 00:00:00')"
    )
    c.execute(
        "INSERT INTO fact_plays (play_id, user_id, track_id, played_at, duration_ms, "
        "reason_start, reason_end, skipped, incognito_mode) "
        "VALUES ('p1','u1','t1','2024-06-01 10:00:00',200000,'clickrow','trackdone',false,false)"
    )
    c.close()

    report = collect_status(db)

    assert report.ok, [f"{p.name}: {p.hint}" for p in report.problems]
    values = {c.name: c.value for c in report.checks}
    assert values["Plays"] == "1"
    assert "2024-06-01" in values["Date range"]


def test_orphan_play_check_reports_zero_on_a_sound_database(tmp_path: Path) -> None:
    """The orphan check runs and finds nothing when the data is sound.

    An orphan cannot be manufactured here: DDL.sql puts a foreign key on
    fact_plays.track_id, and DuckDB refuses to delete a referenced track. The
    check is therefore a guard against manual edits or a future schema change
    that drops the constraint, not against anything the ingest can produce.
    """
    db = tmp_path / "o.db"
    c = _init(db)
    c.execute("INSERT INTO dim_users (user_id, display_name, country) VALUES ('u1','U','US')")
    c.execute(
        "INSERT INTO dim_tracks (track_id, track_name, duration_ms, explicit) "
        "VALUES ('t1','S',200000,false)"
    )
    c.execute(
        "INSERT INTO fact_plays (play_id, user_id, track_id, played_at, duration_ms, "
        "reason_start, reason_end, skipped, incognito_mode) "
        "VALUES ('p1','u1','t1','2024-06-01 10:00:00',200000,'clickrow','trackdone',false,false)"
    )
    with pytest.raises(duckdb.ConstraintException):
        c.execute("DELETE FROM dim_tracks WHERE track_id = 't1'")
    c.close()

    orphan = next(c for c in collect_status(db).checks if c.name == "Orphan plays")

    assert orphan.value == "0"
    assert orphan.ok


def test_missing_schema_is_flagged_without_crashing(tmp_path: Path) -> None:
    """A database with no schema reports the gap rather than raising."""
    db = tmp_path / "bare.db"
    duckdb.connect(str(db)).close()

    report = collect_status(db)

    assert not report.ok
    assert report.checks[0].name == "Schema"
    assert "init-db" in (report.checks[0].hint or "")


def test_cli_exits_nonzero_on_an_empty_database(tmp_path: Path) -> None:
    """So the command can gate a pipeline."""
    db = tmp_path / "e.db"
    _init(db).close()

    result = CliRunner().invoke(main, ["status", "--db", str(db)])

    assert result.exit_code == 1
    assert "Needs attention" in result.output


def test_cli_exits_zero_on_a_healthy_database(tmp_path: Path) -> None:
    """A populated database reports success."""
    db = tmp_path / "p.db"
    c = _init(db)
    c.execute("INSERT INTO dim_users (user_id, display_name, country) VALUES ('u1','U','US')")
    c.execute(
        "INSERT INTO dim_genres (genre_id, slug, name, level, active) VALUES (1,'pop','Pop',1,true)"
    )
    c.execute(
        "INSERT INTO dim_tracks (track_id, track_name, duration_ms, explicit, track_isrc, "
        "album_id) VALUES ('t1','S',200000,false,'I','al1')"
    )
    c.execute(
        "INSERT INTO dim_albums (album_id, album_name, release_year) VALUES ('al1','Alb',2020)"
    )
    c.execute("INSERT INTO dim_artists VALUES ('ar1','A')")
    c.execute("INSERT INTO bridge_track_artists VALUES ('t1','ar1','primary')")
    c.execute("INSERT INTO artist_genres VALUES ('ar1',1)")
    c.execute(
        "INSERT INTO tag_evidence VALUES "
        "('artist','ar1','spotify','pop','genre',1.0,'2024-01-01 00:00:00')"
    )
    c.execute(
        "INSERT INTO fact_plays (play_id, user_id, track_id, played_at, duration_ms, "
        "reason_start, reason_end, skipped, incognito_mode) "
        "VALUES ('p1','u1','t1','2024-06-01 10:00:00',200000,'clickrow','trackdone',false,false)"
    )
    c.close()

    result = CliRunner().invoke(main, ["status", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert "All checks passed" in result.output
