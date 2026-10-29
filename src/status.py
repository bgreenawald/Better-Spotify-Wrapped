"""Database health checks backing the `slopify status` command.

Nothing else in the project tells you whether an ingest actually worked. The
populate commands exit 0 after updating nothing, so a wrong API secret or a
missing cache looks exactly like a completed run. These checks turn that into
row counts and named gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

# Tables the schema is expected to provide. Kept in step with DDL.sql.
EXPECTED_TABLES = (
    "dim_users",
    "dim_tracks",
    "dim_albums",
    "dim_artists",
    "bridge_track_artists",
    "fact_plays",
    "dim_genres",
    "tag_evidence",
    "map_genre",
    "genre_hierarchy",
    "artist_genres",
)


@dataclass
class Check:
    """One health check result.

    Attributes:
        name: Short label.
        value: What was measured, already formatted.
        ok: False when this needs attention.
        hint: What to run or look at when `ok` is False.
    """

    name: str
    value: str
    ok: bool = True
    hint: str | None = None


@dataclass
class StatusReport:
    """The collected checks for one database.

    Attributes:
        db_path: Database that was inspected.
        checks: Results, in display order.
    """

    db_path: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return all(c.ok for c in self.checks)

    @property
    def problems(self) -> list[Check]:
        """Only the checks needing attention."""
        return [c for c in self.checks if not c.ok]


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Run a single-value count query.

    Args:
        con: Open DuckDB connection.
        sql: Query returning one numeric column.

    Returns:
        The value as an int, or 0 when the query returns no row.
    """
    row = con.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def collect_status(db_path: str | Path) -> StatusReport:
    """Inspect a database and report what is populated and what is missing.

    Args:
        db_path: Path to the DuckDB database file.

    Returns:
        The report. `report.ok` is False when anything needs attention.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}. Run `slopify init-db` first.")

    report = StatusReport(db_path=str(path))
    con = duckdb.connect(str(path), read_only=True)
    try:
        present = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        missing = [t for t in EXPECTED_TABLES if t not in present]
        report.checks.append(
            Check(
                name="Schema",
                value=f"{len(present)} objects"
                + (f", missing {', '.join(missing)}" if missing else ""),
                ok=not missing,
                hint="Run `slopify init-db` to create the missing objects." if missing else None,
            )
        )
        if missing:
            # Without the core tables the counts below would just raise.
            return report

        users = _scalar(con, "SELECT COUNT(*) FROM dim_users")
        plays = _scalar(con, "SELECT COUNT(*) FROM fact_plays")
        tracks = _scalar(con, "SELECT COUNT(*) FROM dim_tracks")

        report.checks.append(
            Check(
                name="Users",
                value=str(users),
                ok=users > 0,
                hint="No users. Run `slopify ingest-history`." if users == 0 else None,
            )
        )
        report.checks.append(
            Check(
                name="Plays",
                value=f"{plays:,}",
                ok=plays > 0,
                hint="No listening history. Run `slopify ingest-history`." if plays == 0 else None,
            )
        )
        report.checks.append(Check(name="Tracks", value=f"{tracks:,}"))

        if plays:
            span = con.execute("SELECT MIN(played_at), MAX(played_at) FROM fact_plays").fetchone()
            report.checks.append(
                Check(name="Date range", value=f"{span[0]:%Y-%m-%d} to {span[1]:%Y-%m-%d}")
            )

        # Referential integrity: a play pointing at a track we never stored.
        orphans = _scalar(
            con,
            "SELECT COUNT(*) FROM fact_plays p "
            "LEFT JOIN dim_tracks t ON t.track_id = p.track_id "
            "WHERE t.track_id IS NULL",
        )
        report.checks.append(
            Check(
                name="Orphan plays",
                value=f"{orphans:,}",
                ok=orphans == 0,
                hint="Plays reference tracks absent from dim_tracks; re-run ingest-history."
                if orphans
                else None,
            )
        )

        # Enrichment coverage. Each gap names the command that fills it.
        for label, sql, hint in (
            (
                "Tracks missing duration",
                "SELECT COUNT(*) FROM dim_tracks WHERE duration_ms IS NULL",
                "slopify ingest-track-metadata",
            ),
            (
                "Tracks missing ISRC",
                "SELECT COUNT(*) FROM dim_tracks WHERE track_isrc IS NULL",
                "slopify ingest-track-metadata",
            ),
            (
                "Tracks missing album",
                "SELECT COUNT(*) FROM dim_tracks WHERE album_id IS NULL",
                "slopify ingest-track-albums",
            ),
            (
                "Tracks missing primary artist",
                "SELECT COUNT(*) FROM dim_tracks t WHERE NOT EXISTS ("
                "  SELECT 1 FROM bridge_track_artists b"
                "  WHERE b.track_id = t.track_id AND b.role = 'primary')",
                "slopify ingest-track-artists",
            ),
        ):
            n = _scalar(con, sql)
            pct = (n / tracks * 100) if tracks else 0.0
            report.checks.append(
                Check(
                    name=label,
                    value=f"{n:,} ({pct:.1f}%)",
                    # Enrichment is incremental and some ids never resolve, so
                    # only a wholesale gap is worth flagging.
                    ok=not (tracks and pct > 50.0),
                    hint=f"Run `{hint}`." if tracks and pct > 50.0 else None,
                )
            )

        genres = _scalar(con, "SELECT COUNT(*) FROM dim_genres")
        linked = _scalar(con, "SELECT COUNT(*) FROM artist_genres")
        evidence = _scalar(con, "SELECT COUNT(*) FROM tag_evidence")
        artists = _scalar(con, "SELECT COUNT(*) FROM dim_artists")

        report.checks.append(Check(name="Artists", value=f"{artists:,}"))
        report.checks.append(
            Check(
                name="Genre taxonomy",
                value=f"{genres:,} genres",
                ok=genres > 0,
                hint="No genres. Load the taxonomy, then `slopify link-artist-genres`."
                if genres == 0
                else None,
            )
        )
        report.checks.append(
            Check(
                name="Genre evidence",
                value=f"{evidence:,} tags",
                ok=evidence > 0 or artists == 0,
                hint="No tags. Run `slopify fetch-artist-cache` then `ingest-artist-genres`."
                if evidence == 0 and artists
                else None,
            )
        )
        report.checks.append(
            Check(
                name="Artists linked to genres",
                value=f"{linked:,}",
                ok=linked > 0 or artists == 0,
                hint="No links. Run `slopify link-artist-genres`."
                if linked == 0 and artists
                else None,
            )
        )
        return report
    finally:
        con.close()
