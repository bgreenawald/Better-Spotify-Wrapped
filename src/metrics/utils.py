import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


def extract_track_id(uri: Any) -> str | None:
    """Extract a Spotify track ID from common URI/URL formats.

    Accepts both real Spotify IDs (22-char base62) and simplified IDs used in
    tests/fixtures (e.g., "spotify:track:1"). Returns the last colon segment
    for `spotify:track:*` and the path segment after `/track/` for URLs.

    Args:
        uri: The value to parse. Must be a string; returns None otherwise.

    Returns:
        The extracted track ID, or None if it cannot be parsed.
    """
    if not isinstance(uri, str):
        return None
    s = uri.strip()

    # spotify:track:<id>
    if s.startswith("spotify:track:"):
        tid = s.split(":")[-1].strip()
        return tid or None

    # https://open.spotify.com/track/<id>[?...]
    if "open.spotify.com/track/" in s:
        part = s.split("open.spotify.com/track/")[-1]
        tid = part.split("?")[0].split("/")[0].strip()
        return tid or None

    # Fallback: last colon-delimited segment if present (e.g., "uri:1")
    if ":" in s:
        tid = s.split(":")[-1].strip()
        return tid or None

    return None


def inclusive_end_bound(end: Any) -> Any:
    """Normalize a range end so the whole final day is included.

    Callers pass a date-only or midnight timestamp to mean "through this day"
    (a date picker value, or ``datetime(year, 12, 31)``). Comparing
    ``played_at <= midnight`` silently drops every play on that last day, so a
    midnight bound is pushed to the end of the same day. A bound that carries a
    real time of day is honored as given.

    Args:
        end: The end of the range, or None.

    Returns:
        A ``pandas.Timestamp`` covering the full day, or None if ``end`` is None.
    """
    if end is None:
        return None
    ts = pd.to_datetime(end)
    if pd.isna(ts):
        return None
    if (ts.hour, ts.minute, ts.second, ts.microsecond, getattr(ts, "nanosecond", 0)) == (
        0,
        0,
        0,
        0,
        0,
    ):
        return ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def display_timezone() -> str:
    """The timezone day/month bucketing should use for display.

    `fact_plays.played_at` is stored as naive UTC, but listening happens in
    local time: on the live data 13.6% of plays fall on a different calendar
    day once converted, and a late-evening play can land in the wrong month
    or even the wrong year. Set `DISPLAY_TIMEZONE` to an IANA name to override;
    "UTC" keeps the old behavior.

    Returns:
        An IANA timezone name, defaulting to the machine's local zone.
    """
    configured = os.getenv("DISPLAY_TIMEZONE")
    if configured:
        return configured

    # datetime.now().astimezone() yields an abbreviation like "EDT", which is
    # not an IANA name, so resolve the platform's zoneinfo link instead.
    try:
        link = os.path.realpath("/etc/localtime")
        if "zoneinfo" in link:
            name = link.split("zoneinfo/", 1)[1]
            ZoneInfo(name)
            return name
    except Exception:
        pass

    try:
        name = getattr(datetime.now().astimezone().tzinfo, "key", None)
        if name:
            ZoneInfo(name)
            return name
    except Exception:
        pass
    return "UTC"


def local_ts_sql(column: str, tz: str | None = None) -> str:
    """SQL converting a naive-UTC timestamp column to display-local time.

    Args:
        column: The qualified column expression, e.g. "p.played_at".
        tz: IANA timezone name; defaults to `display_timezone()`.

    Returns:
        A SQL expression yielding a naive local timestamp. When the zone is
        UTC the column is returned untouched, so no ICU work is done.
    """
    zone = tz or display_timezone()
    if zone.upper() == "UTC":
        return column
    return f"(({column}) AT TIME ZONE 'UTC' AT TIME ZONE '{_quote_tz(zone)}')"


def _quote_tz(zone: str) -> str:
    """Reject anything that is not a plain IANA zone name.

    The zone is interpolated into SQL, so it must not carry a quote. It comes
    from the environment rather than from user input, but validating keeps the
    interpolation obviously safe.

    Args:
        zone: Candidate timezone name.

    Returns:
        The zone, unchanged.

    Raises:
        ValueError: If the name is not a valid IANA zone.
    """
    try:
        ZoneInfo(zone)
    except Exception as exc:
        raise ValueError(f"Invalid DISPLAY_TIMEZONE {zone!r}") from exc
    return zone


def table_exists(con: Any, name: str) -> bool:
    """Whether a table or view is present in the connected database.

    Several metrics can use the normalized schema when it is there (to key
    entities by ISRC, say) but must still work against a bare connection
    holding only a registered DataFrame.

    Args:
        con: DuckDB connection.
        name: Unqualified table or view name.

    Returns:
        True when the object exists and can be queried.
    """
    try:
        row = con.execute(
            "SELECT 1 FROM duckdb_tables() WHERE LOWER(table_name) = LOWER(?) "
            "UNION ALL SELECT 1 FROM duckdb_views() WHERE LOWER(view_name) = LOWER(?) LIMIT 1",
            [name, name],
        ).fetchone()
        return row is not None
    except Exception:
        return False


def column_exists(con: Any, table: str, column: str) -> bool:
    """Whether a table exists and has the given column.

    Checking the table alone is not enough: a caller may attach a reduced
    `dim_tracks` that lacks `track_isrc`, and referencing a missing column
    fails at bind time.

    Args:
        con: DuckDB connection.
        table: Unqualified table or view name.
        column: Column name to look for.

    Returns:
        True when both the object and the column are present.
    """
    try:
        row = con.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE LOWER(table_name) = LOWER(?) AND LOWER(column_name) = LOWER(?) LIMIT 1",
            [table, column],
        ).fetchone()
        return row is not None
    except Exception:
        return False
