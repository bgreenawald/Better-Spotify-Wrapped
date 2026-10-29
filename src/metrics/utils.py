from typing import Any

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
