from typing import Any


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
