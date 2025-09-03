import re
from typing import Any


def extract_track_id(uri: Any) -> str | None:
    """Extract track ID from various Spotify URI formats.

    Args:
        uri (Any): The URI string to parse. Should be a string.

    Returns:
        str | None: The extracted track ID, or None if not a valid Spotify URI.
    """
    if not isinstance(uri, str):
        return None
    uri = uri.strip()

    m = re.search(r"\bspotify:track:([A-Za-z0-9]{22})\b", uri)
    if m:
        return m.group(1)
    # Handle Spotify URL format
    if "open.spotify.com/track/" in uri:
        part = uri.split("open.spotify.com/track/")[-1]
        id_part = part.split("?")[0].split("/")[0]
        if len(id_part) == 22 and re.match(r"[A-Za-z0-9]{22}", id_part):
            return id_part
    return None
