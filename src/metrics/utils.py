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
    if uri.startswith("spotify:track:"):
        return uri.split(":")[-1]
    if "open.spotify.com/track/" in uri:
        part = uri.split("open.spotify.com/track/")[-1]
        return part.split("?")[0]
    if ":" in uri:
        return uri.split(":")[-1]
    return None
