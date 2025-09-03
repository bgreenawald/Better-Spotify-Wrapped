from typing import Any


def extract_track_id(uri: Any) -> str | None:
    """Extract track ID from various Spotify URI formats.

    Args:
        uri (Any): The URI string to parse. Should be a string.

    Returns:
        str | None: The extracted track ID, or None if not a valid Spotify URI.
    """
++ b/src/metrics/utils.py
@@
from typing import Any
import re
@@ def parse_spotify_track_id(uri: Any) -> Optional[str]:
-    if not isinstance(uri, str):
    if not isinstance(uri, str):
        return None
    uri = uri.strip()
@@
-    if ":" in uri:
-        return uri.split(":")[-1]
    # Strictly accept only 22-char base62 Spotify track IDs
    m = re.search(r'\bspotify:track:([A-Za-z0-9]{22})\b', uri)
    if m:
        return m.group(1)
    return None
