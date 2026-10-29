"""Pure album scoring and edition-merging helpers.

These are deliberately free of DuckDB and pandas so they can be unit tested
directly. `get_top_albums` in `metrics.py` supplies the data and formats the
result; the arithmetic and the name matching live here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

# Words that mark a release as a variant of another album rather than its own
# album. Matched inside a trailing bracket or after a trailing dash/colon.
_EDITION_WORDS = (
    "deluxe",
    "super deluxe",
    "expanded",
    "extended",
    "remaster",
    "remastered",
    "remasters",
    "reissue",
    "edition",
    "version",
    "anniversary",
    "bonus",
    "bonus track",
    "bonus tracks",
    "special",
    "collector",
    "collectors",
    "explicit",
    "clean",
    "mono",
    "stereo",
    "reimagined",
    "revisited",
    "complete",
)

# A trailing (...) or [...] group that contains an edition word.
_BRACKET_EDITION = re.compile(
    r"\s*[\(\[][^\)\]]*\b(?:" + "|".join(map(re.escape, _EDITION_WORDS)) + r")\b[^\)\]]*[\)\]]\s*$",
    re.IGNORECASE,
)
# A trailing " - Deluxe Edition" / " : Remastered" style suffix.
_DASH_EDITION = re.compile(
    r"\s*[-–—:]\s*[^-–—:]*\b(?:" + "|".join(map(re.escape, _EDITION_WORDS)) + r")\b[^-–—:]*$",
    re.IGNORECASE,
)
# "Taylor's Version" and friends: a possessive followed by "version".
_POSSESSIVE_VERSION = re.compile(r"\s*[\(\[][^\)\]]*'s\s+version[^\)\]]*[\)\]]\s*$", re.IGNORECASE)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_album_name(name: str | None) -> str:
    """Reduce an album name to a comparable base form.

    Strips edition markers so that variants of one album collapse together:
    "1989 (Taylor's Version)", "Abbey Road (Remastered)" and
    "Red - Deluxe Edition" reduce to "1989", "abbey road" and "red".

    Args:
        name: Raw album name, possibly None.

    Returns:
        A lowercase, punctuation-free base name ("" when `name` is empty).
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name))
    # Curly to straight quotes, so "it's" matches "it's".
    s = s.replace("’", "'").replace("‘", "'")

    # Peel repeatedly: "X (Deluxe Edition) (Remastered)" needs two passes.
    for _ in range(4):
        before = s
        s = _POSSESSIVE_VERSION.sub("", s)
        s = _BRACKET_EDITION.sub("", s)
        s = _DASH_EDITION.sub("", s)
        if s == before:
            break

    s = _PUNCT.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


@dataclass(frozen=True)
class AlbumScore:
    """The album-score breakdown for one album.

    Attributes:
        mqpc: Mean of the middle 50% of track play counts.
        t_eff: Tracks clearing half the MQPC (or any play when MQPC is 0).
        t_eff_capped: `t_eff` capped at 15.
        size_factor: 1 + 0.3 * (t_eff_capped / 15).
        album_score: mqpc * size_factor.
    """

    mqpc: float
    t_eff: int
    t_eff_capped: int
    size_factor: float
    album_score: float


def middle_half(play_counts: list[int]) -> list[int]:
    """Return the middle 50% of a play-count list, sorted ascending.

    The old inline version used ceil(0.25n)-1 .. floor(0.75n)-1, which for
    n=8 returned 5 of 8 items rather than half. This trims floor(n/4) from
    each end, so the slice is as close to half as an integer count allows and
    is symmetric.

    Args:
        play_counts: Per-track play counts; order does not matter.

    Returns:
        The middle slice, ascending. Never empty for a non-empty input.
    """
    if not play_counts:
        return []
    ordered = sorted(play_counts)
    n = len(ordered)
    trim = n // 4
    middle = ordered[trim : n - trim] if n - trim > trim else ordered
    return middle or ordered


def compute_album_score(play_counts: list[int]) -> AlbumScore:
    """Score an album from its per-track play counts.

    The score rewards albums listened to as albums: a high median-ish play
    count across the tracklist, nudged up by how much of the album clears
    half that level. An album where most tracks went unplayed scores 0 by
    design -- that is the metric saying "you did not listen to this album",
    not a defect.

    Args:
        play_counts: Per-track play counts, one entry per distinct track.

    Returns:
        The score breakdown.
    """
    if not play_counts:
        return AlbumScore(0.0, 0, 0, 1.0, 0.0)

    middle = middle_half(play_counts)
    mqpc = sum(middle) / len(middle) if middle else 0.0

    if mqpc > 0:
        threshold = 0.5 * mqpc
        t_eff = sum(1 for pc in play_counts if pc >= threshold)
    else:
        # With mqpc == 0 the old threshold was 0, which every track clears --
        # so a totally unplayed album reported t_eff_capped == 15. Count only
        # tracks actually played.
        t_eff = sum(1 for pc in play_counts if pc > 0)

    t_eff_capped = min(t_eff, 15)
    size_factor = 1.0 + 0.3 * (t_eff_capped / 15.0)
    return AlbumScore(
        mqpc=mqpc,
        t_eff=t_eff,
        t_eff_capped=t_eff_capped,
        size_factor=size_factor,
        album_score=mqpc * size_factor,
    )


def cluster_album_variants(
    names: list[str],
    artists: list[str],
    *,
    similarity_threshold: float = 90.0,
) -> list[list[int]]:
    """Group indices of albums that are editions of the same album.

    Two albums cluster when they share an artist and their normalized names
    are equal, or score at least `similarity_threshold` on `fuzz.ratio`.

    `fuzz.ratio` replaces `token_set_ratio`, which returned 100 whenever one
    name's tokens were a subset of the other's -- merging "Love" with "Love
    Songs" and "Live" with "Live in Paris". Edition suffixes are handled by
    normalizing the names first rather than by a loose ratio.

    Clustering is transitive (union-find) and independent of input order. The
    previous implementation compared each album only against later ones, so
    whether two editions merged depended on arbitrary iteration order.

    Args:
        names: Album names, parallel to `artists`.
        artists: Album artists, parallel to `names`.
        similarity_threshold: Minimum `fuzz.ratio` (0-100) to merge.

    Returns:
        Clusters of indices, each sorted ascending, ordered by first index.
    """
    n = len(names)
    if n != len(artists):
        raise ValueError("names and artists must be the same length")

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    normalized = [normalize_album_name(x) for x in names]

    # Only compare within an artist; album names repeat heavily across artists.
    by_artist: dict[str, list[int]] = {}
    for i, artist in enumerate(artists):
        by_artist.setdefault(str(artist or ""), []).append(i)

    for idxs in by_artist.values():
        for pos, i in enumerate(idxs):
            for j in idxs[pos + 1 :]:
                if not normalized[i] or not normalized[j]:
                    continue
                if normalized[i] == normalized[j] or (
                    fuzz.ratio(normalized[i], normalized[j]) >= similarity_threshold
                ):
                    union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [sorted(v) for _, v in sorted(clusters.items())]


def canonical_album_name(names: list[str]) -> str:
    """Pick the name to display for a cluster of album editions.

    Prefers the shortest name, which is almost always the plain album title
    rather than an edition variant ("BRAT" over "Brat and it's the same but
    there's three more songs so it's not").

    Args:
        names: The cluster's album names.

    Returns:
        The chosen display name, or "" when `names` is empty.
    """
    candidates = [str(x) for x in names if x]
    if not candidates:
        return ""
    return min(candidates, key=lambda s: (len(s), s))


MAX_T_EFF = 15
MIN_ALBUM_TRACKS = 6  # Releases with 5 or fewer tracks are singles/EPs.


def is_album_sized(total_tracks: int) -> bool:
    """Whether a release is long enough to count as an album.

    Applied after edition merging. The old SQL applied it before, via
    HAVING COUNT(*) > 5 on unmerged rows, so an album could fall under the
    threshold after dedup and still be displayed.

    Args:
        total_tracks: Distinct tracks after merging editions.

    Returns:
        True when the release has at least `MIN_ALBUM_TRACKS` tracks.
    """
    return total_tracks >= MIN_ALBUM_TRACKS
