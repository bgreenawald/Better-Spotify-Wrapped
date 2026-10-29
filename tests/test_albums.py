"""Tests for album scoring, edition merging, and play attribution."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.metrics.albums import (
    canonical_album_name,
    cluster_album_variants,
    compute_album_score,
    is_album_sized,
    middle_half,
    normalize_album_name,
)
from src.metrics.metrics import get_top_albums

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


# --------------------------------------------------------------------------
# Name normalization and clustering
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1989 (Taylor's Version)", "1989"),
        ("Abbey Road (Remastered)", "abbey road"),
        ("Red - Deluxe Edition", "red"),
        ("CRASH (Deluxe)", "crash"),
        ("Thriller [Special Edition]", "thriller"),
        ("Album (Deluxe Edition) (Remastered)", "album"),
        ("Rumours: 35th Anniversary Edition", "rumours"),
        ("BRAT", "brat"),
        ("Love Songs", "love songs"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_album_name_strips_edition_markers(raw, expected) -> None:
    """Edition suffixes are removed; real titles are left intact."""
    assert normalize_album_name(raw) == expected


def test_normalize_album_name_keeps_distinct_titles_distinct() -> None:
    """Normalization must not collapse genuinely different albums."""
    assert normalize_album_name("Love") != normalize_album_name("Love Songs")
    assert normalize_album_name("Live") != normalize_album_name("Live in Paris")


def test_edition_variants_cluster_together() -> None:
    """A deluxe/remaster/Taylor's Version merges with its base album."""
    names = ["1989", "1989 (Taylor's Version)", "Abbey Road", "Abbey Road (Remastered)"]
    artists = ["T", "T", "B", "B"]

    clusters = cluster_album_variants(names, artists)

    assert sorted(clusters) == [[0, 1], [2, 3]]


def test_subset_names_do_not_cluster() -> None:
    """The token_set_ratio bug: a subset name scored 100 and wrongly merged.

    "Love" and "Love Songs" are different albums, as are "Live" and "Live in
    Paris". They must stay separate.
    """
    for a, b in (("Love", "Love Songs"), ("Live", "Live in Paris"), ("Fine Line", "Fine")):
        assert cluster_album_variants([a, b], ["X", "X"]) == [[0], [1]], f"{a} merged with {b}"


def test_clustering_is_order_independent() -> None:
    """Whether two editions merge must not depend on input order.

    The old implementation compared each album only against later ones, so an
    album could be consumed as an anchor before its variant was reached.
    """
    names = ["Brat and it's completely different", "BRAT", "BRAT (Deluxe)", "Pop 2"]
    artists = ["C"] * 4

    forward = cluster_album_variants(names, artists)
    reversed_names = list(reversed(names))
    backward = cluster_album_variants(reversed_names, artists)

    def as_name_sets(clusters, source):
        return sorted(sorted(source[i] for i in c) for c in clusters)

    assert as_name_sets(forward, names) == as_name_sets(backward, reversed_names)


def test_clustering_is_transitive() -> None:
    """A merges with B and B with C implies all three are one cluster."""
    names = ["Album", "Album (Deluxe)", "Album (Deluxe Edition)"]
    assert cluster_album_variants(names, ["X"] * 3) == [[0, 1, 2]]


def test_different_artists_never_cluster() -> None:
    """Album names repeat across artists; only same-artist albums merge."""
    assert cluster_album_variants(["Greatest Hits", "Greatest Hits"], ["A", "B"]) == [[0], [1]]


def test_canonical_name_prefers_the_plain_title() -> None:
    """The display name is the base title, not an edition variant."""
    assert canonical_album_name(["BRAT (Deluxe Edition)", "BRAT"]) == "BRAT"
    assert canonical_album_name([]) == ""


def test_cluster_rejects_mismatched_inputs() -> None:
    """Parallel lists of different lengths are a programming error."""
    with pytest.raises(ValueError):
        cluster_album_variants(["a", "b"], ["x"])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_middle_half_is_actually_half() -> None:
    """For n=8 the old slice returned 5 of 8 items instead of 4."""
    assert middle_half([1, 2, 3, 4, 5, 6, 7, 8]) == [3, 4, 5, 6]
    assert middle_half([5, 1, 3, 2]) == [2, 3]


def test_middle_half_handles_tiny_and_empty_inputs() -> None:
    """Short tracklists fall back to the whole list rather than emptying."""
    assert middle_half([]) == []
    assert middle_half([7]) == [7]
    assert middle_half([2, 1]) == [1, 2]


def test_album_score_is_mqpc_times_size_factor() -> None:
    """The documented formula, on a uniform album."""
    score = compute_album_score([10] * 12)

    assert score.mqpc == pytest.approx(10.0)
    assert score.t_eff == 12
    assert score.t_eff_capped == 12
    assert score.size_factor == pytest.approx(1.0 + 0.3 * (12 / 15))
    assert score.album_score == pytest.approx(score.mqpc * score.size_factor)


def test_t_eff_is_capped_at_15() -> None:
    """A 30-track album cannot buy more than the cap's worth of size factor."""
    score = compute_album_score([5] * 30)
    assert score.t_eff == 30
    assert score.t_eff_capped == 15
    assert score.size_factor == pytest.approx(1.3)


def test_unplayed_album_reports_no_effective_tracks() -> None:
    """With MQPC 0 the old threshold of 0 made every track count.

    A wholly unplayed album reported t_eff_capped == 15, which was nonsense in
    the UI even though the score was 0 either way.
    """
    score = compute_album_score([0] * 12)

    assert score.mqpc == 0.0
    assert score.album_score == 0.0
    assert score.t_eff == 0
    assert score.t_eff_capped == 0


def test_barely_sampled_album_scores_zero_by_design() -> None:
    """Playing two tracks of a twelve-track album is not album listening."""
    score = compute_album_score([0] * 10 + [30, 25])
    assert score.mqpc == 0.0
    assert score.album_score == 0.0


def test_empty_play_counts_are_safe() -> None:
    """No tracks means no score, not a ZeroDivisionError."""
    score = compute_album_score([])
    assert score.album_score == 0.0
    assert score.t_eff_capped == 0


def test_is_album_sized_excludes_singles_and_eps() -> None:
    """Releases of five or fewer tracks are not albums."""
    assert not is_album_sized(1)
    assert not is_album_sized(5)
    assert is_album_sized(6)


# --------------------------------------------------------------------------
# Attribution, end to end
# --------------------------------------------------------------------------


@pytest.fixture
def con(tmp_path: Path):
    """Two albums by one artist that share a track name.

    `shared` appears on both the studio album and a greatest-hits
    compilation, under different track ids. Only the studio album's copy is
    ever played.
    """
    c = duckdb.connect(str(tmp_path / "a.db"))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))

    c.execute("INSERT INTO dim_artists VALUES ('ar1', 'The Artist')")
    c.execute(
        "INSERT INTO dim_albums (album_id, album_name, release_year) "
        "VALUES ('al1', 'Real Album', 2020), ('al2', 'Greatest Hits', 2024), "
        "('al3', 'Real Album (Deluxe Edition)', 2021)"
    )

    def add_track(tid: str, name: str, album: str) -> None:
        c.execute(
            "INSERT INTO dim_tracks (track_id, track_name, album_id, duration_ms, explicit) "
            "VALUES (?, ?, ?, 200000, false)",
            [tid, name, album],
        )
        c.execute(
            "INSERT INTO bridge_track_artists (track_id, artist_id, role) VALUES (?, 'ar1', ?)",
            [tid, "primary"],
        )

    # Studio album: 6 tracks, one of them named "shared".
    for i in range(5):
        add_track(f"a{i}", f"song{i}", "al1")
    add_track("a5", "shared", "al1")
    # Compilation: 6 tracks, also carrying a track named "shared".
    for i in range(5):
        add_track(f"g{i}", f"hit{i}", "al2")
    add_track("g5", "shared", "al2")
    # Deluxe edition of the studio album: same 6 plus 2 bonus tracks.
    for i in range(5):
        add_track(f"d{i}", f"song{i}", "al3")
    add_track("d5", "shared", "al3")
    add_track("d6", "bonus1", "al3")
    add_track("d7", "bonus2", "al3")

    yield c
    c.close()


def _plays(counts: dict[str, int]) -> pd.DataFrame:
    """Build a plays frame from {track_id: play_count}."""
    rows: list[str] = []
    for tid, n in counts.items():
        rows.extend([tid] * n)
    return pd.DataFrame(
        {
            "track_id": rows,
            "spotify_track_uri": [f"spotify:track:{t}" for t in rows],
            "ms_played": [200000] * len(rows),
        }
    )


def test_compilation_is_not_credited_for_the_studio_albums_plays(con) -> None:
    """Attribution is by track id, not by (track_name, artist).

    Playing the studio album's "shared" must not also credit the
    compilation's identically named track.
    """
    plays = _plays({f"a{i}": 10 for i in range(6)})

    out = get_top_albums(plays, con=con)
    by_name = dict(zip(out["album_name"], out["album_score"], strict=True))

    assert by_name.get("Real Album", 0) > 0
    assert by_name.get("Greatest Hits", 0) == 0, "compilation was credited for plays it never got"


def test_total_attributed_plays_never_exceed_actual_plays(con) -> None:
    """The old name matching inflated totals across albums."""
    plays = _plays({f"a{i}": 4 for i in range(6)} | {"g0": 3})

    out = get_top_albums(plays, con=con)
    attributed = sum(d["play_count"] for td in out["track_details"] for d in td)

    assert attributed <= len(plays)


def test_deluxe_edition_merges_with_its_base_album(con) -> None:
    """One album, not two rows, and the plain title is displayed."""
    plays = _plays({f"a{i}": 6 for i in range(6)} | {"d6": 6})

    out = get_top_albums(plays, con=con)
    names = list(out["album_name"])

    assert "Real Album" in names
    assert "Real Album (Deluxe Edition)" not in names
    # The merged album carries the deluxe tracklist: 6 base + 2 bonus.
    merged = out[out["album_name"] == "Real Album"].iloc[0]
    assert merged["total_tracks"] == 8


def test_merged_album_sums_plays_rather_than_dropping_them(con) -> None:
    """The old dedup kept the highest-scoring variant and discarded the rest."""
    plays = _plays({f"a{i}": 5 for i in range(6)} | {"d6": 7, "d7": 9})

    out = get_top_albums(plays, con=con)
    merged = out[out["album_name"] == "Real Album"].iloc[0]
    total = sum(d["play_count"] for d in merged["track_details"])

    assert total == 6 * 5 + 7 + 9


def test_empty_plays_returns_the_expected_columns(con) -> None:
    """An empty input yields an empty frame with a stable schema."""
    out = get_top_albums(pd.DataFrame({"track_id": [], "spotify_track_uri": []}), con=con)
    for col in ("album_name", "artist", "album_score", "mqpc", "track_details"):
        assert col in out.columns
