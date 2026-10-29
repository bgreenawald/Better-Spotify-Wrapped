"""Tests for entity identity and rank/tie semantics.

Entities are keyed by identity (ISRC for recordings, artist_id for artists)
rather than by display name, and ranks are dense so genuine ties share a rank.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.metrics.metrics import get_most_played_artists, get_most_played_tracks

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


@pytest.fixture
def con(tmp_path: Path):
    """A database exercising every identity case at once.

    - `t_single` and `t_album` are one recording (shared ISRC) on two releases.
    - `t_intro_a` and `t_intro_b` are two different songs both called "Intro"
      by the same artist (different ISRCs).
    - `ar1` appears under an old and a new name but one artist_id.
    - `t_noisrc` has no ISRC at all.
    """
    c = duckdb.connect(str(tmp_path / "k.db"))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))
    c.execute("INSERT INTO dim_artists VALUES ('ar1','New Name'), ('ar2','Other')")

    rows = [
        ("t_single", "Hit Song", "ISRC_HIT"),
        ("t_album", "Hit Song", "ISRC_HIT"),
        ("t_intro_a", "Intro", "ISRC_INTRO_A"),
        ("t_intro_b", "Intro", "ISRC_INTRO_B"),
        ("t_noisrc", "No Isrc Song", None),
    ]
    for tid, name, isrc in rows:
        c.execute(
            "INSERT INTO dim_tracks (track_id, track_name, duration_ms, explicit, track_isrc) "
            "VALUES (?, ?, 200000, false, ?)",
            [tid, name, isrc],
        )
        c.execute(
            "INSERT INTO bridge_track_artists VALUES (?, 'ar1', 'primary')",
            [tid],
        )
    yield c
    c.close()


def _plays(spec: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """Build a plays frame from (track_id, track_name, artist_name, count)."""
    out: list[dict] = []
    for tid, name, artist, n in spec:
        for _ in range(n):
            out.append(
                {
                    "track_id": tid,
                    "spotify_track_uri": f"spotify:track:{tid}",
                    "master_metadata_track_name": name,
                    "master_metadata_album_artist_name": artist,
                    "artist_id": "ar1",
                    "ms_played": 200000,
                }
            )
    return pd.DataFrame(out)


def test_one_recording_across_two_releases_is_one_entity(con) -> None:
    """A song played from both the single and the album is one track."""
    plays = _plays(
        [("t_single", "Hit Song", "New Name", 4), ("t_album", "Hit Song", "New Name", 6)]
    )

    out = get_most_played_tracks(plays, con=con)
    hits = out[out["track_name"] == "Hit Song"]

    assert len(hits) == 1, "the two releases should share one ISRC and merge"
    assert int(hits.iloc[0]["play_count"]) == 10


def test_two_songs_sharing_a_name_stay_separate(con) -> None:
    """Keying by name merged genuinely different recordings."""
    plays = _plays([("t_intro_a", "Intro", "New Name", 5), ("t_intro_b", "Intro", "New Name", 3)])

    out = get_most_played_tracks(plays, con=con)
    intros = out[out["track_name"] == "Intro"]

    assert len(intros) == 2, "different ISRCs are different songs"
    assert sorted(intros["play_count"].astype(int)) == [3, 5]


def test_track_without_an_isrc_is_still_counted(con) -> None:
    """The 9% of tracks with no ISRC fall back to their track_id."""
    plays = _plays([("t_noisrc", "No Isrc Song", "New Name", 7)])

    out = get_most_played_tracks(plays, con=con)

    assert len(out) == 1
    assert int(out.iloc[0]["play_count"]) == 7


def test_play_counts_are_conserved(con) -> None:
    """Re-keying must neither lose nor duplicate plays."""
    plays = _plays(
        [
            ("t_single", "Hit Song", "New Name", 4),
            ("t_album", "Hit Song", "New Name", 6),
            ("t_intro_a", "Intro", "New Name", 5),
            ("t_noisrc", "No Isrc Song", "New Name", 2),
        ]
    )

    out = get_most_played_tracks(plays, con=con)

    assert int(out["play_count"].sum()) == len(plays)


def test_ties_share_a_dense_rank(con) -> None:
    """DENSE_RANK with a name tiebreaker behaved like ROW_NUMBER."""
    plays = _plays(
        [
            ("t_intro_a", "Intro", "New Name", 5),
            ("t_intro_b", "Intro", "New Name", 5),
            ("t_noisrc", "No Isrc Song", "New Name", 1),
        ]
    )

    out = get_most_played_tracks(plays, con=con)
    ranks = dict(zip(out["play_count"].astype(int), out["rank"].astype(int), strict=True))

    # Both five-play recordings rank 1; the single play ranks 2, not 3.
    assert ranks[5] == 1
    assert ranks[1] == 2


def test_renamed_artist_is_one_artist(con) -> None:
    """Grouping by display name split an artist who changed name."""
    plays = pd.concat(
        [
            _plays([("t_single", "Hit Song", "Old Name", 3)]),
            _plays([("t_album", "Hit Song", "New Name", 4)]),
        ],
        ignore_index=True,
    )

    out = get_most_played_artists(plays, con=con)

    assert len(out) == 1, "same artist_id under two names must not split"
    assert int(out.iloc[0]["play_count"]) == 7


def test_artist_unique_tracks_counts_recordings(con) -> None:
    """unique_tracks counted distinct titles, so releases double-counted."""
    plays = _plays(
        [
            ("t_single", "Hit Song", "New Name", 2),
            ("t_album", "Hit Song", "New Name", 2),
            ("t_intro_a", "Intro", "New Name", 1),
        ]
    )

    out = get_most_played_artists(plays, con=con)

    # One recording for Hit Song plus one for Intro.
    assert int(out.iloc[0]["unique_tracks"]) == 2


def test_artist_ties_share_a_dense_rank(con) -> None:
    """Artists on equal plays share a rank."""
    plays = pd.concat(
        [
            _plays([("t_single", "Hit Song", "New Name", 3)]),
            _plays([("t_intro_a", "Intro", "Other", 3)]).assign(artist_id="ar2"),
        ],
        ignore_index=True,
    )

    out = get_most_played_artists(plays, con=con)

    assert len(out) == 2
    assert set(out["rank"].astype(int)) == {1}


def test_metrics_still_work_without_the_schema() -> None:
    """A bare connection holding only the frame must not break.

    These functions were usable against an in-memory connection with no
    normalized schema; the ISRC lookup must degrade rather than fail.
    """
    bare = duckdb.connect(":memory:")
    try:
        plays = _plays([("t1", "A Song", "An Artist", 3), ("t2", "B Song", "An Artist", 1)])
        tracks = get_most_played_tracks(plays, con=bare)
        artists = get_most_played_artists(plays, con=bare)
    finally:
        bare.close()

    assert int(tracks["play_count"].sum()) == 4
    assert int(artists["play_count"].sum()) == 4
