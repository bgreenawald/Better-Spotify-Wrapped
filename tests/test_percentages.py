"""Tests that 'percentage' columns are shares on a consistent 0-100 scale."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.metrics.metrics import (
    get_most_played_artists,
    get_most_played_tracks,
    get_top_artist_genres,
)

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


@pytest.fixture
def con(tmp_path: Path):
    """A DuckDB database where one artist carries three genre tags.

    The multi-tag artist is the whole point: counting a play once per tag is
    what used to make the percentage column sum to far more than 100%.
    """
    c = duckdb.connect(str(tmp_path / "t.db"))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))

    c.execute("INSERT INTO dim_artists VALUES ('ar1', 'Multi Tagged'), ('ar2', 'Single Tagged')")
    c.execute(
        "INSERT INTO dim_tracks (track_id, track_name, duration_ms, explicit) "
        "VALUES ('t1', 'Song One', 200000, false), ('t2', 'Song Two', 200000, false)"
    )
    c.execute(
        "INSERT INTO bridge_track_artists (track_id, artist_id, role) "
        "VALUES ('t1', 'ar1', 'primary'), ('t2', 'ar2', 'primary')"
    )
    for gid, name in ((1, "Pop"), (2, "Rock"), (3, "Edm"), (4, "Jazz")):
        c.execute(
            "INSERT INTO dim_genres (genre_id, slug, name, level, active) VALUES (?, ?, ?, 1, true)",
            [gid, name.lower(), name],
        )
    # ar1 is tagged with three genres, ar2 with one.
    c.execute(
        "INSERT INTO artist_genres (artist_id, genre_id) "
        "VALUES ('ar1', 1), ('ar1', 2), ('ar1', 3), ('ar2', 4)"
    )
    yield c
    c.close()


def _plays() -> pd.DataFrame:
    """Five plays of t1 and three of t2, in the pandas shape metrics expect."""
    rows = [("t1", "Song One", "Multi Tagged")] * 5 + [("t2", "Song Two", "Single Tagged")] * 3
    return pd.DataFrame(
        {
            "track_id": [r[0] for r in rows],
            "spotify_track_uri": [f"spotify:track:{r[0]}" for r in rows],
            "master_metadata_track_name": [r[1] for r in rows],
            "master_metadata_album_artist_name": [r[2] for r in rows],
            "master_metadata_album_album_name": ["Album"] * len(rows),
            "artist_id": ["ar1"] * 5 + ["ar2"] * 3,
            "artist_genres": [()] * len(rows),
            "ms_played": [200000] * len(rows),
        }
    )


def test_genre_percentages_sum_to_100(con) -> None:
    """Genre shares must sum to 100%, not to the tag-inflated total."""
    out = get_top_artist_genres(_plays(), con=con)

    assert not out.empty
    assert out["percentage"].sum() == pytest.approx(100.0, abs=0.5)


def test_genre_play_count_is_a_tag_attribution_count(con) -> None:
    """play_count stays per-tag, so it legitimately exceeds the play total."""
    out = get_top_artist_genres(_plays(), con=con)

    # 5 plays x 3 tags + 3 plays x 1 tag = 18 attributions over 8 plays.
    assert int(out["play_count"].sum()) == 18
    assert set(out["genre"]) == {"Pop", "Rock", "Edm", "Jazz"}


def test_track_percentages_are_on_a_0_100_scale(con) -> None:
    """Tracks must not report a 0-1 fraction while everything else uses 0-100."""
    out = get_most_played_tracks(_plays(), con=con)

    assert out["percentage"].sum() == pytest.approx(100.0, abs=0.5)
    top = out.sort_values("play_count", ascending=False).iloc[0]
    assert top["percentage"] == pytest.approx(62.5, abs=0.5)  # 5 of 8 plays


def test_artist_percentages_are_on_a_0_100_scale(con) -> None:
    """Artists already used 0-100; assert it so the scales stay aligned."""
    out = get_most_played_artists(_plays(), con=con)
    assert out["percentage"].sum() == pytest.approx(100.0, abs=0.5)
