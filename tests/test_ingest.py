"""Tests for the listening-history ingest pipeline.

This pipeline produces the data everything else depends on and previously had
no tests at all, which is why the init-db and cache defects went unnoticed.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.db_ingest import (
    _extract_spotify_track_id,
    _to_timestamp,
    load_history_into_fact_plays,
)
from src.io import load_spotify_history

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


# --------------------------------------------------------------------------
# Unit: timestamp and id parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, pd.NaT, float("nan")])
def test_to_timestamp_rejects_missing_values(value) -> None:
    """pd.NaT used to come back as the string "NaT".

    NaT subclasses datetime.datetime but is not a pd.Timestamp, so it reached
    the datetime branch, stringified, passed the caller's .notna() guard and
    then failed the insert -- rolling back the whole ingest.
    """
    assert _to_timestamp(value) is None


def test_to_timestamp_normalizes_to_naive_utc() -> None:
    """Aware inputs are converted to UTC and stripped of tzinfo."""
    assert _to_timestamp(pd.Timestamp("2024-06-01 10:00:00")) == "2024-06-01 10:00:00"
    assert _to_timestamp(pd.Timestamp("2024-06-01 12:00:00+02:00")) == "2024-06-01 10:00:00"
    assert _to_timestamp("2024-06-01T10:00:00Z") == "2024-06-01 10:00:00"
    aware = dt.datetime(2024, 6, 1, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert _to_timestamp(aware) == "2024-06-01 10:00:00"


def test_to_timestamp_rejects_garbage() -> None:
    """An unparseable string is dropped, not guessed at."""
    assert _to_timestamp("not a date") is None
    assert _to_timestamp(object()) is None


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("spotify:track:abc123", "abc123"),
        ("https://open.spotify.com/track/xyz789", "xyz789"),
        ("https://open.spotify.com/track/xyz789?si=deadbeef", "xyz789"),
        (None, None),
        ("", None),
        ("spotify:episode:pod1", None),
        ("garbage", None),
    ],
)
def test_extract_spotify_track_id(uri, expected) -> None:
    """Both URI forms parse; anything else is None."""
    assert _extract_spotify_track_id(uri) == expected


def test_extract_spotify_track_id_handles_nan() -> None:
    """NaN is truthy, so `if not uri` let it reach .startswith()."""
    assert _extract_spotify_track_id(float("nan")) is None


# --------------------------------------------------------------------------
# io.load_spotify_history
# --------------------------------------------------------------------------


def _write(d: Path, name: str, records: list[dict]) -> None:
    (d / name).write_text(json.dumps(records), encoding="utf-8")


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    """A typo'd path used to look like "you have no listening history"."""
    with pytest.raises(FileNotFoundError):
        load_spotify_history(str(tmp_path / "nope"))


def test_empty_directory_yields_usable_columns(tmp_path: Path) -> None:
    """An empty frame must still have the columns the ingest reads."""
    df = load_spotify_history(str(tmp_path))
    assert df.empty
    for col in ("ts", "ms_played", "spotify_track_uri"):
        assert col in df.columns


def test_video_history_files_are_skipped(tmp_path: Path) -> None:
    """Spotify ships video history in the same folder; it is a different shape."""
    _write(
        tmp_path,
        "Streaming_History_Audio_2024.json",
        [{"ts": "2024-06-01T10:00:00Z", "spotify_track_uri": "spotify:track:a"}],
    )
    _write(
        tmp_path,
        "Streaming_History_Video_2024.json",
        [{"ts": "2024-06-02T10:00:00Z", "episode_name": "A show"}],
    )

    df = load_spotify_history(str(tmp_path))

    assert len(df) == 1
    assert df["spotify_track_uri"].iloc[0] == "spotify:track:a"


def test_mixed_offsets_stay_datetime(tmp_path: Path) -> None:
    """Without utc=True a mixed export parses to object dtype and .dt raises."""
    _write(
        tmp_path,
        "Streaming_History_Audio_0.json",
        [
            {"ts": "2024-06-01T10:00:00Z", "spotify_track_uri": "spotify:track:a"},
            {"ts": "2024-06-01T12:00:00+02:00", "spotify_track_uri": "spotify:track:b"},
        ],
    )

    df = load_spotify_history(str(tmp_path))

    assert str(df["ts"].dtype).startswith("datetime64")
    # Both are the same instant in UTC.
    assert df["ts"].nunique() == 1


def test_non_list_json_is_rejected_by_name(tmp_path: Path) -> None:
    """The error must name the file; an export holds dozens."""
    _write(tmp_path, "Streaming_History_Audio_0.json", [{"ts": "2024-06-01T10:00:00Z"}])
    (tmp_path / "Streaming_History_Audio_1.json").write_text('{"not": "a list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Streaming_History_Audio_1.json"):
        load_spotify_history(str(tmp_path))


def test_invalid_json_is_rejected_by_name(tmp_path: Path) -> None:
    """Likewise for a truncated file."""
    (tmp_path / "Streaming_History_Audio_0.json").write_text("{trunc", encoding="utf-8")

    with pytest.raises(ValueError, match="Streaming_History_Audio_0.json"):
        load_spotify_history(str(tmp_path))


# --------------------------------------------------------------------------
# load_history_into_fact_plays
# --------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """An initialized, empty database."""
    path = tmp_path / "m.db"
    c = duckdb.connect(str(path))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))
    c.close()
    return path


def _records() -> list[dict]:
    """Three good plays, one podcast episode, one unparseable, one NaN URI."""
    return [
        {
            "ts": "2024-06-01T10:00:00Z",
            "ms_played": 200000,
            "spotify_track_uri": "spotify:track:t1",
            "master_metadata_track_name": "Song One",
            "master_metadata_album_artist_name": "Artist",
            "master_metadata_album_album_name": "Album",
            "reason_start": "clickrow",
            "reason_end": "trackdone",
            "skipped": False,
            "incognito_mode": False,
        },
        {
            "ts": "2024-06-01T11:00:00Z",
            "ms_played": 180000,
            "spotify_track_uri": "spotify:track:t2",
            "master_metadata_track_name": "Song Two",
            "master_metadata_album_artist_name": "Artist",
            "reason_start": "clickrow",
            "reason_end": "trackdone",
        },
        {
            "ts": "2024-06-02T09:00:00Z",
            "ms_played": 210000,
            "spotify_track_uri": "spotify:track:t1",
            "master_metadata_track_name": "Song One",
            "master_metadata_album_artist_name": "Artist",
            "reason_start": "clickrow",
            "reason_end": "trackdone",
        },
        {
            "ts": "2024-06-03T09:00:00Z",
            "ms_played": 60000,
            "spotify_track_uri": None,
            "spotify_episode_uri": "spotify:episode:e1",
            "episode_name": "An Episode",
        },
        {
            "ts": "not-a-timestamp",
            "ms_played": 1000,
            "spotify_track_uri": "spotify:track:t3",
            "master_metadata_track_name": "Bad Ts",
        },
        {
            "ts": "2024-06-04T09:00:00Z",
            "ms_played": 1000,
            "spotify_track_uri": float("nan"),
            "master_metadata_track_name": "No Uri",
        },
    ]


def _ingest(db: Path, records: list[dict]):
    return load_history_into_fact_plays(db_path=db, user_id="u1", history_df=pd.DataFrame(records))


def test_ingest_inserts_only_usable_track_plays(db: Path) -> None:
    """Good plays land; episodes and unusable rows are excluded and counted."""
    res = _ingest(db, _records())

    assert res.inserted_plays == 3
    assert res.inserted_tracks == 2  # t1 appears twice
    assert res.skipped_episodes >= 1
    assert res.skipped_rows >= 1, "the bad timestamp and NaN URI must be reported"

    c = duckdb.connect(str(db), read_only=True)
    try:
        assert c.execute("SELECT COUNT(*) FROM fact_plays").fetchone()[0] == 3
        assert c.execute("SELECT COUNT(*) FROM dim_tracks").fetchone()[0] == 2
        assert c.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0] == 1
    finally:
        c.close()


def test_reingest_is_idempotent(db: Path) -> None:
    """The README's idempotency claim, asserted."""
    first = _ingest(db, _records())
    second = _ingest(db, _records())

    assert first.inserted_plays == 3
    assert second.inserted_plays == 0
    assert second.deduped_plays == 3

    c = duckdb.connect(str(db), read_only=True)
    try:
        assert c.execute("SELECT COUNT(*) FROM fact_plays").fetchone()[0] == 3
    finally:
        c.close()


def test_a_bad_row_does_not_roll_back_the_good_ones(db: Path) -> None:
    """The NaT bug failed the single transaction, losing every valid row."""
    records = _records()
    records.append(
        {
            "ts": pd.NaT,
            "ms_played": 5,
            "spotify_track_uri": "spotify:track:t9",
            "master_metadata_track_name": "NaT Row",
        }
    )

    res = _ingest(db, records)

    assert res.inserted_plays == 3
    c = duckdb.connect(str(db), read_only=True)
    try:
        assert c.execute("SELECT COUNT(*) FROM fact_plays").fetchone()[0] == 3
    finally:
        c.close()


def test_ingest_of_nothing_usable_inserts_nothing(db: Path) -> None:
    """An export with no usable rows must not raise."""
    res = _ingest(
        db,
        [{"ts": "nope", "ms_played": 1, "spotify_track_uri": None}],
    )
    assert res.inserted_plays == 0


def test_plays_are_stored_as_naive_utc(db: Path) -> None:
    """A +02:00 timestamp is stored as the equivalent UTC instant."""
    _ingest(
        db,
        [
            {
                "ts": "2024-06-01T12:00:00+02:00",
                "ms_played": 200000,
                "spotify_track_uri": "spotify:track:tz",
                "master_metadata_track_name": "TZ",
                "reason_start": "clickrow",
                "reason_end": "trackdone",
            }
        ],
    )

    c = duckdb.connect(str(db), read_only=True)
    try:
        got = c.execute("SELECT played_at FROM fact_plays").fetchone()[0]
    finally:
        c.close()

    assert got == dt.datetime(2024, 6, 1, 10, 0, 0)
