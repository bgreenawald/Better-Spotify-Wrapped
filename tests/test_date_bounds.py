"""Tests for date-range boundary handling in the plays filter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.metrics.utils import inclusive_end_bound
from src.preprocessing import get_filtered_plays

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


@pytest.fixture
def con(tmp_path: Path):
    """A DuckDB connection with the real schema and a handful of plays."""
    db = tmp_path / "t.db"
    c = duckdb.connect(str(db))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))

    c.execute("INSERT INTO dim_users (user_id, display_name, country) VALUES ('u1', 'U', 'US')")
    c.execute(
        "INSERT INTO dim_tracks (track_id, track_name, duration_ms, explicit) "
        "VALUES ('t1', 'Song', 200000, false)"
    )
    # Plays spanning the last day of the range, including one at 23:00.
    for i, ts in enumerate(
        (
            "2024-06-15 12:00:00",
            "2024-06-30 00:00:00",
            "2024-06-30 13:00:00",
            "2024-06-30 23:00:00",
            "2024-07-01 10:00:00",
        )
    ):
        c.execute(
            "INSERT INTO fact_plays (play_id, user_id, track_id, played_at, duration_ms, "
            "reason_start, reason_end, skipped, incognito_mode) "
            "VALUES (?, 'u1', 't1', ?, 200000, 'clickrow', 'trackdone', false, false)",
            [f"p{i}", ts],
        )
    yield c
    c.close()


def test_last_day_of_range_is_included(con) -> None:
    """A midnight end bound must not drop the final day's plays."""
    df = get_filtered_plays(
        con,
        user_id="u1",
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 6, 30),
        exclude_december=False,
    )

    days = pd.to_datetime(df["ts"]).dt.date
    on_last_day = (days == pd.Timestamp("2024-06-30").date()).sum()

    # Three plays fall on Jun 30 (00:00, 13:00, 23:00); all must survive.
    assert on_last_day == 3, f"expected 3 plays on the end day, got {on_last_day}"
    assert pd.to_datetime(df["ts"]).max() == pd.Timestamp("2024-06-30 23:00:00")


def test_range_still_excludes_the_following_day(con) -> None:
    """Extending the end bound must not leak the next day in."""
    df = get_filtered_plays(
        con,
        user_id="u1",
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 6, 30),
        exclude_december=False,
    )
    assert pd.to_datetime(df["ts"]).max() < pd.Timestamp("2024-07-01")


def test_inclusive_end_bound_honors_an_explicit_time() -> None:
    """A bound with a real time of day is used as given."""
    assert inclusive_end_bound(pd.Timestamp("2024-06-30 14:30:00")) == pd.Timestamp(
        "2024-06-30 14:30:00"
    )


def test_inclusive_end_bound_extends_midnight() -> None:
    """A midnight bound is pushed to the end of that same day."""
    assert inclusive_end_bound(pd.Timestamp("2024-06-30")) == pd.Timestamp(
        "2024-06-30 23:59:59.999999"
    )


def test_inclusive_end_bound_passes_through_none() -> None:
    """An open-ended range stays open-ended."""
    assert inclusive_end_bound(None) is None
    assert inclusive_end_bound(pd.NaT) is None
