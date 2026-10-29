"""Tests for display-timezone handling of naive-UTC play timestamps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.metrics.utils import display_timezone, local_ts_sql
from src.preprocessing import get_filtered_plays

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"


def test_utc_is_passed_through_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A UTC display zone must not emit any conversion SQL."""
    monkeypatch.setenv("DISPLAY_TIMEZONE", "UTC")
    assert local_ts_sql("p.played_at") == "p.played_at"


def test_named_zone_produces_a_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """A named zone converts from UTC to that zone."""
    monkeypatch.setenv("DISPLAY_TIMEZONE", "America/New_York")
    sql = local_ts_sql("p.played_at")
    assert "AT TIME ZONE 'UTC'" in sql
    assert "America/New_York" in sql


def test_env_overrides_the_detected_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    """DISPLAY_TIMEZONE wins over whatever the host is set to."""
    monkeypatch.setenv("DISPLAY_TIMEZONE", "Asia/Tokyo")
    assert display_timezone() == "Asia/Tokyo"


def test_invalid_zone_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad zone name fails loudly rather than reaching SQL."""
    monkeypatch.setenv("DISPLAY_TIMEZONE", "Not/AZone")
    with pytest.raises(ValueError):
        local_ts_sql("p.played_at")


def test_a_zone_name_with_a_quote_cannot_reach_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zone is interpolated, so a quote must be refused."""
    monkeypatch.setenv("DISPLAY_TIMEZONE", "UTC'; DROP TABLE fact_plays; --")
    with pytest.raises(ValueError):
        local_ts_sql("p.played_at")


@pytest.fixture
def con(tmp_path: Path):
    """A database with one late-evening US play.

    2024-07-01 02:30 UTC is 2024-06-30 22:30 in America/New_York, so this one
    row lands in a different day, month and quarter depending on the zone.
    """
    c = duckdb.connect(str(tmp_path / "tz.db"))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))
    c.execute("INSERT INTO dim_users (user_id, display_name, country) VALUES ('u1','U','US')")
    c.execute(
        "INSERT INTO dim_tracks (track_id, track_name, duration_ms, explicit) "
        "VALUES ('t1','Song',200000,false)"
    )
    c.execute(
        "INSERT INTO fact_plays (play_id, user_id, track_id, played_at, duration_ms, "
        "reason_start, reason_end, skipped, incognito_mode) "
        "VALUES ('p1','u1','t1','2024-07-01 02:30:00',200000,'clickrow','trackdone',false,false)"
    )
    yield c
    c.close()


def _june(con) -> pd.DataFrame:
    return get_filtered_plays(
        con,
        user_id="u1",
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 6, 30),
        exclude_december=False,
    )


def test_late_evening_play_counts_in_the_local_month(con, monkeypatch: pytest.MonkeyPatch) -> None:
    """In US Eastern the play belongs to June, not July."""
    monkeypatch.setenv("DISPLAY_TIMEZONE", "America/New_York")

    df = _june(con)

    assert len(df) == 1
    assert pd.to_datetime(df["ts"].iloc[0]) == pd.Timestamp("2024-06-30 22:30:00")


def test_the_same_play_falls_outside_june_in_utc(con, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under UTC it is a July play, which is the old behavior."""
    monkeypatch.setenv("DISPLAY_TIMEZONE", "UTC")
    assert _june(con).empty


def test_filters_and_output_use_the_same_basis(con, monkeypatch: pytest.MonkeyPatch) -> None:
    """A play inside the range must not come back stamped outside it.

    If the filter ran on UTC while ts was converted, a row could be returned
    with a timestamp outside the requested range.
    """
    monkeypatch.setenv("DISPLAY_TIMEZONE", "America/New_York")

    df = _june(con)
    ts = pd.to_datetime(df["ts"])

    assert (ts >= pd.Timestamp("2024-06-01")).all()
    assert (ts < pd.Timestamp("2024-07-01")).all()
