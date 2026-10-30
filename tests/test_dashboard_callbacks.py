"""Behavioural tests for the dashboard callbacks rebuilt from duplicated code.

The date-range controls (six callbacks, two copies) and the dropdown searches
(six callbacks) were collapsed into one registration function each. These
tests pin the behaviour of the originals so the shared versions cannot drift.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import dash
import duckdb
import pandas as pd
import pytest
from dash import Dash
from dash.exceptions import PreventUpdate

from dashboard.callbacks import date_range, search, wrapped
from src.preprocessing import get_filtered_plays

DDL_PATH = Path(__file__).resolve().parents[1] / "DDL.sql"

# For registering modules whose bounds-based callbacks a test does not exercise.
_NO_BOUNDS = pd.DataFrame(
    {"ts": pd.Series(dtype="datetime64[ns]"), "user_id": pd.Series(dtype=str)}
)


def _callback(app: Dash, name: str, *, output_prefix: str):
    """The undecorated function behind a registered callback.

    Both tabs register functions with the same names, so the output id is what
    tells them apart.
    """
    for key, cb in app.callback_map.items():
        # Clientside callbacks have no server function; multi-output keys
        # start with "..", single-output ones do not.
        fn = getattr(cb.get("callback"), "__wrapped__", None)
        if fn is not None and fn.__name__ == name and key.lstrip(".").startswith(output_prefix):
            return fn
    raise LookupError(f"no callback {name} with output {output_prefix}")


# --------------------------------------------------------------------------
# Date range
# --------------------------------------------------------------------------


@pytest.fixture
def date_app() -> Dash:
    """Two users with different spans, so per-user bounds are observable."""
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2020-03-05 08:00",
                    "2024-11-20 23:30",
                    "2022-01-10 12:00",
                    "2023-06-15 09:00",
                ]
            ),
            "user_id": ["alice", "alice", "bob", "bob"],
        }
    )
    app = Dash(__name__)
    date_range.register(app, df)
    return app


def test_reset_spans_the_selected_users_data(date_app: Dash) -> None:
    """Reset lands on whole days covering only that user's plays."""
    reset = _callback(date_app, "reset_date_range", output_prefix="date-range-mc")

    assert reset(1, "bob") == ([datetime(2022, 1, 10), datetime(2023, 6, 15)], "reset")
    assert reset(1, None) == ([datetime(2020, 3, 5), datetime(2024, 11, 20)], "reset")


def test_reset_ignores_the_initial_unclicked_state(date_app: Dash) -> None:
    reset = _callback(date_app, "reset_date_range", output_prefix="date-range-mc")
    with pytest.raises(PreventUpdate):
        reset(0, "alice")


def test_unknown_user_falls_back_to_all_data(date_app: Dash) -> None:
    """A selection matching nothing must still give a usable range."""
    reset = _callback(date_app, "reset_date_range", output_prefix="date-range-mc")
    assert reset(1, "nobody")[0] == [datetime(2020, 3, 5), datetime(2024, 11, 20)]


@pytest.mark.parametrize(
    ("preset", "start"),
    [
        ("60d", datetime(2024, 9, 21)),
        ("2y", datetime(2022, 11, 21)),
        ("5y", datetime(2019, 11, 22)),
        ("ytd", datetime(2024, 1, 1)),
        ("all", datetime(2020, 3, 5)),
    ],
)
def test_presets_reach_back_from_the_last_day(date_app: Dash, preset: str, start) -> None:
    apply = _callback(date_app, "apply_date_preset", output_prefix="date-range-mc")
    assert apply(preset, "alice") == ([start, datetime(2024, 11, 20)], "preset")


@pytest.mark.parametrize("preset", [None, "", "custom", "bogus"])
def test_non_presets_change_nothing(date_app: Dash, preset) -> None:
    apply = _callback(date_app, "apply_date_preset", output_prefix="date-range-mc")
    with pytest.raises(PreventUpdate):
        apply(preset, "alice")


def test_a_manual_date_change_marks_the_preset_custom(date_app: Dash) -> None:
    on_change = _callback(date_app, "set_preset_on_date_change", output_prefix="date-range-preset")
    assert on_change(["2024-01-01", "2024-02-01"], "manual") == ("custom", "manual")
    assert on_change(["2024-01-01", "2024-02-01"], None) == ("custom", "manual")


def test_a_programmatic_change_keeps_the_preset(date_app: Dash) -> None:
    """A preset or reset moves the dates too; that must not read as manual."""
    on_change = _callback(date_app, "set_preset_on_date_change", output_prefix="date-range-preset")
    for source in ("preset", "reset"):
        assert on_change(["2024-01-01", "2024-02-01"], source) == (dash.no_update, None)


def test_an_incomplete_range_is_ignored(date_app: Dash) -> None:
    on_change = _callback(date_app, "set_preset_on_date_change", output_prefix="date-range-preset")
    for value in (None, [], ["2024-01-01"]):
        with pytest.raises(PreventUpdate):
            on_change(value, "manual")


def test_social_controls_take_a_list_of_users(date_app: Dash) -> None:
    """The Social copy selects several users, and must span all of them."""
    reset = _callback(date_app, "reset_date_range", output_prefix="social-date-range-mc")
    apply = _callback(date_app, "apply_date_preset", output_prefix="social-date-range-mc")

    assert reset(1, ["bob"])[0] == [datetime(2022, 1, 10), datetime(2023, 6, 15)]
    assert reset(1, ["alice", "bob"])[0] == [datetime(2020, 3, 5), datetime(2024, 11, 20)]
    assert reset(1, [])[0] == [datetime(2020, 3, 5), datetime(2024, 11, 20)]
    assert apply("ytd", ["bob"]) == ([datetime(2023, 1, 1), datetime(2023, 6, 15)], "preset")


def test_the_two_tabs_do_not_share_components(date_app: Dash) -> None:
    """Each copy writes only to its own tab's picker."""
    outputs = list(date_app.callback_map)
    assert sum(k.startswith("..date-range-mc") for k in outputs) == 2
    assert sum(k.startswith("..social-date-range-mc") for k in outputs) == 2


# --------------------------------------------------------------------------
# Dropdown search
# --------------------------------------------------------------------------


@pytest.fixture
def music_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny database behind `db_cursor`: two users, one play each in 2024."""
    db = tmp_path / "s.db"
    c = duckdb.connect(str(db))
    c.execute(DDL_PATH.read_text(encoding="utf-8"))
    c.execute("INSERT INTO dim_users (user_id) VALUES ('u1'), ('u2')")
    c.execute("INSERT INTO dim_artists VALUES ('a1', 'Radiohead'), ('a2', 'Rachmaninoff')")
    c.execute("INSERT INTO dim_albums (album_id, album_name) VALUES ('al1', 'OK Computer')")
    for tid, name, artist, album in (
        ("t1", "Paranoid Android", "a1", "al1"),
        ("t2", "Piano Concerto No. 2", "a2", None),
    ):
        c.execute(
            "INSERT INTO dim_tracks (track_id, track_name, album_id) VALUES (?, ?, ?)",
            [tid, name, album],
        )
        c.execute("INSERT INTO bridge_track_artists VALUES (?, ?, 'primary')", [tid, artist])
    c.execute(
        "INSERT INTO fact_plays "
        "(play_id, user_id, track_id, played_at, duration_ms, reason_start, reason_end) VALUES "
        "('p1', 'u1', 't1', '2024-06-01 12:00', 200000, 'clickrow', 'trackdone'), "
        "('p2', 'u2', 't2', '2024-06-02 12:00', 200000, 'clickrow', 'trackdone')"
    )
    c.execute(
        "INSERT INTO dim_genres (genre_id, name, slug, level) VALUES "
        "(1, 'Rock', 'rock', 0), (2, 'Art Rock', 'art-rock', 1)"
    )
    c.close()

    monkeypatch.setenv("MUSIC_DB", str(db))
    return db


@pytest.fixture
def search_app(music_db: Path) -> Dash:
    """The search callbacks over `music_db`."""
    assert music_db.exists()  # db_cursor() reads it through MUSIC_DB
    app = Dash(__name__)
    search.register(app)
    return app


def _search_fn(app: Dash, dropdown_id: str):
    return app.callback_map[f"{dropdown_id}.options"]["callback"].__wrapped__


def test_every_dropdown_gets_a_search_callback(search_app: Dash) -> None:
    for dropdown in (
        "excluded-artists-filter-dropdown",
        "excluded-albums-filter-dropdown",
        "excluded-tracks-filter-dropdown",
        "excluded-genres-filter-dropdown",
        "artist-filter-dropdown",
        "track-filter-dropdown",
    ):
        assert f"{dropdown}.options" in search_app.callback_map


def test_search_without_a_user_does_nothing(search_app: Dash) -> None:
    fn = _search_fn(search_app, "excluded-artists-filter-dropdown")
    with pytest.raises(PreventUpdate):
        fn("radio", None, None)


def test_short_searches_only_keep_the_selection(search_app: Dash) -> None:
    """Under three characters no query runs, but chosen values stay listed."""
    fn = _search_fn(search_app, "excluded-artists-filter-dropdown")
    assert fn("ra", "u1", None) == []
    assert fn("ra", "u1", ["Kept"]) == [{"label": "Kept", "value": "Kept"}]


def test_name_searches_are_scoped_to_the_user(search_app: Dash) -> None:
    """Both artists match "rach"/"rad", but each user only played one."""
    fn = _search_fn(search_app, "excluded-artists-filter-dropdown")
    assert fn("Ra", "u1", None) == []  # still under the minimum
    assert [o["value"] for o in fn("rad", "u1", None)] == ["Radiohead"]
    assert fn("rad", "u2", None) == []
    assert [o["value"] for o in fn("rach", "u2", None)] == ["Rachmaninoff"]


def test_album_and_track_searches_use_their_own_columns(search_app: Dash) -> None:
    albums = _search_fn(search_app, "excluded-albums-filter-dropdown")
    tracks = _search_fn(search_app, "excluded-tracks-filter-dropdown")
    assert [o["value"] for o in albums("comp", "u1", None)] == ["OK Computer"]
    assert [o["value"] for o in tracks("android", "u1", None)] == ["Paranoid Android"]


def test_the_selection_is_not_duplicated(search_app: Dash) -> None:
    fn = _search_fn(search_app, "excluded-artists-filter-dropdown")
    out = fn("rad", "u1", ["Radiohead", "Other"])
    assert [o["value"] for o in out] == ["Radiohead", "Other"]


def test_genre_search_labels_the_taxonomy_level(search_app: Dash) -> None:
    fn = _search_fn(search_app, "excluded-genres-filter-dropdown")
    assert fn("rock", "u1", None) == [
        {"label": "Rock (L0)", "value": "Rock"},
        {"label": "Art Rock (L1)", "value": "Art Rock"},
    ]
    assert fn("zzz", "u1", None) == []


def test_trend_track_search_matches_title_or_artist(search_app: Dash) -> None:
    """The Trends dropdown keys tracks as "Title - Artist"."""
    fn = _search_fn(search_app, "track-filter-dropdown")
    expected = [{"label": "Paranoid Android - Radiohead", "value": "Paranoid Android - Radiohead"}]
    assert fn("paranoid", "u1", None) == expected
    assert fn("radiohead", "u1", None) == expected


def test_trend_artist_search_matches_the_filter_one(search_app: Dash) -> None:
    trend = _search_fn(search_app, "artist-filter-dropdown")
    excluded = _search_fn(search_app, "excluded-artists-filter-dropdown")
    assert trend("rad", "u1", None) == excluded("rad", "u1", None)


# --------------------------------------------------------------------------
# Empty selections
# --------------------------------------------------------------------------


def test_filtered_plays_keep_their_columns_when_nothing_matches(music_db: Path) -> None:
    """The empty result used to keep the raw SQL names.

    The renames to `master_metadata_*` only ran on a non-empty frame, so code
    reading those columns worked for every year with plays and raised KeyError
    for one without.
    """
    con = duckdb.connect(str(music_db), read_only=True)
    try:
        some = get_filtered_plays(con, user_id="u1", start_date=datetime(2024, 1, 1))
        none = get_filtered_plays(con, user_id="u1", start_date=datetime(2030, 1, 1))
    finally:
        con.close()

    assert not some.empty
    assert none.empty
    assert list(none.columns) == list(some.columns)


@pytest.mark.usefixtures("music_db")
def test_wrapped_tab_handles_a_year_with_no_plays() -> None:
    """A selection matching no plays crashed the Wrapped data callback.

    Reachable two ways: the year dropdown lists every user's years, not just
    the selected user's, and the exclusions can empty a year that does have
    plays (Ben's only 2011 plays fall on New Year's Eve, so "Exclude December"
    leaves nothing). It should produce an empty, zeroed summary instead.
    """
    app = Dash(__name__)
    wrapped.register(app, _NO_BOUNDS)
    update = _callback(app, "update_tab_1_data", output_prefix="tab-1-data")

    data = update("u1", 2011, False, False, [], [], [], [])

    stats = data["stats_summary"]
    assert stats["Unique Tracks"] == "0"
    assert stats["Days Listened"] == "0 of 0"
    assert stats["Most Active Day"] == "—"


@pytest.mark.usefixtures("music_db")
def test_wrapped_tab_still_summarizes_a_year_with_plays() -> None:
    app = Dash(__name__)
    wrapped.register(app, _NO_BOUNDS)
    update = _callback(app, "update_tab_1_data", output_prefix="tab-1-data")

    data = update("u1", 2024, False, False, [], [], [], [])

    assert data["stats_summary"]["Unique Tracks"] == "1"


# --------------------------------------------------------------------------
# Year dropdown
# --------------------------------------------------------------------------


@pytest.fixture
def year_options():
    """The user -> year-dropdown callback over three users' bounds.

    Mirrors the startup bounds frame: rows per (user, year), in local time.
    Erin's history starts in 2011, before anyone else's.
    """
    bounds = pd.DataFrame(
        {
            "user_id": ["erin", "erin", "erin", "ben", "ben", "ben"],
            "ts": pd.to_datetime(
                [
                    "2011-03-01",
                    "2012-07-01",
                    "2025-01-01",
                    "2011-12-31 20:00",  # New Year's Eve, local time
                    "2012-05-01",
                    "2025-11-01",
                ],
                format="ISO8601",
            ),
        }
    )
    app = Dash(__name__)
    wrapped.register(app, bounds)
    return _callback(app, "update_year_options", output_prefix="year-dropdown.options")


def _years(options: list[dict]) -> list[int]:
    return [o["value"] for o in options]


def test_years_are_only_the_selected_users(year_options) -> None:
    """Listing every user's years offered ones this user has no plays in."""
    options, _ = year_options("ben", 2025)
    assert _years(options) == [2011, 2012, 2025]
    options, _ = year_options("erin", 2025)
    assert _years(options) == [2011, 2012, 2025]


def test_years_with_no_plays_are_not_offered() -> None:
    """A user with a gap in their history does not get the gap year."""
    bounds = pd.DataFrame(
        {
            "user_id": ["margo", "margo", "erin"],
            "ts": pd.to_datetime(["2013-02-01", "2015-02-01", "2014-06-01"]),
        }
    )
    app = Dash(__name__)
    wrapped.register(app, bounds)
    update = _callback(app, "update_year_options", output_prefix="year-dropdown.options")

    options, _ = update("margo", 2013)
    assert _years(options) == [2013, 2015]


def test_a_valid_year_is_kept_across_a_user_switch(year_options) -> None:
    _, value = year_options("ben", 2012)
    assert value == 2012


def test_an_invalid_year_moves_to_the_users_latest() -> None:
    """Switching to a user without the chosen year lands on their newest one."""
    bounds = pd.DataFrame(
        {
            "user_id": ["erin", "margo", "margo"],
            "ts": pd.to_datetime(["2011-03-01", "2013-02-01", "2016-02-01"]),
        }
    )
    app = Dash(__name__)
    wrapped.register(app, bounds)
    update = _callback(app, "update_year_options", output_prefix="year-dropdown.options")

    options, value = update("margo", 2011)
    assert _years(options) == [2013, 2016]
    assert value == 2016


def test_a_missing_year_selects_the_latest(year_options) -> None:
    _, value = year_options("ben", None)
    assert value == 2025


def test_no_user_falls_back_to_every_year(year_options) -> None:
    options, value = year_options(None, 2012)
    assert _years(options) == [2011, 2012, 2025]
    assert value == 2012


def test_an_unknown_user_gets_no_years(year_options) -> None:
    """Better an empty dropdown than years that will all come back empty."""
    options, value = year_options("nobody", 2025)
    assert options == []
    assert value is None


def test_year_labels_are_strings(year_options) -> None:
    options, _ = year_options("ben", 2025)
    assert options[0] == {"label": "2011", "value": 2011}
