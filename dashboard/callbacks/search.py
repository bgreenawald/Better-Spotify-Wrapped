"""Server-side search for the dashboard's dropdowns.

The filter dropdowns cover tens of thousands of artists, albums and tracks.
Shipping every option to the browser made the initial payload enormous, so each
dropdown instead queries as the user types.

All seven dropdowns behaved identically apart from their query -- same
three-character minimum, same "keep the current selection visible" step, same
guard on a missing user. Only the query is written out per dropdown now.
"""

from __future__ import annotations

from collections.abc import Callable

from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

from dashboard.conn import db_cursor

# Searching on one or two characters matches most of the library, which is both
# slow and useless as a narrowing step.
MIN_SEARCH_CHARS = 3

# A fetcher takes an open connection, the selected user and the search text,
# and returns Dash dropdown options.
Fetcher = Callable[[object, str, str], list[dict]]


def _merge_selected(options: list[dict], selected: list[str] | None) -> list[dict]:
    """Ensure selected values remain visible in options."""
    if not selected:
        return options
    existing = {o.get("value") for o in options}
    merged = options[:]
    for v in selected:
        if v not in existing and v is not None:
            merged.append({"label": v, "value": v})
    return merged


def _register_search(app: Dash, dropdown_id: str, fetcher: Fetcher) -> None:
    """Register one typed-search callback for a dropdown.

    Args:
        app: Dash application instance.
        dropdown_id: The dropdown whose `options` this callback fills.
        fetcher: Runs the query and shapes the options.
    """

    @app.callback(
        Output(dropdown_id, "options"),
        [
            Input(dropdown_id, "search_value"),
            Input("user-id-dropdown", "value"),
        ],
        State(dropdown_id, "value"),
        prevent_initial_call=True,
    )
    def _search(search_value, user_id, selected_values):
        if not user_id:
            raise PreventUpdate
        search = (search_value or "").strip()
        options: list[dict] = []
        if len(search) >= MIN_SEARCH_CHARS:
            with db_cursor() as con:
                options = fetcher(con, user_id, search)
        return _merge_selected(options, selected_values)


def _enriched_name_fetcher(column: str) -> Fetcher:
    """A fetcher listing one distinct name column from `v_plays_enriched`.

    `column` is interpolated into the SQL, so it must stay a literal chosen
    here rather than anything reaching this module from a request.
    """

    def fetch(con, user_id: str, search: str) -> list[dict]:
        sql = (
            f"SELECT DISTINCT ve.{column} AS name "
            "FROM v_plays_enriched ve "
            f"WHERE ve.user_id = ? AND ve.{column} ILIKE '%' || ? || '%' "
            f"AND ve.{column} IS NOT NULL "
            f"ORDER BY ve.{column} "
        )
        df_opt = con.execute(sql, [user_id, search]).df()
        return [{"label": n, "value": n} for n in df_opt["name"].tolist()]

    return fetch


def _fetch_genres(con, _user_id, search: str) -> list[dict]:
    """Canonical genres, labelled with their depth in the taxonomy.

    The query itself ignores the user: the taxonomy is shared across users.
    """
    sql = (
        "SELECT name, COALESCE(level, 0) AS level "
        "FROM dim_genres "
        "WHERE COALESCE(active, TRUE) AND (name ILIKE '%' || ? || '%' OR slug ILIKE '%' || ? || '%') "
        "ORDER BY level, name"
    )
    df_opt = con.execute(sql, [search, search]).df()
    if df_opt.empty:
        return []
    names = df_opt["name"].astype(str).tolist()
    levels = df_opt["level"].astype(int).tolist()
    # Show level in the label; keep raw name as the value
    return [{"label": f"{n} (L{lvl})", "value": n} for n, lvl in zip(names, levels, strict=False)]


def _fetch_track_with_artist(con, user_id: str, search: str) -> list[dict]:
    """Tracks as "Title - Artist", which is what the Trends dropdown keys on."""
    sql = (
        "SELECT DISTINCT (ve.track_name || ' - ' || ve.artist_name) AS track_artist "
        "FROM v_plays_enriched ve "
        "WHERE ve.user_id = ? AND (ve.track_name ILIKE '%' || ? || '%' "
        "OR ve.artist_name ILIKE '%' || ? || '%') "
        "AND ve.track_name IS NOT NULL AND ve.artist_name IS NOT NULL "
        "ORDER BY track_artist "
    )
    df_opt = con.execute(sql, [user_id, search, search]).df()
    return [{"label": v, "value": v} for v in df_opt["track_artist"].tolist()]


def register(app: Dash) -> None:
    """Register every dropdown search callback."""
    _register_search(app, "excluded-artists-filter-dropdown", _enriched_name_fetcher("artist_name"))
    _register_search(app, "excluded-albums-filter-dropdown", _enriched_name_fetcher("album_name"))
    _register_search(app, "excluded-tracks-filter-dropdown", _enriched_name_fetcher("track_name"))
    _register_search(app, "excluded-genres-filter-dropdown", _fetch_genres)
    _register_search(app, "artist-filter-dropdown", _enriched_name_fetcher("artist_name"))
    _register_search(app, "track-filter-dropdown", _fetch_track_with_artist)
