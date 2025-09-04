import contextlib
from collections import OrderedDict
from datetime import datetime
from io import StringIO

import dash
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.dependencies import ClientsideFunction
from dash.exceptions import PreventUpdate

from dashboard.components.filters import (
    create_artist_trends_layout,
    create_genre_trends_layout,
    create_monthly_trend_filter,
    create_track_trends_layout,
)
from dashboard.components.graphs import create_daily_top_heatmap
from dashboard.conn import get_db_connection
from src.metrics.metrics import (
    get_most_played_artists,
    get_most_played_tracks,
    get_playcount_by_day,
    get_top_albums,
    get_top_artist_genres,
)
from src.metrics.trends import (
    get_artist_trends,
    get_genre_trends,
    get_listening_time_by_month,
    get_track_trends,
)
from src.preprocessing import get_filtered_plays


class SimpleLRUCache:
    """Very small, in-process LRU cache for DataFrames keyed by strings."""

    def __init__(self, maxsize: int = 20):
        self.maxsize = maxsize
        self._store: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, key: str) -> pd.DataFrame | None:
        if key in self._store:
            val = self._store.pop(key)
            self._store[key] = val
            return val
        return None

    def set(self, key: str, value: pd.DataFrame) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)


# Lightweight in-process LRU caches for expensive, DuckDB-backed computations
_GENRE_TRENDS_CACHE = SimpleLRUCache(maxsize=20)
_OVERALL_GENRES_CACHE = SimpleLRUCache(maxsize=20)
_TOP_ALBUMS_CACHE = SimpleLRUCache(maxsize=20)
_TOP_GENRES_WRAPPED_CACHE = SimpleLRUCache(maxsize=20)


def _make_filter_cache_key(
    user_id,
    start_date,
    end_date,
    exclude_december,
    remove_incognito,
    excluded_tracks,
    excluded_genres,
    excluded_artists,
    excluded_albums,
) -> str:
    """Create a stable cache key from filter inputs."""

    def _to_tuple(x):
        if x is None:
            return ()
        if isinstance(x, list | tuple | set):
            return tuple(sorted(map(str, x)))
        return (str(x),)

    parts = (
        _to_tuple(user_id),
        _to_tuple(start_date),
        _to_tuple(end_date),
        _to_tuple(bool(exclude_december)),
        _to_tuple(bool(remove_incognito)),
        _to_tuple(excluded_tracks),
        _to_tuple(excluded_genres),
        _to_tuple(excluded_artists),
        _to_tuple(excluded_albums),
    )
    return "|".join([";".join(p) for p in parts])


def _make_wrapped_cache_key(
    user_id,
    selected_year,
    exclude_december,
    remove_incognito,
    excluded_tracks,
    excluded_artists,
    excluded_albums,
) -> str:
    parts = (
        str(user_id or ""),
        f"year:{selected_year}" if selected_year else "year:",
        f"exdec:{bool(exclude_december)}",
        f"rm_incog:{bool(remove_incognito)}",
        "tracks:" + ",".join(sorted(map(str, excluded_tracks or []))),
        "artists:" + ",".join(sorted(map(str, excluded_artists or []))),
        "albums:" + ",".join(sorted(map(str, excluded_albums or []))),
    )
    return "|".join(parts)


def _fmt_genres(g):
    """Format a genre value that may be a scalar, list-like, or NaN."""
    if isinstance(g, list | tuple | set):
        return ", ".join(map(str, g))
    try:
        if pd.isna(g):
            return ""
    except Exception:
        pass
    if isinstance(g, str):
        return g
    return str(g) if g is not None else ""


def get_plotly_theme(is_dark=False):
    """Get Plotly theme configuration based on dark mode setting.

    Args:
        is_dark (bool): Whether dark mode is enabled.

    Returns:
        dict: Plotly layout configuration for the theme.
    """
    if is_dark:
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "#1e1e1e",
            "plot_bgcolor": "#1e1e1e",
            "font": {"color": "#e0e0e0", "family": "Segoe UI, sans-serif"},
            "xaxis": {"gridcolor": "#333"},
            "yaxis": {"gridcolor": "#333"},
            "colorway": [
                "#1DB954",
                "#1ed760",
                "#21e065",
                "#5eb859",
                "#7dd069",
                "#9be082",
                "#b5e8a3",
            ],
        }
    else:
        return {
            "template": "plotly",
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "font": {"family": "Segoe UI, sans-serif"},
            "xaxis": {"gridcolor": "#eee"},
            "yaxis": {"gridcolor": "#eee"},
        }


def register_callbacks(app: Dash, df: pd.DataFrame) -> None:
    """Register all Dash callbacks for the listening dashboard.

    Args:
        app (Dash): Dash application instance.
        df (pd.DataFrame): DataFrame of listening events.
    """

    @app.callback(
        Output("collapse", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse", "is_open")],
    )
    def toggle_collapse(n_clicks: int, is_open: bool) -> bool:
        """Toggle a collapsible UI component.

        Args:
            n_clicks (int): Number of times the toggle button was clicked.
            is_open (bool): Current open/closed state.

        Returns:
            bool: New open/closed state.
        """
        if n_clicks:
            return not is_open
        return is_open

    # ----------------------------------------------------------------------------
    # Server-side dropdown search (artists/albums/tracks) to reduce payload size
    # ----------------------------------------------------------------------------

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

    @app.callback(
        Output("excluded-artists-filter-dropdown", "options"),
        [
            Input("excluded-artists-filter-dropdown", "search_value"),
            Input("user-id-dropdown", "value"),
        ],
        State("excluded-artists-filter-dropdown", "value"),
        prevent_initial_call=True,
    )
    def search_excluded_artists(search_value, user_id, selected_values):
        if not user_id:
            raise PreventUpdate
        search = (search_value or "").strip()
        options: list[dict] = []
        if search:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT ar.artist_name AS name "
                    "FROM fact_plays p "
                    "LEFT JOIN bridge_track_artists b ON b.track_id = p.track_id AND b.role = 'primary' "
                    "LEFT JOIN dim_artists ar ON ar.artist_id = b.artist_id "
                    "WHERE p.user_id = ? AND ar.artist_name ILIKE '%' || ? || '%' "
                    "AND ar.artist_name IS NOT NULL "
                    "ORDER BY ar.artist_name "
                    "LIMIT 25"
                )
                df_opt = con.execute(sql, [user_id, search]).df()
                options = [{"label": n, "value": n} for n in df_opt["name"].tolist()]
            finally:
                with contextlib.suppress(Exception):
                    con.close()
        return _merge_selected(options, selected_values)

    @app.callback(
        Output("excluded-albums-filter-dropdown", "options"),
        [
            Input("excluded-albums-filter-dropdown", "search_value"),
            Input("user-id-dropdown", "value"),
        ],
        State("excluded-albums-filter-dropdown", "value"),
        prevent_initial_call=True,
    )
    def search_excluded_albums(search_value, user_id, selected_values):
        if not user_id:
            raise PreventUpdate
        search = (search_value or "").strip()
        options: list[dict] = []
        if search:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT al.album_name AS name "
                    "FROM fact_plays p "
                    "LEFT JOIN dim_tracks t ON t.track_id = p.track_id "
                    "LEFT JOIN dim_albums al ON al.album_id = t.album_id "
                    "WHERE p.user_id = ? AND al.album_name ILIKE '%' || ? || '%' "
                    "AND al.album_name IS NOT NULL "
                    "ORDER BY al.album_name "
                    "LIMIT 25"
                )
                df_opt = con.execute(sql, [user_id, search]).df()
                options = [{"label": n, "value": n} for n in df_opt["name"].tolist()]
            finally:
                with contextlib.suppress(Exception):
                    con.close()
        return _merge_selected(options, selected_values)

    @app.callback(
        Output("excluded-genres-filter-dropdown", "options"),
        [
            Input("excluded-genres-filter-dropdown", "search_value"),
            Input("user-id-dropdown", "value"),
        ],
        State("excluded-genres-filter-dropdown", "value"),
        prevent_initial_call=True,
    )
    def search_excluded_genres(search_value, user_id, selected_values):
        if not user_id:
            raise PreventUpdate
        search = (search_value or "").strip()
        options: list[dict] = []
        if search:
            con = get_db_connection()
            try:
                try:
                    sql = (
                        "SELECT DISTINCT g.name AS name "
                        "FROM fact_plays p "
                        "JOIN track_genres tg ON tg.track_id = p.track_id "
                        "JOIN dim_genres g ON g.genre_id = tg.genre_id "
                        "WHERE p.user_id = ? AND g.name ILIKE '%' || ? || '%' "
                        "AND g.name IS NOT NULL "
                        "ORDER BY g.name "
                        "LIMIT 25"
                    )
                    df_opt = con.execute(sql, [user_id, search]).df()
                except Exception:
                    # Fallback if normalized genre tables are not present yet
                    sql_fb = (
                        "SELECT name FROM dim_genres "
                        "WHERE name ILIKE '%' || ? || '%' "
                        "ORDER BY name LIMIT 25"
                    )
                    df_opt = con.execute(sql_fb, [search]).df()
                options = [{"label": n, "value": n} for n in df_opt.iloc[:, 0].astype(str).tolist()]
            finally:
                with contextlib.suppress(Exception):
                    con.close()
        return _merge_selected(options, selected_values)

    @app.callback(
        Output("excluded-tracks-filter-dropdown", "options"),
        [
            Input("excluded-tracks-filter-dropdown", "search_value"),
            Input("user-id-dropdown", "value"),
        ],
        State("excluded-tracks-filter-dropdown", "value"),
        prevent_initial_call=True,
    )
    def search_excluded_tracks(search_value, user_id, selected_values):
        if not user_id:
            raise PreventUpdate
        search = (search_value or "").strip()
        options: list[dict] = []
        if search:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT t.track_name AS name "
                    "FROM fact_plays p "
                    "LEFT JOIN dim_tracks t ON t.track_id = p.track_id "
                    "WHERE p.user_id = ? AND t.track_name ILIKE '%' || ? || '%' "
                    "AND t.track_name IS NOT NULL "
                    "ORDER BY t.track_name "
                    "LIMIT 25"
                )
                df_opt = con.execute(sql, [user_id, search]).df()
                options = [{"label": n, "value": n} for n in df_opt["name"].tolist()]
            finally:
                with contextlib.suppress(Exception):
                    con.close()
        return _merge_selected(options, selected_values)

    @app.callback(
        Output("artist-filter-dropdown", "options"),
        [
            Input("artist-filter-dropdown", "search_value"),
            Input("user-id-dropdown", "value"),
        ],
        State("artist-filter-dropdown", "value"),
        prevent_initial_call=True,
    )
    def search_trend_artists(search_value, user_id, selected_values):
        if not user_id:
            raise PreventUpdate
        search = (search_value or "").strip()
        options: list[dict] = []
        if search:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT ar.artist_name AS name "
                    "FROM fact_plays p "
                    "LEFT JOIN bridge_track_artists b ON b.track_id = p.track_id AND b.role = 'primary' "
                    "LEFT JOIN dim_artists ar ON ar.artist_id = b.artist_id "
                    "WHERE p.user_id = ? AND ar.artist_name ILIKE '%' || ? || '%' "
                    "AND ar.artist_name IS NOT NULL "
                    "ORDER BY ar.artist_name "
                    "LIMIT 25"
                )
                df_opt = con.execute(sql, [user_id, search]).df()
                options = [{"label": n, "value": n} for n in df_opt["name"].tolist()]
            finally:
                with contextlib.suppress(Exception):
                    con.close()
        return _merge_selected(options, selected_values)

    @app.callback(
        Output("track-filter-dropdown", "options"),
        [
            Input("track-filter-dropdown", "search_value"),
            Input("user-id-dropdown", "value"),
        ],
        State("track-filter-dropdown", "value"),
        prevent_initial_call=True,
    )
    def search_trend_tracks(search_value, user_id, selected_values):
        if not user_id:
            raise PreventUpdate
        search = (search_value or "").strip()
        options: list[dict] = []
        if search:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT (t.track_name || ' - ' || ar.artist_name) AS track_artist "
                    "FROM fact_plays p "
                    "LEFT JOIN dim_tracks t ON t.track_id = p.track_id "
                    "LEFT JOIN bridge_track_artists b ON b.track_id = p.track_id AND b.role = 'primary' "
                    "LEFT JOIN dim_artists ar ON ar.artist_id = b.artist_id "
                    "WHERE p.user_id = ? AND (t.track_name ILIKE '%' || ? || '%' OR ar.artist_name ILIKE '%' || ? || '%') "
                    "AND t.track_name IS NOT NULL AND ar.artist_name IS NOT NULL "
                    "ORDER BY track_artist "
                    "LIMIT 25"
                )
                df_opt = con.execute(sql, [user_id, search, search]).df()
                vals = df_opt["track_artist"].tolist()
                options = [{"label": v, "value": v} for v in vals]
            finally:
                with contextlib.suppress(Exception):
                    con.close()
        return _merge_selected(options, selected_values)

    @app.callback(
        [
            Output("date-range", "start_date"),
            Output("date-range", "end_date"),
        ],
        Input("reset-date-range", "n_clicks"),
        State("user-id-dropdown", "value"),
    )
    def reset_year_range_filter(n_clicks: int, user_id: str | None):
        """Reset the date picker to span the full range of the data.

        Args:
            n_clicks (int): Number of times the reset button was clicked.

        Returns:
            tuple[datetime, datetime]: Earliest and latest dates in `df`.
        """
        if not n_clicks:
            raise PreventUpdate

        df_user = df if not user_id else df[df.get("user_id") == user_id]
        if df_user.empty:
            df_user = df
        min_ts = df_user["ts"].min()
        max_ts = df_user["ts"].max()
        start = datetime(min_ts.year, min_ts.month, min_ts.day)
        end = datetime(max_ts.year, max_ts.month, max_ts.day)
        return start, end

    # --- Wrapped Tab (Tab 1): Build a single data store, then per-figure callbacks ---

    @app.callback(
        Output("tab-1-data", "data"),
        [
            Input("user-id-dropdown", "value"),
            Input("year-dropdown", "value"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-genres-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
        ],
    )
    def update_tab_1_data(
        user_id,
        selected_year,
        exclude_december,
        remove_incognito,
        excluded_tracks,
        excluded_genres,
        excluded_artists,
        excluded_albums,
    ):
        """Serialize filtered data for Wrapped tab to a single store.

        Returns JSON-serialized DataFrames and preformatted stats summary.
        """
        if not selected_year or not user_id:
            # No context; leave store unchanged to avoid clearing graphs
            raise PreventUpdate

        con = get_db_connection()
        try:
            filtered = get_filtered_plays(
                con,
                user_id=user_id,
                start_date=datetime(selected_year, 1, 1),
                end_date=datetime(selected_year, 12, 31),
                exclude_december=exclude_december,
                remove_incognito=remove_incognito,
                excluded_tracks=excluded_tracks,
                excluded_artists=excluded_artists,
                excluded_albums=excluded_albums,
                excluded_genres=excluded_genres,
            )

            # Compute per-section aggregates (with small LRU caches where applicable)
            top_tracks = get_most_played_tracks(filtered, con=con, limit=10)
            top_artists = get_most_played_artists(filtered, con=con, limit=10)

            w_key = _make_wrapped_cache_key(
                user_id,
                selected_year,
                exclude_december,
                remove_incognito,
                excluded_tracks or [],
                excluded_artists or [],
                excluded_albums or [],
            )
            cached_albums = _TOP_ALBUMS_CACHE.get(w_key)
            if cached_albums is None:
                top_albums = get_top_albums(filtered, con=con)
                _TOP_ALBUMS_CACHE.set(w_key, top_albums)
            else:
                top_albums = cached_albums

            cached_top_genres = _TOP_GENRES_WRAPPED_CACHE.get(w_key)
            if cached_top_genres is None:
                top_genres = get_top_artist_genres(filtered, con=con)
                _TOP_GENRES_WRAPPED_CACHE.set(w_key, top_genres)
            else:
                top_genres = cached_top_genres

            daily_counts = get_playcount_by_day(filtered, con=con, top_only=True)

            # Stats summary (preformatted to avoid shipping large DataFrame)
            from dashboard.components.stats import MS_PER_HOUR, format_stat

            total_ms = filtered["ms_played"].sum()
            total_hours = total_ms / MS_PER_HOUR
            unique_tracks = filtered["master_metadata_track_name"].nunique()
            unique_artists = filtered["master_metadata_album_artist_name"].nunique()
            unique_days = filtered["ts"].dt.date.nunique()
            avg_daily_hours = total_hours / unique_days if unique_days > 0 else 0.0
            daily_ms = filtered.groupby(filtered["ts"].dt.date)["ms_played"].sum()
            most_active_day = daily_ms.idxmax().strftime("%Y-%m-%d") if not daily_ms.empty else "—"
            stats_summary = {
                "Total Listening Time": f"{format_stat(total_hours)} hours",
                "Unique Tracks": format_stat(unique_tracks),
                "Unique Artists": format_stat(unique_artists),
                "Average Daily Listening Time": f"{format_stat(avg_daily_hours)} hours",
                "Most Active Day": most_active_day,
            }

            return {
                "top_tracks": top_tracks.to_json(orient="split"),
                "top_artists": top_artists.to_json(orient="split"),
                "top_albums": top_albums.to_json(orient="split"),
                "top_genres": top_genres.to_json(orient="split"),
                "daily_counts": daily_counts.to_json(date_format="iso", orient="split"),
                "stats_summary": stats_summary,
            }
        finally:
            with contextlib.suppress(Exception):
                con.close()

    # Per-figure callbacks (Wrapped)

    @app.callback(
        Output("top-tracks-graph", "figure"),
        Input("tab-1-data", "data"),
    )
    def render_top_tracks_figure(data):
        theme = get_plotly_theme(False)
        if not data or "top_tracks" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["top_tracks"]), orient="split")
        if df.empty:
            return {"data": [], "layout": theme}
        return {
            "data": [
                {
                    "type": "bar",
                    "x": df["play_count"].head(10),
                    "y": df["track_name"].head(10),
                    "orientation": "h",
                    "text": df["play_count"].head(10),
                    "marker": {"color": "#1DB954"},
                    "customdata": df["track_artist"].head(10),
                    "hovertemplate": ("Track: %{customdata}<br>Plays: %{x}<extra></extra>"),
                }
            ],
            "layout": {
                **theme,
                "xaxis": {**theme.get("xaxis", {}), "title": "Play Count"},
                "yaxis": {**theme.get("yaxis", {}), "title": ""},
            },
        }

    @app.callback(
        Output("top-artists-graph", "figure"),
        Input("tab-1-data", "data"),
    )
    def render_top_artists_figure(data):
        theme = get_plotly_theme(False)
        if not data or "top_artists" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["top_artists"]), orient="split")
        if df.empty:
            return {"data": [], "layout": theme}
        return {
            "data": [
                {
                    "type": "bar",
                    "x": df["play_count"].head(10),
                    "y": df["artist"].head(10),
                    "orientation": "h",
                    "text": df["play_count"].head(10),
                    "customdata": df["unique_tracks"].head(10),
                    "marker": {"color": "#1DB954"},
                    "hovertemplate": (
                        "Artist: %{y}<br>Plays: %{x}<br>Unique Tracks: %{customdata}<extra></extra>"
                    ),
                }
            ],
            "layout": {
                **theme,
                "xaxis": {**theme.get("xaxis", {}), "title": "Play Count"},
                "yaxis": {**theme.get("yaxis", {}), "title": ""},
            },
        }

    @app.callback(
        Output("top-albums-graph", "figure"),
        Input("tab-1-data", "data"),
    )
    def render_top_albums_figure(data):
        theme = get_plotly_theme(False)
        if not data or "top_albums" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["top_albums"]), orient="split")
        if df.empty:
            return {"data": [], "layout": theme}
        return {
            "data": [
                {
                    "type": "bar",
                    "x": df["median_plays"].head(10),
                    "y": df["album_name"].head(10),
                    "orientation": "h",
                    "text": df["median_plays"].round(1).head(10),
                    "customdata": df[["artist"]].head(10).values,
                    "marker": {"color": "#1DB954"},
                    "hovertemplate": (
                        "Album: %{y}<br>Median Plays: %{x}<br>Artist: %{customdata[0]}<extra></extra>"
                    ),
                }
            ],
            "layout": {
                **theme,
                "xaxis": {**theme.get("xaxis", {}), "title": "Median Plays"},
                "yaxis": {**theme.get("yaxis", {}), "title": ""},
            },
        }

    @app.callback(
        Output("top-genres-graph", "figure"),
        Input("tab-1-data", "data"),
    )
    def render_top_genres_figure(data):
        theme = get_plotly_theme(False)
        if not data or "top_genres" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["top_genres"]), orient="split")
        if df.empty:
            return {"data": [], "layout": theme}
        return {
            "data": [
                {
                    "type": "bar",
                    "x": df["play_count"].head(10),
                    "y": df["genre"].head(10),
                    "orientation": "h",
                    "text": [f"{pct:.1f}%" for pct in df["percentage"].head(10)],
                    "customdata": df[["percentage", "top_artists"]].head(10).values,
                    "marker": {"color": "#1DB954"},
                    "hovertemplate": (
                        "Genre: %{y}<br>Tracks: %{x}<br>Percentage: %{customdata[0]:.1f}%<br>Top Artists: %{customdata[1]}<extra></extra>"
                    ),
                }
            ],
            "layout": {
                **theme,
                "xaxis": {**theme.get("xaxis", {}), "title": "Track Count"},
                "yaxis": {**theme.get("yaxis", {}), "title": ""},
            },
        }

    @app.callback(
        Output("detailed-stats", "children"),
        Input("tab-1-data", "data"),
    )
    def render_stats_table(data):
        if not data or "stats_summary" not in data:
            return []
        from dash import html

        stats = data["stats_summary"]
        return html.Table(
            [
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                html.Tbody([html.Tr([html.Td(k), html.Td(v)]) for k, v in stats.items()]),
            ],
            className="stats-table",
        )

    @app.callback(
        Output("daily-song-heatmap", "figure"),
        Input("tab-1-data", "data"),
    )
    def render_daily_heatmap(data):
        theme = get_plotly_theme(False)
        if not data or "daily_counts" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["daily_counts"]), orient="split")
        return create_daily_top_heatmap(df, theme)

    @app.callback(
        Output("tab-2-content", "children"),
        Input("tab-2-chart-selector", "value"),
    )
    def render_trends_selected_content(selection: str):
        """Render only the selected Trends chart and its controls.

        This avoids loading all charts at once by dynamically mounting
        the selected chart's layout into Tab 2.
        """
        if selection == "listening":
            return html.Div(
                [
                    html.H3("Listening Over Time", className="card-title"),
                    create_monthly_trend_filter(),
                    html.Div(
                        [
                            html.H3("Listening Trends", className="card-title"),
                            dcc.Loading(
                                children=dcc.Graph(
                                    id="trends-graph",
                                    figure={},
                                    config={"displayModeBar": False},
                                )
                            ),
                        ]
                    ),
                ],
                className="card",
            )
        if selection == "genres":
            return html.Div(
                [
                    html.H3("Genres Over Time", className="card-title"),
                    create_genre_trends_layout(df),
                    dcc.Loading(
                        children=[
                            html.Div(
                                dcc.Graph(
                                    id="genre-trends-graph",
                                    config={"displayModeBar": False},
                                )
                            ),
                            html.Div(
                                html.Div(
                                    id="genre-trends-table",
                                    className="table-container",
                                )
                            ),
                        ]
                    ),
                ],
                className="card",
            )
        if selection == "artists":
            return html.Div(
                [
                    html.H3("Artists Over Time", className="card-title"),
                    create_artist_trends_layout(df),
                    dcc.Loading(
                        children=[
                            html.Div(
                                dcc.Graph(
                                    id="artist-trends-graph",
                                    config={"displayModeBar": False},
                                )
                            ),
                            html.Div(
                                html.Div(
                                    id="artist-trends-table",
                                    className="table-container",
                                )
                            ),
                        ]
                    ),
                ],
                className="card",
            )
        if selection == "tracks":
            return html.Div(
                [
                    html.H3("Tracks Over Time", className="card-title"),
                    create_track_trends_layout(df),
                    dcc.Loading(
                        children=[
                            dcc.Graph(
                                id="track-trends-graph",
                                config={"displayModeBar": False},
                            ),
                            html.Div(
                                html.Div(
                                    id="track-trends-table",
                                    className="table-container",
                                )
                            ),
                        ]
                    ),
                ],
                className="card",
            )
        # Fallback empty container
        return html.Div()

    @app.callback(
        Output("tab-2-data", "data"),
        [
            Input("user-id-dropdown", "value"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-genres-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
            Input("tab-2-chart-selector", "value"),
        ],
    )
    def update_tab_2_data(
        user_id,
        start_date,
        end_date,
        exclude_december,
        remove_incognito,
        excluded_tracks,
        excluded_genres,
        excluded_artists,
        excluded_albums,
        selection,
    ):
        """Serialize filtered data for Tab 2 (trends and tables).

        Returns JSON-serialized DataFrames for callbacks in Tab 2.
        """
        if not user_id:
            # Without a user context, trends are undefined in DB-backed path
            raise PreventUpdate
        con = get_db_connection()
        try:
            filtered = get_filtered_plays(
                con,
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                exclude_december=exclude_december,
                remove_incognito=remove_incognito,
                excluded_tracks=excluded_tracks,
                excluded_artists=excluded_artists,
                excluded_albums=excluded_albums,
                excluded_genres=excluded_genres,
            )
        except Exception:
            with contextlib.suppress(Exception):
                con.close()
            raise
        data_out: dict[str, str] = {}
        # Compute-on-demand by selected chart to reduce work
        if selection == "listening":
            monthly_stats = get_listening_time_by_month(filtered, con=con)
            data_out["monthly_stats"] = monthly_stats.to_json(date_format="iso", orient="split")
        elif selection == "artists":
            artist_trends = get_artist_trends(filtered, con=con)
            overall_artists = get_most_played_artists(filtered, con=con)
            data_out["artist_trends"] = artist_trends.to_json(date_format="iso", orient="split")
            data_out["overall_artists"] = overall_artists.to_json(date_format="iso", orient="split")
        elif selection == "tracks":
            track_trends = get_track_trends(filtered, con=con)
            overall_tracks = get_most_played_tracks(filtered, con=con)
            data_out["track_trends"] = track_trends.to_json(date_format="iso", orient="split")
            data_out["overall_tracks"] = overall_tracks.to_json(date_format="iso", orient="split")
        elif selection == "genres":
            # Cache heavy DuckDB-backed computations keyed by filter signature
            key = _make_filter_cache_key(
                user_id,
                start_date,
                end_date,
                exclude_december,
                remove_incognito,
                excluded_tracks,
                excluded_genres,
                excluded_artists,
                excluded_albums,
            )
            genre_trends_df = _GENRE_TRENDS_CACHE.get(key)
            if genre_trends_df is None:
                # Use a fresh read-only connection for the heavy join
                con_g = None
                try:
                    con_g = get_db_connection()
                    genre_trends_df = get_genre_trends(filtered, con=con_g)
                finally:
                    if con_g is not None:
                        with contextlib.suppress(Exception):
                            con_g.close()
                _GENRE_TRENDS_CACHE.set(key, genre_trends_df)
            overall_genres_df = _OVERALL_GENRES_CACHE.get(key)
            if overall_genres_df is None:
                con_g2 = None
                try:
                    con_g2 = get_db_connection()
                    overall_genres_df = get_top_artist_genres(filtered, con=con_g2)
                finally:
                    if con_g2 is not None:
                        with contextlib.suppress(Exception):
                            con_g2.close()
                _OVERALL_GENRES_CACHE.set(key, overall_genres_df)
            data_out["genre_trends"] = genre_trends_df.to_json(date_format="iso", orient="split")
            data_out["overall_genres"] = overall_genres_df.to_json(
                date_format="iso", orient="split"
            )

        with contextlib.suppress(Exception):
            con.close()
        return data_out

    @app.callback(
        [
            Output("genre-options-store", "data"),
            Output("genre-filter-dropdown", "options"),
        ],
        [Input("tab-2-data", "data"), Input("tab-2-chart-selector", "value")],
        prevent_initial_call=True,
    )
    def populate_genre_options(data, selection):
        """Populate genre dropdown options from precomputed Tab 2 data.

        Avoids expensive DB work during layout creation by deriving the
        distinct genre list from the already computed `overall_genres`.
        Falls back to dim_genres if overall is not available yet.
        """
        if selection != "genres":
            raise PreventUpdate
        # Prefer precomputed overall_genres to reflect filters
        if data and "overall_genres" in data:
            overall = pd.read_json(StringIO(data["overall_genres"]), orient="split")
            if overall.empty or "genre" not in overall.columns:
                return dash.no_update, []
            genres = sorted(overall["genre"].dropna().unique())
            opts = [{"label": g.title(), "value": g} for g in genres]
            return opts, opts
        # Fallback: load distinct genres from dim_genres (fast query)
        try:
            con = get_db_connection()
            genres_df = con.execute("SELECT name FROM dim_genres ORDER BY name").df()
            genres = genres_df["name"].dropna().astype(str).tolist()
            opts = [{"label": g.title(), "value": g} for g in genres]
            return opts, opts
        except Exception:
            # If DB unavailable, leave empty options
            return dash.no_update, []
        finally:
            with contextlib.suppress(Exception):
                con.close()

    @app.callback(
        Output("trends-graph", "figure"),
        [
            Input("metric-dropdown", "value"),
            Input("tab-2-data", "data"),
        ],
    )
    def update_trend_dashboard(selected_metric, data):
        """Render monthly line chart for the selected metric.

        Uses a default (light) theme; a separate restyle-only callback
        applies the active theme without triggering recomputation.
        """
        theme = get_plotly_theme(False)
        if not data or "monthly_stats" not in data:
            raise PreventUpdate
        monthly = pd.read_json(StringIO(data["monthly_stats"]), orient="split")
        labels = {
            "total_hours": "Total Listening Hours",
            "unique_tracks": "Unique Tracks",
            "unique_artists": "Unique Artists",
            "avg_hours_per_day": "Average Hours per Day",
        }
        title = f"Monthly {labels[selected_metric]}"
        fig = px.line(
            monthly,
            x="month",
            y=selected_metric,
            labels={"month": "Month", selected_metric: labels[selected_metric]},
            title=title,
        )
        fig.update_layout(
            **theme,
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
            showlegend=False,
        )
        fig.update_traces(line_color="#1DB954")
        return fig

    @app.callback(
        [
            Output("genre-trends-graph", "figure"),
            Output("genre-trends-table", "children"),
        ],
        [
            Input("genre-filter-dropdown", "value"),
            Input("top-genres-slider", "value"),
            Input("genre-display-type-radio", "value"),
            Input("tab-2-data", "data"),
        ],
    )
    def update_genre_trends_graph(selected_genres, top_n, display_type, data):
        """Update genre trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        theme = get_plotly_theme(False)
        if not data or "genre_trends" not in data or "overall_genres" not in data:
            raise PreventUpdate
        trends = pd.read_json(StringIO(data["genre_trends"]), orient="split")
        overall = pd.read_json(StringIO(data["overall_genres"]), orient="split")

        # Choose y-axis field
        if display_type == "percentage":
            y_col, y_title = "percentage", "Percentage of Tracks"
        else:
            y_col, y_title = "play_count", "Number of Plays"

        # Auto-select top genres if none chosen
        if not selected_genres:
            avg = overall.groupby("genre")[y_col].mean().sort_values(ascending=False)
            selected_genres = avg.head(top_n).index.tolist()

        plot_df = trends[trends["genre"].isin(selected_genres)]
        fig = px.line(
            plot_df,
            x="month",
            y=y_col,
            color="genre",
            labels={"month": "Month", y_col: y_title, "genre": "Genre"},
            hover_data=["top_artists"],
            title="Genre Trends Over Time",
        )
        fig.update_layout(
            **theme,
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
        )

        table = dash_table.DataTable(
            overall.to_dict("records"),
            [{"name": c, "id": c} for c in overall.columns],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            page_size=10,
            export_format="csv",
            export_headers="display",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "8px",
                "whiteSpace": "normal",
                "height": "auto",
            },
        )
        return fig, [table]

    @app.callback(
        [
            Output("artist-trends-graph", "figure"),
            Output("artist-trends-table", "children"),
        ],
        [
            Input("artist-filter-dropdown", "value"),
            Input("top-artist-slider", "value"),
            Input("artist-display-type-radio", "value"),
            Input("tab-2-data", "data"),
        ],
    )
    def update_artist_trends_graph(selected_artists, top_n, display_type, data):
        """Update artist trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        theme = get_plotly_theme(False)
        if not data or "artist_trends" not in data or "overall_artists" not in data:
            raise PreventUpdate
        trends = pd.read_json(StringIO(data["artist_trends"]), orient="split")
        overall = pd.read_json(StringIO(data["overall_artists"]), orient="split")

        # Determine y-axis field
        if display_type == "percentage":
            y_col, y_title = "percentage", "Percentage of Tracks"
        elif display_type == "play_count":
            y_col, y_title = "play_count", "Number of Plays"
        else:
            y_col, y_title = "unique_tracks", "Unique Tracks"

        # Auto-select if none chosen
        if not selected_artists:
            avg = overall.groupby("artist")[y_col].mean().sort_values(ascending=False)
            selected_artists = avg.head(top_n).index.tolist()

        plot_df = trends[trends["artist"].isin(selected_artists)]
        fig = px.line(
            plot_df,
            x="month",
            y=y_col,
            color="artist",
            labels={"month": "Month", y_col: y_title, "artist": "Artist"},
            hover_data=["top_tracks"],
            title="Artist Trends Over Time",
        )
        fig.update_layout(
            **theme,
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
        )

        # Format genres and build table (robust to NaN/strings)

        overall["artist_genres"] = overall["artist_genres"].apply(_fmt_genres)
        cols = ["artist", "artist_genres", "play_count", "unique_tracks", "percentage"]
        table = dash_table.DataTable(
            overall[cols].to_dict("records"),
            [{"name": c, "id": c} for c in cols],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            page_size=10,
            export_format="csv",
            export_headers="display",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "8px",
                "whiteSpace": "normal",
                "height": "auto",
            },
        )
        return fig, [table]

    @app.callback(
        [
            Output("track-trends-graph", "figure"),
            Output("track-trends-table", "children"),
        ],
        [
            Input("track-filter-dropdown", "value"),
            Input("top-track-slider", "value"),
            Input("track-display-type-radio", "value"),
            Input("tab-2-data", "data"),
        ],
    )
    def update_track_trends_graph(selected_tracks, top_n, display_type, data):
        """Update track trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        theme = get_plotly_theme(False)
        if not data or "track_trends" not in data or "overall_tracks" not in data:
            raise PreventUpdate
        trends = pd.read_json(StringIO(data["track_trends"]), orient="split")
        overall = pd.read_json(StringIO(data["overall_tracks"]), orient="split")

        # Determine y-axis field
        if display_type == "percentage":
            y_col, y_title = "percentage", "Percentage of Plays"
        else:
            y_col, y_title = "play_count", "Number of Plays"

        # Auto-select if none chosen
        if not selected_tracks:
            avg = overall.groupby("track_artist")[y_col].mean().sort_values(ascending=False)
            selected_tracks = avg.head(top_n).index.tolist()

        plot_df = trends[trends["track_artist"].isin(selected_tracks)]
        fig = px.line(
            plot_df,
            x="month",
            y=y_col,
            color="track_artist",
            labels={"month": "Month", y_col: y_title, "track_artist": "Track"},
            hover_data=["track_artist"],
            title="Track Trends Over Time",
            markers=True,
        )
        fig.update_layout(
            **theme,
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
        )

        # Prepare summary table
        overall = overall.drop(columns=["track_artist"])
        overall["artist_genres"] = overall["artist_genres"].apply(_fmt_genres)
        cols = ["track_name", "artist", "artist_genres", "play_count", "percentage"]
        table = dash_table.DataTable(
            overall[cols].to_dict("records"),
            [{"name": c, "id": c} for c in cols],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            page_size=10,
            export_format="csv",
            export_headers="display",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "8px",
                "whiteSpace": "normal",
                "height": "auto",
            },
        )
        return fig, [table]

    @app.callback(
        [
            Output("theme-toggle", "value"),
            Output("app-container", "className"),
            Output("theme-icon", "children"),
            Output("theme-store", "data"),
        ],
        [
            Input("theme-store", "data"),
            Input("theme-toggle", "value"),
        ],
        prevent_initial_call=False,
    )
    def handle_theme(theme_data, toggle_value):
        """Handle theme initialization and toggling.

        This callback manages both theme initialization from localStorage
        and theme toggling from the toggle switch, preventing callback conflicts.

        Args:
            theme_data (dict): Stored theme data from localStorage.
            toggle_value (bool): Current state of the theme toggle switch.

        Returns:
            tuple: Toggle value, theme class name, icon, and theme data to store.
        """
        ctx = dash.callback_context

        # Determine which input triggered the callback
        if not ctx.triggered:
            # Initial load - use stored theme data
            is_dark = theme_data.get("dark", False) if theme_data else False
        else:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id == "theme-store":
                # Theme initialization from storage
                is_dark = theme_data.get("dark", False) if theme_data else False
            elif trigger_id == "theme-toggle":
                # Theme toggle from user interaction
                is_dark = toggle_value if toggle_value is not None else False
            else:
                # Fallback
                is_dark = False

        theme_class = "dark-theme" if is_dark else ""
        icon = "☀️" if is_dark else "🌙"
        theme_store_data = {"dark": is_dark}

        return is_dark, theme_class, icon, theme_store_data

    # Clientside restyle-only callbacks to avoid server roundtrips
    app.clientside_callback(
        ClientsideFunction(namespace="theme", function_name="restyle_wrapped"),
        Output("top-tracks-graph", "figure", allow_duplicate=True),
        Output("top-artists-graph", "figure", allow_duplicate=True),
        Output("top-albums-graph", "figure", allow_duplicate=True),
        Output("top-genres-graph", "figure", allow_duplicate=True),
        Output("daily-song-heatmap", "figure", allow_duplicate=True),
        Input("theme-store", "data"),
        State("top-tracks-graph", "figure"),
        State("top-artists-graph", "figure"),
        State("top-albums-graph", "figure"),
        State("top-genres-graph", "figure"),
        State("daily-song-heatmap", "figure"),
        prevent_initial_call=True,
    )

    # Batch restyle for Trends tab graphs (mounted one at a time)
    app.clientside_callback(
        ClientsideFunction(namespace="theme", function_name="restyle_trends_batch"),
        Output("trends-graph", "figure", allow_duplicate=True),
        Output("genre-trends-graph", "figure", allow_duplicate=True),
        Output("artist-trends-graph", "figure", allow_duplicate=True),
        Output("track-trends-graph", "figure", allow_duplicate=True),
        Input("theme-store", "data"),
        State("trends-graph", "figure"),
        State("genre-trends-graph", "figure"),
        State("artist-trends-graph", "figure"),
        State("track-trends-graph", "figure"),
        prevent_initial_call=True,
    )
