import contextlib
from collections import OrderedDict
from datetime import datetime
from io import StringIO

import dash
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.components.filters import (
    create_artist_trends_layout,
    create_genre_trends_layout,
    create_monthly_trend_filter,
    create_track_trends_layout,
)
from dashboard.components.graphs import create_daily_top_heatmap
from dashboard.components.stats import create_stats_table
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
from src.preprocessing import filter_songs


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

    @app.callback(
        [
            Output("top-tracks-graph", "figure"),
            Output("top-artists-graph", "figure"),
            Output("top-albums-graph", "figure"),
            Output("top-genres-graph", "figure"),
            Output("detailed-stats", "children"),
            Output("daily-song-heatmap", "figure"),
        ],
        [
            Input("user-id-dropdown", "value"),
            Input("year-dropdown", "value"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
            Input("theme-store", "data"),
        ],
    )
    def update_dashboard(
        user_id,
        selected_year,
        exclude_december,
        remove_incognito,
        excluded_tracks,
        excluded_artists,
        excluded_albums,
        theme_data,
    ):
        """Update top-level charts and stats based on user filters.

        Args:
            selected_year (int): Year selected by the user.
            exclude_december (bool): Exclude December if True.
            remove_incognito (bool): Exclude incognito plays if True.
            excluded_tracks (list): Tracks to exclude.
            excluded_artists (list): Artists to exclude.
            excluded_albums (list): Albums to exclude.

        Returns:
            tuple: Figures for tracks, artists, albums, genres, stats table,
                   and daily heatmap.
        """
        # Get theme settings
        is_dark = False
        if isinstance(theme_data, dict):
            dark_value = theme_data.get("dark")
            if isinstance(dark_value, bool):
                is_dark = dark_value
        theme = get_plotly_theme(is_dark)

        # Return empty figures if no year is selected
        if not selected_year:
            empty_fig = {"data": [], "layout": theme}
            return empty_fig, empty_fig, empty_fig, empty_fig, [], empty_fig

        # Filter the dataset (delegate user scoping to filter_songs)
        filtered = filter_songs(
            df,
            user_id=user_id,
            start_date=pd.Timestamp(datetime(selected_year, 1, 1)),
            end_date=pd.Timestamp(datetime(selected_year, 12, 31)),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
            excluded_tracks=excluded_tracks,
            excluded_artists=excluded_artists,
            excluded_albums=excluded_albums,
        )

        # Top tracks
        top_tracks = get_most_played_tracks(filtered)
        if top_tracks.empty:
            tracks_fig = {"data": [], "layout": theme}
        else:
            tracks_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_tracks["play_count"].head(10),
                        "y": top_tracks["track_name"].head(10),
                        "orientation": "h",
                        "text": top_tracks["play_count"].head(10),
                        "marker": {"color": "#1DB954"},
                        "customdata": top_tracks["track_artist"].head(10),
                        "hovertemplate": ("Track: %{customdata}<br>Plays: %{x}<extra></extra>"),
                    }
                ],
                "layout": {
                    **theme,
                    "xaxis": {**theme.get("xaxis", {}), "title": "Play Count"},
                    "yaxis": {**theme.get("yaxis", {}), "title": ""},
                },
            }

        # Top artists
        top_artists = get_most_played_artists(filtered)
        if top_artists.empty:
            artists_fig = {"data": [], "layout": theme}
        else:
            artists_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_artists["play_count"].head(10),
                        "y": top_artists["artist"].head(10),
                        "orientation": "h",
                        "text": top_artists["play_count"].head(10),
                        "customdata": top_artists["unique_tracks"].head(10),
                        "marker": {"color": "#1DB954"},
                        "hovertemplate": (
                            "Artist: %{y}<br>Plays: %{x}"
                            "<br>Unique Tracks: %{customdata}<extra></extra>"
                        ),
                    }
                ],
                "layout": {
                    **theme,
                    "xaxis": {**theme.get("xaxis", {}), "title": "Play Count"},
                    "yaxis": {**theme.get("yaxis", {}), "title": ""},
                },
            }

        # Top albums (DuckDB-backed) with LRU cache by filter signature
        w_key = _make_wrapped_cache_key(
            user_id,
            selected_year,
            exclude_december,
            remove_incognito,
            excluded_tracks,
            excluded_artists,
            excluded_albums,
        )
        cached_albums = _TOP_ALBUMS_CACHE.get(w_key)
        if cached_albums is None:
            con = None
            try:
                con = get_db_connection()
                top_albums = get_top_albums(filtered, con=con)
            finally:
                if con is not None:
                    with contextlib.suppress(Exception):
                        con.close()
            _TOP_ALBUMS_CACHE.set(w_key, top_albums)
        else:
            top_albums = cached_albums
        if top_albums.empty:
            albums_fig = {"data": [], "layout": theme}
        else:
            albums_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_albums["median_plays"].head(10),
                        "y": top_albums["album_name"].head(10),
                        "orientation": "h",
                        "text": top_albums["median_plays"].round(1).head(10),
                        "customdata": top_albums[["artist"]].head(10).values,
                        "marker": {"color": "#1DB954"},
                        "hovertemplate": (
                            "Album: %{y}<br>Median Plays: %{x}"
                            "<br>Artist: %{customdata[0]}<extra></extra>"
                        ),
                    }
                ],
                "layout": {
                    **theme,
                    "xaxis": {**theme.get("xaxis", {}), "title": "Median Plays"},
                    "yaxis": {**theme.get("yaxis", {}), "title": ""},
                },
            }

        # Top genres (DuckDB-backed) with LRU cache by filter signature
        cached_top_genres = _TOP_GENRES_WRAPPED_CACHE.get(w_key)
        if cached_top_genres is None:
            top_genres = get_top_artist_genres(filtered, db_path="data/db/music.db")
            _TOP_GENRES_WRAPPED_CACHE.set(w_key, top_genres)
        else:
            top_genres = cached_top_genres
        if top_genres.empty:
            genres_fig = {"data": [], "layout": theme}
        else:
            genres_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_genres["play_count"].head(10),
                        "y": top_genres["genre"].head(10),
                        "orientation": "h",
                        "text": [f"{pct:.1f}%" for pct in top_genres["percentage"].head(10)],
                        "customdata": top_genres[["percentage", "top_artists"]].head(10).values,
                        "marker": {"color": "#1DB954"},
                        "hovertemplate": (
                            "Genre: %{y}<br>Tracks: %{x}"
                            "<br>Percentage: %{customdata[0]:.1f}%"
                            "<br>Top Artists: %{customdata[1]}<extra></extra>"
                        ),
                    }
                ],
                "layout": {
                    **theme,
                    "xaxis": {**theme.get("xaxis", {}), "title": "Track Count"},
                    "yaxis": {**theme.get("yaxis", {}), "title": ""},
                },
            }

        # Stats table
        stats_table = create_stats_table(filtered)

        # Daily heatmap
        daily_counts = get_playcount_by_day(filtered)
        heatmap_fig = create_daily_top_heatmap(daily_counts, theme)

        return (
            tracks_fig,
            artists_fig,
            albums_fig,
            genres_fig,
            stats_table,
            heatmap_fig,
        )

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
        filtered = filter_songs(
            df,
            user_id=user_id,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
            excluded_genres=excluded_genres,
            excluded_tracks=excluded_tracks,
            excluded_artists=excluded_artists,
            excluded_albums=excluded_albums,
        )
        data_out: dict[str, str] = {}
        # Compute-on-demand by selected chart to reduce work
        if selection == "listening":
            monthly_stats = get_listening_time_by_month(filtered)
            data_out["monthly_stats"] = monthly_stats.to_json(date_format="iso", orient="split")
        elif selection == "artists":
            artist_trends = get_artist_trends(filtered)
            overall_artists = get_most_played_artists(filtered)
            data_out["artist_trends"] = artist_trends.to_json(date_format="iso", orient="split")
            data_out["overall_artists"] = overall_artists.to_json(date_format="iso", orient="split")
        elif selection == "tracks":
            track_trends = get_track_trends(filtered)
            overall_tracks = get_most_played_tracks(filtered)
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
                genre_trends_df = get_genre_trends(filtered, db_path="data/db/music.db")
                _GENRE_TRENDS_CACHE.set(key, genre_trends_df)
            overall_genres_df = _OVERALL_GENRES_CACHE.get(key)
            if overall_genres_df is None:
                overall_genres_df = get_top_artist_genres(filtered, db_path="data/db/music.db")
                _OVERALL_GENRES_CACHE.set(key, overall_genres_df)
            data_out["genre_trends"] = genre_trends_df.to_json(date_format="iso", orient="split")
            data_out["overall_genres"] = overall_genres_df.to_json(
                date_format="iso", orient="split"
            )

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
            Input("theme-store", "data"),
        ],
    )
    def update_trend_dashboard(selected_metric, data, theme_data):
        """Render monthly line chart for the selected metric."""
        is_dark = theme_data.get("dark", False) if theme_data else False
        theme = get_plotly_theme(is_dark)
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
            Input("theme-store", "data"),
        ],
    )
    def update_genre_trends_graph(selected_genres, top_n, display_type, data, theme_data):
        """Update genre trends line chart and summary table."""
        is_dark = theme_data.get("dark", False) if theme_data else False
        theme = get_plotly_theme(is_dark)
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
            Input("theme-store", "data"),
        ],
    )
    def update_artist_trends_graph(selected_artists, top_n, display_type, data, theme_data):
        """Update artist trends line chart and summary table."""
        is_dark = theme_data.get("dark", False) if theme_data else False
        theme = get_plotly_theme(is_dark)
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

        # Format genres and build table
        overall["artist_genres"] = overall["artist_genres"].apply(lambda genres: ", ".join(genres))
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
            Input("theme-store", "data"),
        ],
    )
    def update_track_trends_graph(selected_tracks, top_n, display_type, data, theme_data):
        """Update track trends line chart and summary table."""
        is_dark = theme_data.get("dark", False) if theme_data else False
        theme = get_plotly_theme(is_dark)
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
        overall["artist_genres"] = overall["artist_genres"].apply(lambda genres: ", ".join(genres))
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
