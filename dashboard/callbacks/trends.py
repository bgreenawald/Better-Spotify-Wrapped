"""Trends tab: how listening moved month to month.

One chart is mounted at a time, chosen by `tab-2-chart-selector`, which is why
each graph callback re-checks the selection before doing any work.
"""

from __future__ import annotations

import logging
from io import StringIO

import dash
import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.callbacks.cache import (
    GENRE_TRENDS_CACHE,
    OVERALL_GENRES_CACHE,
    make_filter_cache_key,
)
from dashboard.callbacks.helpers import (
    fmt_genres,
    month_categories,
    register_highcharts_renderer,
)
from dashboard.components.filters import (
    create_artist_trends_layout,
    create_genre_trends_layout,
    create_monthly_trend_filter,
    create_track_trends_layout,
)
from dashboard.conn import db_cursor
from src.metrics.metrics import (
    get_most_played_artists,
    get_most_played_tracks,
    get_top_artist_genres,
)
from src.metrics.trends import (
    get_artist_trends,
    get_genre_trends,
    get_listening_time_by_month,
    get_track_trends,
)
from src.preprocessing import get_filtered_plays

logger = logging.getLogger(__name__)


def _series_by_month(plot_df: pd.DataFrame, key: str, y_col: str) -> tuple[list[str], list[dict]]:
    """One Highcharts series per entity, aligned to a gap-free month axis.

    Args:
        plot_df: Monthly rows with a `month` column, the `key` column naming
            the entity, and the `y_col` value to plot.
        key: Column that identifies each line (genre, artist, track).
        y_col: Column holding the plotted value.

    Returns:
        The month categories and the series; a month with no row plots as 0.
    """
    cats = month_categories(plot_df["month"])
    series = []
    for name, sub in plot_df.groupby(key):
        by_month = sub.set_index(sub["month"].astype(str))[y_col]
        data = [float(by_month.get(m, 0) or 0) for m in cats]
        series.append({"name": str(name), "data": data})
    return cats, series


def _summary_table(records: list[dict], columns: list[dict]) -> dash_table.DataTable:
    """The sortable, filterable, exportable table under each trend chart."""
    return dash_table.DataTable(
        records,
        columns,
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


def _multi_line_options(title: str, cats: list[str], series: list[dict], y_title: str) -> dict:
    """Highcharts options for a multi-line trend chart.

    Carries no colours: the clientside renderer merges the active theme over
    every chart (see `applyTheme` in `assets/highcharts_renderer.js`).
    """
    return {
        "chart": {"type": "line", "height": 360},
        "title": {"text": title},
        "credits": {"enabled": False},
        "xAxis": {"categories": cats, "title": {"text": "Month"}},
        "yAxis": {"title": {"text": y_title}},
        "tooltip": {"shared": True},
        "series": series,
    }


def register(app: Dash, df: pd.DataFrame) -> None:
    """Register the Trends tab's callbacks."""

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
                    dcc.Store(id="trends-options"),
                    dcc.Loading(
                        children=html.Div(
                            id="trends-container",
                            children=html.Div(id="trends-container-root"),
                            style={"minHeight": "380px"},
                        ),
                        delay_show=300,
                        overlay_style={
                            "visibility": "visible",
                            "backgroundColor": "rgba(0,0,0,0.15)",
                        },
                        type="default",
                    ),
                ],
                className="card",
            )
        if selection == "genres":
            return html.Div(
                [
                    html.H3("Genres Over Time", className="card-title"),
                    create_genre_trends_layout(df),
                    dcc.Store(id="genre-trends-options"),
                    dcc.Loading(
                        children=html.Div(
                            id="genre-trends-container",
                            children=html.Div(id="genre-trends-container-root"),
                            style={"minHeight": "380px"},
                        ),
                        delay_show=300,
                        overlay_style={
                            "visibility": "visible",
                            "backgroundColor": "rgba(0,0,0,0.15)",
                        },
                        type="default",
                    ),
                    html.Div(id="genre-trends-table", className="table-container"),
                ],
                className="card",
            )
        if selection == "artists":
            return html.Div(
                [
                    html.H3("Artists Over Time", className="card-title"),
                    create_artist_trends_layout(df),
                    dcc.Store(id="artist-trends-options"),
                    dcc.Loading(
                        children=html.Div(
                            id="artist-trends-container",
                            children=html.Div(id="artist-trends-container-root"),
                            style={"minHeight": "380px"},
                        ),
                        delay_show=300,
                        overlay_style={
                            "visibility": "visible",
                            "backgroundColor": "rgba(0,0,0,0.15)",
                        },
                        type="default",
                    ),
                    html.Div(id="artist-trends-table", className="table-container"),
                ],
                className="card",
            )
        if selection == "tracks":
            return html.Div(
                [
                    html.H3("Tracks Over Time", className="card-title"),
                    create_track_trends_layout(df),
                    dcc.Store(id="track-trends-options"),
                    dcc.Loading(
                        children=html.Div(
                            id="track-trends-container",
                            children=html.Div(id="track-trends-container-root"),
                            style={"minHeight": "380px"},
                        ),
                        delay_show=300,
                        overlay_style={
                            "visibility": "visible",
                            "backgroundColor": "rgba(0,0,0,0.15)",
                        },
                        type="default",
                    ),
                    html.Div(id="track-trends-table", className="table-container"),
                ],
                className="card",
            )
        # Fallback empty container
        return html.Div()

    @app.callback(
        Output("tab-2-data", "data"),
        [
            Input("user-id-dropdown", "value"),
            Input("date-range-mc", "value"),
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
        date_range,
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
        # One `with`, so the connection is released even if a metric raises.
        # The hand-rolled version closed it on two paths and leaked on any
        # other, and could reach a close with `con` unbound.
        with db_cursor() as con:
            if not date_range or len(date_range) != 2:
                raise PreventUpdate
            start_date, end_date = date_range
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
            data_out: dict[str, str] = {}
            # Compute-on-demand by selected chart to reduce work
            if selection == "listening":
                monthly_stats = get_listening_time_by_month(filtered, con=con)
                data_out["monthly_stats"] = monthly_stats.to_json(date_format="iso", orient="split")
            elif selection == "artists":
                artist_trends = get_artist_trends(filtered, con=con)
                overall_artists = get_most_played_artists(filtered, con=con)
                data_out["artist_trends"] = artist_trends.to_json(date_format="iso", orient="split")
                data_out["overall_artists"] = overall_artists.to_json(
                    date_format="iso", orient="split"
                )
            elif selection == "tracks":
                track_trends = get_track_trends(filtered, con=con)
                overall_tracks = get_most_played_tracks(filtered, con=con)
                data_out["track_trends"] = track_trends.to_json(date_format="iso", orient="split")
                data_out["overall_tracks"] = overall_tracks.to_json(
                    date_format="iso", orient="split"
                )
            elif selection == "genres":
                # Cache heavy DuckDB-backed computations keyed by filter signature
                key = make_filter_cache_key(
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
                genre_trends_df = GENRE_TRENDS_CACHE.get(key)
                if genre_trends_df is None:
                    # Use existing connection to avoid setup overhead
                    genre_trends_df = get_genre_trends(filtered, con=con)
                    GENRE_TRENDS_CACHE.set(key, genre_trends_df)
                overall_genres_df = OVERALL_GENRES_CACHE.get(key)
                if overall_genres_df is None:
                    overall_genres_df = get_top_artist_genres(filtered, con=con)
                    OVERALL_GENRES_CACHE.set(key, overall_genres_df)
                data_out["genre_trends"] = genre_trends_df.to_json(
                    date_format="iso", orient="split"
                )
                data_out["overall_genres"] = overall_genres_df.to_json(
                    date_format="iso", orient="split"
                )

            return data_out

    @app.callback(
        Output("genre-options-store", "data"),
        [
            Input("tab-2-data", "data"),
            Input("tab-2-chart-selector", "value"),
        ],
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
                return dash.no_update
            genres = sorted(overall["genre"].dropna().unique())
            opts = [{"label": g.title(), "value": g} for g in genres]
            return opts
        # Fallback: load distinct genres from dim_genres (fast query)
        try:
            with db_cursor() as con:
                genres_df = con.execute("SELECT name, level FROM dim_genres ORDER BY name").df()
            genres = genres_df["name"].dropna().astype(str).tolist()
            return [{"label": g.title(), "value": g} for g in genres]
        except Exception:
            logger.warning("Could not load genre options from dim_genres", exc_info=True)
            return dash.no_update

    # Mirror the genre hide-level0 radio into a store that always exists
    @app.callback(
        Output("genre-hide-level0-store", "data"),
        Input("genre-hide-level0-radio", "value"),
        prevent_initial_call=False,
    )
    def sync_genre_hide_level0(value):
        return bool(value)

    @app.callback(
        Output("genre-filter-dropdown", "options"),
        [
            Input("genre-options-store", "data"),
            Input("tab-2-chart-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def set_genre_dropdown_options(options_data, selection):
        if selection != "genres" or not options_data:
            raise PreventUpdate
        return options_data

    @app.callback(
        Output("trends-options", "data"),
        [
            Input("metric-dropdown", "value"),
            Input("tab-2-data", "data"),
            Input("tab-2-chart-selector", "value"),
        ],
        prevent_initial_call=False,
    )
    def update_trend_dashboard(selected_metric, data, selection):
        """Render monthly line chart for the selected metric.

        Uses a default (light) theme; a separate restyle-only callback
        applies the active theme without triggering recomputation.
        """
        # Only run when the Listening chart is mounted
        if selection != "listening":
            raise PreventUpdate
        if not data or "monthly_stats" not in data:
            raise PreventUpdate
        monthly = pd.read_json(StringIO(data["monthly_stats"]), orient="split")
        labels = {
            "total_hours": "Total Listening Hours",
            "unique_tracks": "Unique Tracks",
            "unique_artists": "Unique Artists",
            "avg_hours_per_day": "Avg Hours per Day (observed days in month)",
        }
        title = f"Monthly {labels[selected_metric]}"
        cats = monthly["month"].astype(str).tolist()
        series_data = [float(x) if pd.notna(x) else 0.0 for x in monthly[selected_metric].tolist()]
        return {
            "chart": {
                "type": "line",
                "height": 360,
            },
            "title": {"text": title},
            "credits": {"enabled": False},
            "legend": {"enabled": False},
            "xAxis": {"categories": cats, "title": {"text": "Month"}},
            "yAxis": {"title": {"text": labels[selected_metric]}},
            "colors": ["#1DB954"],
            "tooltip": {"shared": True},
            "series": [{"name": labels[selected_metric], "data": series_data}],
        }

    @app.callback(
        [
            Output("genre-trends-options", "data"),
            Output("genre-trends-table", "children"),
        ],
        [
            Input("genre-filter-dropdown", "value"),
            Input("top-genres-slider", "value"),
            Input("genre-display-type-radio", "value"),
            Input("tab-2-data", "data"),
            Input("tab-2-chart-selector", "value"),
            Input("genre-hide-level0-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def update_genre_trends_graph(
        selected_genres, top_n, display_type, data, selection, hide_level0_store
    ):
        """Update genre trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        # Only run when the Genres chart is mounted
        if selection != "genres":
            raise PreventUpdate
        if not data or "genre_trends" not in data or "overall_genres" not in data:
            raise PreventUpdate
        hide_level0 = bool(hide_level0_store)
        trends = pd.read_json(StringIO(data["genre_trends"]), orient="split")
        overall = pd.read_json(StringIO(data["overall_genres"]), orient="split")

        # Optionally remove level-0 (parent) genres from both datasets
        if hide_level0:
            try:
                with db_cursor() as con:
                    lvl0 = con.execute(
                        "SELECT name FROM dim_genres WHERE level = 0 AND COALESCE(active, TRUE)"
                    ).df()
                if not lvl0.empty:
                    lvl0_set = set(lvl0["name"].astype(str).tolist())
                    trends = trends[~trends["genre"].isin(lvl0_set)]
                    overall = overall[~overall["genre"].isin(lvl0_set)]
                    # If the user had selected any level-0 genres, drop them
                    if selected_genres:
                        selected_genres = [g for g in selected_genres if g not in lvl0_set]
            except Exception:
                # Parent-genre hiding is cosmetic; keep the chart rather than
                # failing it, but say so instead of swallowing silently.
                logger.warning("Could not hide level-0 genres", exc_info=True)

        # Choose y-axis field
        if display_type == "percentage":
            y_col, y_title = "percentage", "Percentage of Tracks"
        else:
            y_col, y_title = "play_count", "Number of Plays"

        # If filtering removed all rows, return empty artifacts gracefully
        if overall.empty or trends.empty:
            return None, []

        # Auto-select top genres if none chosen
        if not selected_genres:
            avg = overall.groupby("genre")[y_col].mean().sort_values(ascending=False)
            selected_genres = avg.head(top_n).index.tolist()

        plot_df = trends[trends["genre"].isin(selected_genres)]
        cats, series = _series_by_month(plot_df, "genre", y_col)

        table = _summary_table(
            overall.to_dict("records"),
            [
                {"name": "Genre", "id": "genre", "type": "text"},
                {
                    "name": "Plays",
                    "id": "play_count",
                    "type": "numeric",
                    "format": {"specifier": "d"},
                },
                {
                    "name": "Percentage",
                    "id": "percentage",
                    "type": "numeric",
                    "format": {"specifier": ".2g"},
                },
                {"name": "Top Artists", "id": "top_artists", "type": "text"},
            ],
        )
        options = _multi_line_options("Genre Trends Over Time", cats, series, y_title)
        return options, [table]

    @app.callback(
        [
            Output("artist-trends-options", "data"),
            Output("artist-trends-table", "children"),
        ],
        [
            Input("artist-filter-dropdown", "value"),
            Input("top-artist-slider", "value"),
            Input("artist-display-type-radio", "value"),
            Input("tab-2-data", "data"),
            Input("tab-2-chart-selector", "value"),
        ],
        prevent_initial_call=False,
    )
    def update_artist_trends_graph(selected_artists, top_n, display_type, data, selection):
        """Update artist trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        # Only run when the Artists chart is mounted
        if selection != "artists":
            raise PreventUpdate
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

        plot_df = trends[trends["artist"].isin(selected_artists)].copy()
        # Attach formatted artist genres for hover
        if "artist" in overall.columns and "artist_genres" in overall.columns and not plot_df.empty:
            genre_map = (
                overall[["artist", "artist_genres"]]
                .drop_duplicates("artist")
                .assign(artist_genres=lambda d: d["artist_genres"].apply(fmt_genres))
                .set_index("artist")["artist_genres"]
                .to_dict()
            )
            plot_df["artist_genres"] = plot_df["artist"].map(genre_map)
        # Build hover_data conditionally
        hover_data = []
        if "artist_genres" in plot_df.columns:
            hover_data.append("artist_genres")
        if "top_tracks" in plot_df.columns:
            hover_data.append("top_tracks")
        cats, series = _series_by_month(plot_df, "artist", y_col)

        # Format genres and build table (robust to NaN/strings)

        overall["artist_genres"] = overall["artist_genres"].apply(fmt_genres)
        cols = ["artist", "artist_genres", "play_count", "unique_tracks", "percentage"]
        table = _summary_table(
            overall[cols].to_dict("records"),
            [
                {"name": "Artist", "id": "artist", "type": "text"},
                {"name": "Genres", "id": "artist_genres", "type": "text"},
                {
                    "name": "Plays",
                    "id": "play_count",
                    "type": "numeric",
                    "format": {"specifier": "d"},
                },
                {
                    "name": "Unique Tracks",
                    "id": "unique_tracks",
                    "type": "numeric",
                    "format": {"specifier": "d"},
                },
                {
                    "name": "Percentage",
                    "id": "percentage",
                    "type": "numeric",
                    "format": {"specifier": ".2g"},
                },
            ],
        )
        options = _multi_line_options("Artist Trends Over Time", cats, series, y_title)
        return options, [table]

    @app.callback(
        [
            Output("track-trends-options", "data"),
            Output("track-trends-table", "children"),
        ],
        [
            Input("track-filter-dropdown", "value"),
            Input("top-track-slider", "value"),
            Input("track-display-type-radio", "value"),
            Input("tab-2-data", "data"),
            Input("tab-2-chart-selector", "value"),
        ],
        prevent_initial_call=False,
    )
    def update_track_trends_graph(selected_tracks, top_n, display_type, data, selection):
        """Update track trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        # Only run when the Tracks chart is mounted
        if selection != "tracks":
            raise PreventUpdate
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
        cats, series = _series_by_month(plot_df, "track_artist", y_col)

        # Prepare summary table
        overall = overall.drop(columns=["track_artist"])
        overall["artist_genres"] = overall["artist_genres"].apply(fmt_genres)
        cols = ["track_name", "artist", "artist_genres", "play_count", "percentage"]
        table = _summary_table(
            overall[cols].to_dict("records"),
            [
                {"name": "Track Name", "id": "track_name", "type": "text"},
                {"name": "Artist", "id": "artist", "type": "text"},
                {"name": "Genres", "id": "artist_genres", "type": "text"},
                {
                    "name": "Plays",
                    "id": "play_count",
                    "type": "numeric",
                    "format": {"specifier": "d"},
                },
            ],
        )
        options = _multi_line_options("Track Trends Over Time", cats, series, y_title)
        return options, [table]

    # Each chart also re-renders when the selector swaps which one is mounted,
    # which is what `rerender_on_tab_content` covers.
    for container_id, options_id in (
        ("trends-container", "trends-options"),
        ("genre-trends-container", "genre-trends-options"),
        ("artist-trends-container", "artist-trends-options"),
        ("track-trends-container", "track-trends-options"),
    ):
        register_highcharts_renderer(app, container_id, options_id, rerender_on_tab_content=True)
