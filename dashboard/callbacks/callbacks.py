import contextlib
from collections import OrderedDict
from datetime import datetime, timedelta
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
from src.metrics.social import compute_social_regions
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
_GENRE_TRENDS_CACHE = SimpleLRUCache(maxsize=50)
_OVERALL_GENRES_CACHE = SimpleLRUCache(maxsize=50)
_TOP_ALBUMS_CACHE = SimpleLRUCache(maxsize=50)
_TOP_GENRES_WRAPPED_CACHE = SimpleLRUCache(maxsize=50)


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
    excluded_genres,
) -> str:
    parts = (
        str(user_id or ""),
        f"year:{selected_year}" if selected_year else "year:",
        f"exdec:{bool(exclude_december)}",
        f"rm_incog:{bool(remove_incognito)}",
        "tracks:" + ",".join(sorted(map(str, excluded_tracks or []))),
        "artists:" + ",".join(sorted(map(str, excluded_artists or []))),
        "albums:" + ",".join(sorted(map(str, excluded_albums or []))),
        "genres:" + ",".join(sorted(map(str, excluded_genres or []))),
    )
    return "|".join(parts)


def _fmt_genres(g):
    """Format a genre value that may be a scalar, list-like, or NaN."""
    if isinstance(g, list | tuple | set):
        return ", ".join(map(str, g))
    # Handle numpy arrays or other iterables (but not strings/dicts)
    try:
        if hasattr(g, "__iter__") and not isinstance(g, str | bytes | dict):
            return ", ".join(map(str, list(g)))
    except Exception:
        pass
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
            # Trace cycling
            "colorway": [
                "#1DB954",
                "#1ed760",
                "#21e065",
                "#5eb859",
                "#7dd069",
                "#9be082",
                "#b5e8a3",
            ],
            # Sunburst/treemap sector cycling (restores Spotify-green look)
            "sunburstcolorway": [
                "#1DB954",
                "#1ed760",
                "#21e065",
                "#5eb859",
                "#7dd069",
                "#9be082",
                "#b5e8a3",
            ],
            "extendsunburstcolors": True,
            "treemapcolorway": [
                "#1DB954",
                "#1ed760",
                "#21e065",
                "#5eb859",
                "#7dd069",
                "#9be082",
                "#b5e8a3",
            ],
            "extendtreemapcolors": True,
        }
    else:
        return {
            "template": "plotly",
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "font": {"family": "Segoe UI, sans-serif"},
            "xaxis": {"gridcolor": "#eee"},
            "yaxis": {"gridcolor": "#eee"},
            # Keep Spotify-green palette in light theme too
            "colorway": [
                "#1DB954",
                "#1ed760",
                "#21e065",
                "#5eb859",
                "#7dd069",
                "#9be082",
                "#b5e8a3",
            ],
            "sunburstcolorway": [
                "#1DB954",
                "#1ed760",
                "#21e065",
                "#5eb859",
                "#7dd069",
                "#9be082",
                "#b5e8a3",
            ],
            "extendsunburstcolors": True,
            "treemapcolorway": [
                "#1DB954",
                "#1ed760",
                "#21e065",
                "#5eb859",
                "#7dd069",
                "#9be082",
                "#b5e8a3",
            ],
            "extendtreemapcolors": True,
        }


def register_callbacks(app: Dash, df: pd.DataFrame) -> None:
    """Register all Dash callbacks for the listening dashboard.

    Args:
        app (Dash): Dash application instance.
        df (pd.DataFrame): DataFrame of listening events.
    """

    # ---- Helpers for figure label handling ----------------------------------
    def _truncate_label(s: str, max_len: int = 28) -> str:
        try:
            s = str(s)
        except Exception:
            s = ""
        return (s[: max_len - 1] + "…") if len(s) > max_len else s

    def _wrap_or_truncate_labels(labels: list[str]) -> list[str]:
        # Prefer a single-line truncation with a shorter limit to preserve width
        return [_truncate_label(x, 28) for x in labels]

    def _compute_left_margin(labels: list[str]) -> int:
        # Approximate character width to pixels with a tighter cap
        max_len = max((len(str(x)) for x in labels), default=0)
        px = int(8 + max_len * 6.0)  # 6.0 px per character heuristic
        return max(110, min(px, 210))

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

    # Disable global single-user selector on Social tab
    @app.callback(
        Output("user-id-dropdown", "disabled"),
        Input("main-tabs", "value"),
    )
    def disable_user_dropdown_on_social(tab_value: str):
        return tab_value == "social"

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
        # Require at least 3 characters to search
        if len(search) >= 3:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT ve.artist_name AS name "
                    "FROM v_plays_enriched ve "
                    "WHERE ve.user_id = ? AND ve.artist_name ILIKE '%' || ? || '%' "
                    "AND ve.artist_name IS NOT NULL "
                    "ORDER BY ve.artist_name "
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
        # Require at least 3 characters to search
        if len(search) >= 3:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT ve.album_name AS name "
                    "FROM v_plays_enriched ve "
                    "WHERE ve.user_id = ? AND ve.album_name ILIKE '%' || ? || '%' "
                    "AND ve.album_name IS NOT NULL "
                    "ORDER BY ve.album_name "
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
        # Require at least 3 characters to search
        if len(search) >= 3:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT name, COALESCE(level, 0) AS level "
                    "FROM dim_genres "
                    "WHERE COALESCE(active, TRUE) AND (name ILIKE '%' || ? || '%' OR slug ILIKE '%' || ? || '%') "
                    "ORDER BY level, name"
                )
                df_opt = con.execute(sql, [search, search]).df()
                names = df_opt.get("name").astype(str).tolist() if not df_opt.empty else []
                levels = df_opt.get("level").astype(int).tolist() if not df_opt.empty else []
                # Show level in the label; keep raw name as the value
                options = [
                    {"label": f"{n} (L{lvl})", "value": n}
                    for n, lvl in zip(names, levels, strict=False)
                ]
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
        # Require at least 3 characters to search
        if len(search) >= 3:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT ve.track_name AS name "
                    "FROM v_plays_enriched ve "
                    "WHERE ve.user_id = ? AND ve.track_name ILIKE '%' || ? || '%' "
                    "AND ve.track_name IS NOT NULL "
                    "ORDER BY ve.track_name "
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
        # Require at least 3 characters to search
        if len(search) >= 3:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT ve.artist_name AS name "
                    "FROM v_plays_enriched ve "
                    "WHERE ve.user_id = ? AND ve.artist_name ILIKE '%' || ? || '%' "
                    "AND ve.artist_name IS NOT NULL "
                    "ORDER BY ve.artist_name "
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
        # Require at least 3 characters to search
        if len(search) >= 3:
            con = get_db_connection()
            try:
                sql = (
                    "SELECT DISTINCT (ve.track_name || ' - ' || ve.artist_name) AS track_artist "
                    "FROM v_plays_enriched ve "
                    "WHERE ve.user_id = ? AND (ve.track_name ILIKE '%' || ? || '%' OR ve.artist_name ILIKE '%' || ? || '%') "
                    "AND ve.track_name IS NOT NULL AND ve.artist_name IS NOT NULL "
                    "ORDER BY track_artist "
                )
                df_opt = con.execute(sql, [user_id, search, search]).df()
                vals = df_opt["track_artist"].tolist()
                options = [{"label": v, "value": v} for v in vals]
            finally:
                with contextlib.suppress(Exception):
                    con.close()
        return _merge_selected(options, selected_values)

    @app.callback(
        Output("date-range-mc", "value", allow_duplicate=True),
        Output("date-range-source", "data", allow_duplicate=True),
        Input("reset-date-range", "n_clicks"),
        State("user-id-dropdown", "value"),
        prevent_initial_call=True,
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
        return [start, end], "reset"

    # Preset chips -> update Mantine range value
    @app.callback(
        Output("date-range-mc", "value", allow_duplicate=True),
        Output("date-range-source", "data", allow_duplicate=True),
        Input("date-range-preset", "value"),
        State("user-id-dropdown", "value"),
        prevent_initial_call=True,
    )
    def apply_date_preset(preset: str, user_id: str | None):
        if not preset or preset == "custom":
            raise PreventUpdate
        df_user = df if not user_id else df[df.get("user_id") == user_id]
        if df_user.empty:
            df_user = df
        min_ts = df_user["ts"].min()
        max_ts = df_user["ts"].max()
        end = datetime(max_ts.year, max_ts.month, max_ts.day)
        if preset == "60d":
            start = end - timedelta(days=60)
        elif preset == "2y":
            start = end - timedelta(days=2 * 365)
        elif preset == "5y":
            start = end - timedelta(days=5 * 365)
        elif preset == "ytd":
            start = datetime(end.year, 1, 1)
        elif preset == "all":
            start = datetime(min_ts.year, min_ts.month, min_ts.day)
        else:
            raise PreventUpdate
        return [start, end], "preset"

    # When date range changes, set preset to custom only for manual changes.
    @app.callback(
        Output("date-range-preset", "value"),
        Output("date-range-source", "data", allow_duplicate=True),
        Input("date-range-mc", "value"),
        State("date-range-source", "data"),
        prevent_initial_call=True,
    )
    def set_preset_on_date_change(_value, source):
        if not _value or len(_value) != 2:
            raise PreventUpdate
        # If the change originated from a preset or reset, keep the preset selection
        if source in {"preset", "reset"}:
            # Clear the source after handling to allow detecting future manual changes
            return dash.no_update, None
        # Otherwise, treat as manual change and set preset to custom
        return "custom", "manual"

    # --- Social tab date-range controls (independent IDs) ---

    @app.callback(
        Output("social-date-range-mc", "value", allow_duplicate=True),
        Output("social-date-range-source", "data", allow_duplicate=True),
        Input("social-reset-date-range", "n_clicks"),
        State("social-users-dropdown", "value"),
        prevent_initial_call=True,
    )
    def social_reset_date_range(n_clicks: int, users_selected):
        if not n_clicks:
            raise PreventUpdate
        df_sel = df
        if users_selected:
            df_sel = df[df.get("user_id").isin(users_selected)]
            if df_sel.empty:
                df_sel = df
        min_ts = df_sel["ts"].min()
        max_ts = df_sel["ts"].max()
        start = datetime(min_ts.year, min_ts.month, min_ts.day)
        end = datetime(max_ts.year, max_ts.month, max_ts.day)
        return [start, end], "reset"

    @app.callback(
        Output("social-date-range-mc", "value", allow_duplicate=True),
        Output("social-date-range-source", "data", allow_duplicate=True),
        Input("social-date-range-preset", "value"),
        State("social-users-dropdown", "value"),
        prevent_initial_call=True,
    )
    def social_apply_date_preset(preset: str, users_selected):
        if not preset or preset == "custom":
            raise PreventUpdate
        df_sel = df
        if users_selected:
            df_sel = df[df.get("user_id").isin(users_selected)]
            if df_sel.empty:
                df_sel = df
        min_ts = df_sel["ts"].min()
        max_ts = df_sel["ts"].max()
        end = datetime(max_ts.year, max_ts.month, max_ts.day)
        if preset == "60d":
            start = end - timedelta(days=60)
        elif preset == "2y":
            start = end - timedelta(days=2 * 365)
        elif preset == "5y":
            start = end - timedelta(days=5 * 365)
        elif preset == "ytd":
            start = datetime(end.year, 1, 1)
        elif preset == "all":
            start = datetime(min_ts.year, min_ts.month, min_ts.day)
        else:
            raise PreventUpdate
        return [start, end], "preset"

    @app.callback(
        Output("social-date-range-preset", "value"),
        Output("social-date-range-source", "data", allow_duplicate=True),
        Input("social-date-range-mc", "value"),
        State("social-date-range-source", "data"),
        prevent_initial_call=True,
    )
    def social_set_preset_on_date_change(_value, source):
        if not _value or len(_value) != 2:
            raise PreventUpdate
        if source in {"preset", "reset"}:
            return dash.no_update, None
        return "custom", "manual"

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
                excluded_genres or [],
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

            # Build hierarchical rows for sunburst (parents + children)
            try:
                from src.metrics.metrics import get_genre_sunburst_rows

                sb_rows = get_genre_sunburst_rows(filtered, con=con)
            except Exception:
                sb_rows = pd.DataFrame(columns=["parent", "child", "value"])

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
                "top_genres_sunburst": sb_rows.to_json(orient="split"),
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
        State("theme-store", "data"),
    )
    def render_top_tracks_figure(data, theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        theme = get_plotly_theme(is_dark)
        if not data or "top_tracks" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["top_tracks"]), orient="split")
        if df.empty:
            return {"data": [], "layout": theme}
        dff = df.head(10)
        y_vals = dff["track_name"].astype(str).tolist()
        y_ticktext = _wrap_or_truncate_labels(y_vals)
        left_margin = _compute_left_margin(y_ticktext)
        return {
            "data": [
                {
                    "type": "bar",
                    "x": dff["play_count"],
                    "y": y_vals,
                    "orientation": "h",
                    "text": dff["play_count"],
                    "marker": {"color": "#1DB954"},
                    "customdata": dff["track_artist"],
                    "hovertemplate": ("Track: %{customdata}<br>Plays: %{x}<extra></extra>"),
                    "textposition": "outside",
                    "cliponaxis": False,
                }
            ],
            "layout": {
                **theme,
                "margin": {"t": 30, "b": 30, "l": left_margin, "r": 30},
                "height": 440,
                "xaxis": {**theme.get("xaxis", {}), "title": "Play Count", "automargin": True},
                "yaxis": {
                    **theme.get("yaxis", {}),
                    "title": "",
                    "automargin": True,
                    "tickmode": "array",
                    "tickvals": y_vals,
                    "ticktext": y_ticktext,
                    "categoryorder": "array",
                    "categoryarray": y_vals,
                    "tickfont": {"size": 10},
                },
            },
        }

    # -------------------------- Social tab callbacks --------------------------

    @app.callback(
        Output("social-data", "data"),
        [
            Input("social-users-dropdown", "value"),
            Input("social-date-range-mc", "value"),
            Input("social-mode", "value"),
            Input("social-genre-hide-level0-radio", "value"),
            # Global filters respected
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-genres-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
        ],
    )
    def compute_social_data(
        users,
        date_range_value,
        mode,
        hide_parent_genres,
        excluded_artists,
        excluded_genres,
        excluded_albums,
        excluded_tracks,
        exclude_december,
        remove_incognito,
    ):
        # Validate selection
        users = users or []
        if len(users) < 2:
            return {
                "error": "Select 2–3 users to compare.",
                "users": users,
            }
        if len(users) > 3:
            users = users[:3]

        # Parse date range
        start_date = None
        end_date = None
        if date_range_value and isinstance(date_range_value, list) and len(date_range_value) == 2:
            start_date = pd.to_datetime(date_range_value[0]) if date_range_value[0] else None
            end_date = pd.to_datetime(date_range_value[1]) if date_range_value[1] else None

        # Compute with DuckDB backend
        con = get_db_connection()
        try:
            kwargs = {
                "con": con,
                "users": users,
                "start": pd.to_datetime(start_date) if start_date else None,
                "end": pd.to_datetime(end_date) if end_date else None,
                "mode": mode or "tracks",
                "exclude_december": bool(exclude_december),
                "remove_incognito": bool(remove_incognito),
                "excluded_tracks": excluded_tracks,
                "excluded_artists": excluded_artists,
                "excluded_albums": excluded_albums,
                "excluded_genres": excluded_genres,
                "limit_per_region": 10,
            }
            if (mode or "tracks") == "genres":
                kwargs["hide_parent_genres"] = bool(hide_parent_genres)
            out = compute_social_regions(**kwargs)
        finally:
            with contextlib.suppress(Exception):
                con.close()
        return out

    # Show/hide Social genre parent toggle depending on mode
    @app.callback(
        Output("social-genre-hide-level0-container", "style"),
        Input("social-mode", "value"),
    )
    def toggle_social_genre_parent_visibility(mode):
        if mode == "genres":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("social-venn-graph", "figure"),
        Output("social-region-lists", "children"),
        Input("social-data", "data"),
        Input("social-selected-region", "data"),
        Input("theme-store", "data"),
    )
    def render_social(data, selected_region, theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        theme = get_plotly_theme(is_dark)
        import plotly.graph_objects as go
        from dash import html

        if not data or data.get("error"):
            msg = data.get("error") if data else "Select 2–3 users to compare."
            return {"data": [], "layout": theme}, html.Div(msg)
        # Warn and block when any selected user has zero plays within filters
        user_counts = data.get("user_counts", {})
        if any(user_counts.get(str(u), 0) == 0 for u in data.get("users", [])):
            return {"data": [], "layout": theme}, html.Div(
                "One or more selected users have no plays in range; adjust filters or deselect."
            )

        users = data.get("users", [])
        regions = data.get("regions", {})
        totals = data.get("totals", {})
        user_labels = data.get("user_labels", {})

        def lbl(u):
            return user_labels.get(str(u), str(u))

        # Build a minimalist Venn with circles and counts in annotations
        fig = go.Figure()
        # Merge axis visibility overrides with theme without duplicating kwargs
        theme_no_axes = {k: v for k, v in theme.items() if k not in ("xaxis", "yaxis")}
        xaxis_cfg = {**theme.get("xaxis", {}), "visible": False}
        yaxis_cfg = {**theme.get("yaxis", {}), "visible": False}
        fig.update_layout(**theme_no_axes, xaxis=xaxis_cfg, yaxis=yaxis_cfg, height=380)

        ann = []

        def _add_circle(x, y, r, color, name):
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=x - r,
                y0=y - r,
                x1=x + r,
                y1=y + r,
                line={"color": color, "width": 2},
                fillcolor=color,
                opacity=0.12,
            )
            ann.append({"x": x, "y": y + r + 0.1, "text": name, "showarrow": False})

        hover_traces = []
        if len(users) == 2:
            # Place two circles with overlap
            _add_circle(0.0, 0.0, 1.2, "#1DB954", lbl(users[0]))
            _add_circle(1.2, 0.0, 1.2, "#0f7a35", lbl(users[1]))
            # Put intersection count
            inter_key = f"{users[0]}_{users[1]}"
            inter_count = totals.get(inter_key, 0)
            ann.append({"x": 0.6, "y": 0.0, "text": f"∩: {inter_count}", "showarrow": False})
            fig.update_xaxes(range=[-1.6, 2.8])
            fig.update_yaxes(range=[-1.6, 1.8])

            # Hover/select points for regions
            def _tooltip_for(key: str, title: str) -> str:
                items = regions.get(key, [])
                lines = [title, f"Total: {totals.get(key, 0)}"]
                # Include per-user ranks and counts and joint rank for each item
                for it in items[:10]:
                    parts = []
                    # joint rank rounded
                    try:
                        jr = it.get("joint_rank")
                        if jr is not None:
                            parts.append(f"JR {round(float(jr), 2)}")
                    except Exception:
                        pass
                    for u in users:
                        if u in it.get("ranks", {}):
                            rnk = it["ranks"].get(u)
                            cnt = it["counts"].get(u, 0)
                            parts.append(f"{lbl(u)}: r{rnk}/{cnt}")
                    lines.append("- " + it["name"] + (" — " + "; ".join(parts) if parts else ""))
                if totals.get(key, 0) > len(items):
                    lines.append("+ more not shown")
                return "<br>".join(lines)

            # Filled circle polygons for larger hover/click target areas
            import numpy as _np

            def circle_poly(xc, yc, r, steps=100):
                t = _np.linspace(0, 2 * _np.pi, steps)
                return (xc + r * _np.cos(t), yc + r * _np.sin(t))

            # Lens polygon for intersection
            def lens_poly(xc1, yc1, xc2, yc2, r, steps=60):
                import math as _m

                dx = xc2 - xc1
                dy = yc2 - yc1
                d = _m.hypot(dx, dy)
                if d == 0 or d > 2 * r:
                    return ([], [])
                # angles for circle1
                # intersection points angles relative to circle1
                # vector to intersection points from midpoint perpendicular
                # For equal radii and our layout dy=0
                # general case formula
                h = _m.sqrt(max(r * r - (d / 2) * (d / 2), 0.0))
                # unit perpendicular vector
                ux = -dy / d if d != 0 else 0
                uy = dx / d if d != 0 else 0
                # intersection points
                x1 = (xc1 + xc2) / 2 + ux * h
                y1 = (yc1 + yc2) / 2 + uy * h
                x2 = (xc1 + xc2) / 2 - ux * h
                y2 = (yc1 + yc2) / 2 - uy * h
                # angles on each circle
                a1 = _m.atan2(y1 - yc1, x1 - xc1)
                a2 = _m.atan2(y2 - yc1, x2 - xc1)
                b1 = _m.atan2(y1 - yc2, x1 - xc2)
                b2 = _m.atan2(y2 - yc2, x2 - xc2)

                # sample arcs: circle1 from a1 to a2 (short way), circle2 from b2 to b1 (short way)
                def arc(xc, yc, r, ang_start, ang_end, n):
                    # go the shorter direction
                    da = ang_end - ang_start
                    while da <= -_m.pi:
                        da += 2 * _m.pi
                    while da > _m.pi:
                        da -= 2 * _m.pi
                    ts = [ang_start + da * i / (n - 1) for i in range(n)]
                    return [xc + r * _m.cos(t) for t in ts], [yc + r * _m.sin(t) for t in ts]

                xA, yA = arc(xc1, yc1, r, a1, a2, steps)
                xB, yB = arc(xc2, yc2, r, b2, b1, steps)
                return (xA + xB, yA + yB)

            u1, u2 = users
            x1, y1 = circle_poly(0.0, 0.0, 1.2)
            x2, y2 = circle_poly(1.2, 0.0, 1.2)
            xl, yl = lens_poly(0.0, 0.0, 1.2, 0.0, 1.2)

            hover_traces = [
                {
                    "type": "scatter",
                    "x": x1,
                    "y": y1,
                    "mode": "lines",
                    "fill": "toself",
                    "opacity": 0.08,
                    "line": {"color": "#1DB954"},
                    "hovertemplate": _tooltip_for(f"{u1}_only", f"{lbl(u1)} only")
                    + "<extra></extra>",
                    "customdata": [f"{u1}_only"] * len(x1),
                    "name": f"{lbl(u1)} only",
                },
                {
                    "type": "scatter",
                    "x": x2,
                    "y": y2,
                    "mode": "lines",
                    "fill": "toself",
                    "opacity": 0.08,
                    "line": {"color": "#0f7a35"},
                    "hovertemplate": _tooltip_for(f"{u2}_only", f"{lbl(u2)} only")
                    + "<extra></extra>",
                    "customdata": [f"{u2}_only"] * len(x2),
                    "name": f"{lbl(u2)} only",
                },
                {
                    "type": "scatter",
                    "x": xl,
                    "y": yl,
                    "mode": "lines",
                    "fill": "toself",
                    "opacity": 0.14,
                    "line": {"color": "#11803b"},
                    "hovertemplate": _tooltip_for(inter_key, f"{lbl(u1)} ∩ {lbl(u2)}")
                    + "<extra></extra>",
                    "customdata": [inter_key] * len(xl),
                    "name": f"{lbl(u1)} ∩ {lbl(u2)}",
                },
            ]
        else:
            # Three circles positioned in a triangle
            _add_circle(0.0, 0.6, 1.2, "#1DB954", lbl(users[0]))
            _add_circle(1.4, 0.6, 1.2, "#0f7a35", lbl(users[1]))
            _add_circle(0.7, -0.6, 1.2, "#169c48", lbl(users[2]))
            inter_key = f"{users[0]}_{users[1]}_{users[2]}"
            inter_count = totals.get(inter_key, 0)
            ann.append({"x": 0.7, "y": 0.2, "text": f"∩3: {inter_count}", "showarrow": False})
            fig.update_xaxes(range=[-1.6, 3.0])
            fig.update_yaxes(range=[-1.8, 2.0])

            def _tooltip_for(key: str, title: str) -> str:
                items = regions.get(key, [])
                lines = [title, f"Total: {totals.get(key, 0)}"]
                for it in items[:10]:
                    parts = []
                    try:
                        jr = it.get("joint_rank")
                        if jr is not None:
                            parts.append(f"JR {round(float(jr), 2)}")
                    except Exception:
                        pass
                    for u in users:
                        if u in it.get("ranks", {}):
                            rnk = it["ranks"].get(u)
                            cnt = it["counts"].get(u, 0)
                            parts.append(f"{lbl(u)}: r{rnk}/{cnt}")
                    lines.append("- " + it["name"] + (" — " + "; ".join(parts) if parts else ""))
                if totals.get(key, 0) > len(items):
                    lines.append("+ more not shown")
                return "<br>".join(lines)

            u1, u2, u3 = users
            # Fill each user's circle to enlarge hover/click area for only-regions
            import numpy as _np

            def circle_poly(xc, yc, r, steps=100):
                t = _np.linspace(0, 2 * _np.pi, steps)
                return (xc + r * _np.cos(t), yc + r * _np.sin(t))

            x1, y1 = circle_poly(0.0, 0.6, 1.2)
            x2, y2 = circle_poly(1.4, 0.6, 1.2)
            x3, y3 = circle_poly(0.7, -0.6, 1.2)
            fig.add_trace(
                {
                    "type": "scatter",
                    "x": x1,
                    "y": y1,
                    "mode": "lines",
                    "fill": "toself",
                    "opacity": 0.08,
                    "line": {"color": "#1DB954"},
                    "hovertemplate": _tooltip_for(f"{u1}_only", f"{lbl(u1)} only")
                    + "<extra></extra>",
                    "customdata": [f"{u1}_only"] * len(x1),
                    "name": f"{lbl(u1)} only",
                }
            )
            fig.add_trace(
                {
                    "type": "scatter",
                    "x": x2,
                    "y": y2,
                    "mode": "lines",
                    "fill": "toself",
                    "opacity": 0.08,
                    "line": {"color": "#0f7a35"},
                    "hovertemplate": _tooltip_for(f"{u2}_only", f"{lbl(u2)} only")
                    + "<extra></extra>",
                    "customdata": [f"{u2}_only"] * len(x2),
                    "name": f"{lbl(u2)} only",
                }
            )
            fig.add_trace(
                {
                    "type": "scatter",
                    "x": x3,
                    "y": y3,
                    "mode": "lines",
                    "fill": "toself",
                    "opacity": 0.08,
                    "line": {"color": "#169c48"},
                    "hovertemplate": _tooltip_for(f"{u3}_only", f"{lbl(u3)} only")
                    + "<extra></extra>",
                    "customdata": [f"{u3}_only"] * len(x3),
                    "name": f"{lbl(u3)} only",
                }
            )
            hover_traces = [
                # only regions
                {
                    "type": "scatter",
                    "x": [-0.6],
                    "y": [0.8],
                    "mode": "markers",
                    "marker": {
                        "size": 28 if selected_region == f"{u1}_only" else 20,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                    },
                    "name": f"{lbl(u1)} only",
                    "hovertemplate": _tooltip_for(f"{u1}_only", f"{u1} only") + "<extra></extra>",
                    "customdata": [f"{u1}_only"],
                },
                {
                    "type": "scatter",
                    "x": [2.0],
                    "y": [0.8],
                    "mode": "markers",
                    "marker": {
                        "size": 28 if selected_region == f"{u2}_only" else 20,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                    },
                    "name": f"{lbl(u2)} only",
                    "hovertemplate": _tooltip_for(f"{u2}_only", f"{u2} only") + "<extra></extra>",
                    "customdata": [f"{u2}_only"],
                },
                {
                    "type": "scatter",
                    "x": [0.7],
                    "y": [-1.5],
                    "mode": "markers",
                    "marker": {
                        "size": 28 if selected_region == f"{u3}_only" else 20,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                    },
                    "name": f"{lbl(u3)} only",
                    "hovertemplate": _tooltip_for(f"{u3}_only", f"{u3} only") + "<extra></extra>",
                    "customdata": [f"{u3}_only"],
                },
                # pairwise exact intersections
                {
                    "type": "scatter",
                    "x": [0.7],
                    "y": [1.2],
                    "mode": "markers",
                    "marker": {
                        "size": 28 if selected_region == f"{u1}_{u2}" else 20,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                    },
                    "name": f"{lbl(u1)} ∩ {lbl(u2)}",
                    "hovertemplate": _tooltip_for(f"{u1}_{u2}", f"{lbl(u1)} ∩ {lbl(u2)}")
                    + "<extra></extra>",
                    "customdata": [f"{u1}_{u2}"],
                },
                {
                    "type": "scatter",
                    "x": [0.2],
                    "y": [0.0],
                    "mode": "markers",
                    "marker": {
                        "size": 28 if selected_region == f"{u1}_{u3}" else 20,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                    },
                    "name": f"{lbl(u1)} ∩ {lbl(u3)}",
                    "hovertemplate": _tooltip_for(f"{u1}_{u3}", f"{lbl(u1)} ∩ {lbl(u3)}")
                    + "<extra></extra>",
                    "customdata": [f"{u1}_{u3}"],
                },
                {
                    "type": "scatter",
                    "x": [1.2],
                    "y": [0.0],
                    "mode": "markers",
                    "marker": {
                        "size": 28 if selected_region == f"{u2}_{u3}" else 20,
                        "color": "rgba(0,0,0,0)",
                        "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                    },
                    "name": f"{lbl(u2)} ∩ {lbl(u3)}",
                    "hovertemplate": _tooltip_for(f"{u2}_{u3}", f"{lbl(u2)} ∩ {lbl(u3)}")
                    + "<extra></extra>",
                    "customdata": [f"{u2}_{u3}"],
                },
                # 3-way
                {
                    "type": "scatter",
                    "x": [0.7],
                    "y": [0.2],
                    "mode": "markers",
                    "marker": {
                        "size": 32 if selected_region == inter_key else 22,
                        "color": "rgba(0,0,0,0)",
                        "line": {
                            "width": 0,
                            "color": "rgba(0,0,0,0)",
                        },
                    },
                    "name": f"{lbl(u1)} ∩ {lbl(u2)} ∩ {lbl(u3)}",
                    "hovertemplate": _tooltip_for(inter_key, f"{lbl(u1)} ∩ {lbl(u2)} ∩ {lbl(u3)}")
                    + "<extra></extra>",
                    "customdata": [inter_key],
                },
            ]

        for tr in hover_traces:
            fig.add_trace(tr)
        fig.update_layout(annotations=ann, showlegend=False)

        # Region lists
        def _region_block(title: str, items: list[dict], total: int | None = None):
            rows = []
            for it in items[:10]:
                tooltip = " | ".join(
                    [
                        f"{lbl(u)}: r{it['ranks'].get(u, '-')}, {it['counts'].get(u, 0)} plays"
                        for u in users
                        if u in it["ranks"]
                    ]
                )
                rows.append(html.Li([html.Span(it["name"]), html.Span(f" ({tooltip})")]))
            if not rows:
                rows = [html.Li("No items in this region")]
            subtitle = None
            if total is not None and total > len(items):
                subtitle = html.Div(
                    f"+ {total - len(items)} more not shown", className="card-subtitle"
                )
            return html.Div(
                [html.H5(title, className="card-title"), subtitle, html.Ul(rows)], className="card"
            )

        blocks = []
        if len(users) == 2:
            u1, u2 = users
            if selected_region:
                key = selected_region
                title = (
                    f"{u1} only"
                    if key == f"{u1}_only"
                    else (f"{u2} only" if key == f"{u2}_only" else f"{u1} ∩ {u2}")
                )
                blocks.append(_region_block(title, regions.get(key, []), totals.get(key)))
            else:
                blocks.append(
                    html.Div(
                        [html.Div("Click a region to show details", className="card-subtitle")],
                        className="card",
                    )
                )
        else:
            u1, u2, u3 = users
            if selected_region:
                key = selected_region
                if key == f"{u1}_only":
                    title = f"{u1} only"
                elif key == f"{u2}_only":
                    title = f"{u2} only"
                elif key == f"{u3}_only":
                    title = f"{u3} only"
                elif key == f"{u1}_{u2}":
                    title = f"{u1} ∩ {u2}"
                elif key == f"{u1}_{u3}":
                    title = f"{u1} ∩ {u3}"
                elif key == f"{u2}_{u3}":
                    title = f"{u2} ∩ {u3}"
                else:
                    title = f"{u1} ∩ {u2} ∩ {u3}"
                blocks.append(_region_block(title, regions.get(key, []), totals.get(key)))
            else:
                blocks.append(
                    html.Div(
                        [html.Div("Click a region to show details", className="card-subtitle")],
                        className="card",
                    )
                )

        # Explanatory note about consideration limits and display caps
        mode = (data or {}).get("mode", "tracks")
        consider_caps = {"tracks": 250, "artists": 100, "genres": 50}
        cap = consider_caps.get(mode, 10)
        note = html.Div(
            (
                "Note: Social regions use per‑mode consideration caps "
                f"(Tracks 250, Artists 100, Genres 50). Totals are capped by the current mode's cap (now {cap}); "
                "lists show up to 10 items. Less‑specific regions exclude only items already selected in more‑specific regions."
            ),
            className="card-subtitle",
        )

        return fig, html.Div([note, html.Div(blocks, className="graph-container")])

    @app.callback(
        Output("social-selected-region", "data"),
        Input("social-venn-graph", "clickData"),
        State("social-selected-region", "data"),
        prevent_initial_call=True,
    )
    def set_selected_region(click_data, prev):
        if not click_data:
            raise PreventUpdate
        try:
            cd = click_data["points"][0].get("customdata")
        except Exception:
            cd = None
        if not cd:
            raise PreventUpdate
        key = cd if isinstance(cd, str) else cd[0]
        if key == prev:
            return None
        return key

    @app.callback(
        Output("top-artists-graph", "figure"),
        Input("tab-1-data", "data"),
        State("theme-store", "data"),
    )
    def render_top_artists_figure(data, theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        theme = get_plotly_theme(is_dark)
        if not data or "top_artists" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["top_artists"]), orient="split")
        if df.empty:
            return {"data": [], "layout": theme}
        dff = df.head(10)
        y_vals = dff["artist"].astype(str).tolist()
        y_ticktext = _wrap_or_truncate_labels(y_vals)
        left_margin = _compute_left_margin(y_ticktext)
        return {
            "data": [
                {
                    "type": "bar",
                    "x": dff["play_count"],
                    "y": y_vals,
                    "orientation": "h",
                    "text": dff["play_count"],
                    "customdata": dff["unique_tracks"],
                    "marker": {"color": "#1DB954"},
                    "hovertemplate": (
                        "Artist: %{y}<br>Plays: %{x}<br>Unique Tracks: %{customdata}<extra></extra>"
                    ),
                    "textposition": "outside",
                    "cliponaxis": False,
                }
            ],
            "layout": {
                **theme,
                "margin": {"t": 30, "b": 30, "l": left_margin, "r": 30},
                "height": 440,
                "xaxis": {**theme.get("xaxis", {}), "title": "Play Count", "automargin": True},
                "yaxis": {
                    **theme.get("yaxis", {}),
                    "title": "",
                    "automargin": True,
                    "tickmode": "array",
                    "tickvals": y_vals,
                    "ticktext": y_ticktext,
                    "categoryorder": "array",
                    "categoryarray": y_vals,
                    "tickfont": {"size": 10},
                },
            },
        }

    @app.callback(
        Output("top-albums-graph", "figure"),
        Input("tab-1-data", "data"),
        State("theme-store", "data"),
    )
    def render_top_albums_figure(data, theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        theme = get_plotly_theme(is_dark)
        if not data or "top_albums" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["top_albums"]), orient="split")
        if df.empty:
            return {"data": [], "layout": theme}
        dff = df.head(10)
        y_vals = dff["album_name"].astype(str).tolist()
        y_ticktext = _wrap_or_truncate_labels(y_vals)
        left_margin = _compute_left_margin(y_ticktext)
        return {
            "data": [
                {
                    "type": "bar",
                    "x": dff["median_plays"],
                    "y": y_vals,
                    "orientation": "h",
                    "text": dff["median_plays"].round(1),
                    "customdata": dff[["artist"]].values,
                    "marker": {"color": "#1DB954"},
                    "hovertemplate": (
                        "Album: %{y}<br>Median Plays: %{x}<br>Artist: %{customdata[0]}<extra></extra>"
                    ),
                    "textposition": "outside",
                    "cliponaxis": False,
                }
            ],
            "layout": {
                **theme,
                "margin": {"t": 30, "b": 30, "l": left_margin, "r": 30},
                "height": 440,
                "xaxis": {**theme.get("xaxis", {}), "title": "Median Plays", "automargin": True},
                "yaxis": {
                    **theme.get("yaxis", {}),
                    "title": "",
                    "automargin": True,
                    "tickmode": "array",
                    "tickvals": y_vals,
                    "ticktext": y_ticktext,
                    "categoryorder": "array",
                    "categoryarray": y_vals,
                    "tickfont": {"size": 10},
                },
            },
        }

    @app.callback(
        Output("top-genres-graph", "figure"),
        Input("tab-1-data", "data"),
        State("theme-store", "data"),
    )
    def render_top_genres_figure(data, theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        theme = get_plotly_theme(is_dark)
        if not data or ("top_genres" not in data and "top_genres_sunburst" not in data):
            return {"data": [], "layout": theme}
        # Prefer precomputed hierarchical rows when available
        if "top_genres_sunburst" in data:
            sb = pd.read_json(StringIO(data["top_genres_sunburst"]), orient="split")
        else:
            # Fallback: derive rows from flat top_genres + taxonomy
            df = pd.read_json(StringIO(data["top_genres"]), orient="split")
            if df.empty:
                return {"data": [], "layout": theme}
            try:
                con = get_db_connection()
                edges = con.execute(
                    """
                    SELECT p.name AS parent, c.name AS child
                    FROM genre_hierarchy gh
                    JOIN dim_genres p ON p.genre_id = gh.parent_genre_id
                    JOIN dim_genres c ON c.genre_id = gh.child_genre_id
                    """
                ).df()
                parents = con.execute(
                    "SELECT name FROM dim_genres WHERE level = 0 AND COALESCE(active, TRUE)"
                ).df()
            finally:
                with contextlib.suppress(Exception):
                    con.close()

            parents_set = {str(x).strip() for x in parents["name"]} if not parents.empty else set()
            child_to_parents: dict[str, list[str]] = {}
            if not edges.empty:
                for _, row in edges.iterrows():
                    child = str(row["child"]).strip()
                    parent = str(row["parent"]).strip()
                    child_to_parents.setdefault(child.lower(), []).append(parent)
            rows: list[dict[str, str | int | float]] = []
            for _, r in df.iterrows():
                g = str(r.get("genre", "")).strip()
                if not g:
                    continue
                count = int(r.get("play_count") or r.get("track_count") or 0)
                if count <= 0:
                    continue
                gl = g.lower()
                parents_for_child = child_to_parents.get(gl)
                if parents_for_child:
                    for p in parents_for_child:
                        rows.append({"parent": p, "child": g, "value": count})
                elif g in parents_set:
                    rows.append({"parent": g, "child": f"{g} (direct)", "value": count})
                else:
                    rows.append({"parent": "Other", "child": g, "value": count})
            sb = pd.DataFrame(rows)

        if sb.empty:
            return {"data": [], "layout": theme}

        import plotly.express as px

        # Include top artists in hover for child/direct nodes
        # Pass template and discrete color sequence at creation time so initial
        # render uses the intended palette before any clientside restyle occurs.
        fig = px.sunburst(
            sb,
            path=["parent", "child"],
            values="value",
            custom_data=["top_artists"] if "top_artists" in sb.columns else None,
            template=theme.get("template", None),
            color_discrete_sequence=theme.get("sunburstcolorway") or theme.get("colorway"),
        )
        fig.update_layout(
            **theme,
            margin={"t": 30, "b": 30, "l": 30, "r": 30},
            height=450,
        )
        # Parent nodes may not carry customdata; child nodes will include top artists when available
        base_hover = "%{label}<br>Plays: %{value}"
        artists_hover = "<br>Top Artists: %{customdata[0]}" if "top_artists" in sb.columns else ""
        fig.update_traces(hovertemplate=base_hover + artists_hover + "<extra></extra>")
        return fig

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
        State("theme-store", "data"),
    )
    def render_daily_heatmap(data, theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        theme = get_plotly_theme(is_dark)
        if not data or "daily_counts" not in data:
            return {"data": [], "layout": theme}
        df = pd.read_json(StringIO(data["daily_counts"]), orient="split")
        fig_or_div = create_daily_top_heatmap(df, theme)
        if isinstance(fig_or_div, dict) or hasattr(fig_or_div, "to_plotly_json"):
            return fig_or_div
        return {"data": [], "layout": theme}

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
                            dcc.Loading(
                                children=dcc.Graph(
                                    id="trends-graph",
                                    figure={},
                                    config={"displayModeBar": False},
                                ),
                                delay_show=300,
                                overlay_style={
                                    "visibility": "visible",
                                    "backgroundColor": "rgba(0,0,0,0.15)",
                                },
                                type="default",
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
                        ],
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
                        ],
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
                        ],
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
        con = get_db_connection()
        try:
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
                # Use existing connection to avoid setup overhead
                genre_trends_df = get_genre_trends(filtered, con=con)
                _GENRE_TRENDS_CACHE.set(key, genre_trends_df)
            overall_genres_df = _OVERALL_GENRES_CACHE.get(key)
            if overall_genres_df is None:
                overall_genres_df = get_top_artist_genres(filtered, con=con)
                _OVERALL_GENRES_CACHE.set(key, overall_genres_df)
            data_out["genre_trends"] = genre_trends_df.to_json(date_format="iso", orient="split")
            data_out["overall_genres"] = overall_genres_df.to_json(
                date_format="iso", orient="split"
            )

        with contextlib.suppress(Exception):
            con.close()
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
            con = get_db_connection()
            genres_df = con.execute("SELECT name, level FROM dim_genres ORDER BY name").df()
            genres = genres_df["name"].dropna().astype(str).tolist()
            opts = [{"label": g.title(), "value": g} for g in genres]
            return opts
        except Exception:
            # If DB unavailable, leave empty options
            return dash.no_update
        finally:
            with contextlib.suppress(Exception):
                con.close()

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
        Output("trends-graph", "figure"),
        [
            Input("metric-dropdown", "value"),
            Input("tab-2-data", "data"),
            Input("theme-store", "data"),
            Input("tab-2-chart-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_trend_dashboard(selected_metric, data, theme_data, selection):
        """Render monthly line chart for the selected metric.

        Uses a default (light) theme; a separate restyle-only callback
        applies the active theme without triggering recomputation.
        """
        # Only run when the Listening chart is mounted
        if selection != "listening":
            raise PreventUpdate
        is_dark = bool(theme_data and theme_data.get("dark"))
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
            Input("tab-2-chart-selector", "value"),
            Input("genre-hide-level0-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_genre_trends_graph(
        selected_genres, top_n, display_type, data, theme_data, selection, hide_level0_store
    ):
        """Update genre trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        # Only run when the Genres chart is mounted
        if selection != "genres":
            raise PreventUpdate
        is_dark = bool(theme_data and theme_data.get("dark"))
        theme = get_plotly_theme(is_dark)
        if not data or "genre_trends" not in data or "overall_genres" not in data:
            raise PreventUpdate
        hide_level0 = bool(hide_level0_store)
        trends = pd.read_json(StringIO(data["genre_trends"]), orient="split")
        overall = pd.read_json(StringIO(data["overall_genres"]), orient="split")

        # Optionally remove level-0 (parent) genres from both datasets
        if hide_level0:
            try:
                con = get_db_connection()
                lvl0 = con.execute(
                    "SELECT name FROM dim_genres WHERE level = 0 AND COALESCE(active, TRUE)"
                ).df()
                if not lvl0.empty:
                    lvl0_set = set(lvl0["name"].astype(str).tolist())
                    trends = trends[~trends["genre"].isin(lvl0_set)]
                    overall = overall[~overall["genre"].isin(lvl0_set)]
                    # If user had selected any level-0 genres, drop them
                    if selected_genres:
                        selected_genres = [g for g in selected_genres if g not in lvl0_set]
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    con.close()

        # Choose y-axis field
        if display_type == "percentage":
            y_col, y_title = "percentage", "Percentage of Tracks"
        else:
            y_col, y_title = "play_count", "Number of Plays"

        # If filtering removed all rows, return empty artifacts gracefully
        if overall.empty or trends.empty:
            return {"data": [], "layout": theme}, []

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
            Input("tab-2-chart-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_artist_trends_graph(
        selected_artists, top_n, display_type, data, theme_data, selection
    ):
        """Update artist trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        # Only run when the Artists chart is mounted
        if selection != "artists":
            raise PreventUpdate
        is_dark = bool(theme_data and theme_data.get("dark"))
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

        plot_df = trends[trends["artist"].isin(selected_artists)].copy()
        # Attach formatted artist genres for hover
        if "artist" in overall.columns and "artist_genres" in overall.columns and not plot_df.empty:
            genre_map = (
                overall[["artist", "artist_genres"]]
                .drop_duplicates("artist")
                .assign(artist_genres=lambda d: d["artist_genres"].apply(_fmt_genres))
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
        fig = px.line(
            plot_df,
            x="month",
            y=y_col,
            color="artist",
            labels={"month": "Month", y_col: y_title, "artist": "Artist"},
            hover_data=hover_data if hover_data else None,
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
            Input("tab-2-chart-selector", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_track_trends_graph(
        selected_tracks, top_n, display_type, data, theme_data, selection
    ):
        """Update track trends line chart and summary table.

        Uses a default (light) theme; a restyle-only callback updates
        layout colors/templates on theme changes.
        """
        # Only run when the Tracks chart is mounted
        if selection != "tracks":
            raise PreventUpdate
        is_dark = bool(theme_data and theme_data.get("dark"))
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
        overall["artist_genres"] = overall["artist_genres"].apply(_fmt_genres)
        cols = ["track_name", "artist", "artist_genres", "play_count", "percentage"]
        table = dash_table.DataTable(
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
            Output("theme-toggle", "checked"),
            Output("app-container", "className"),
            Output("theme-store", "data"),
        ],
        [
            Input("theme-store", "data"),
            Input("theme-toggle", "checked"),
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
        theme_store_data = {"dark": is_dark}

        return is_dark, theme_class, theme_store_data

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

    # Note: Trends graphs are themed in their server callbacks using theme-store,
    # so we avoid a clientside batch restyle here to prevent warnings when
    # outputs are not mounted in the current layout.

    # Update Mantine provider theme for dmc components
    @app.callback(
        Output("mantine-provider", "forceColorScheme"),
        Output("mantine-provider", "theme"),
        Input("theme-store", "data"),
    )
    def set_mantine_theme(theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        scheme = "dark" if is_dark else "light"
        return scheme, {"primaryColor": "green"}
