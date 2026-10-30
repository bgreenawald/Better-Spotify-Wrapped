"""Wrapped tab: one filtered data store, then a callback per figure.

The expensive filtering and metric work happens once in `update_tab_1_data`,
which writes its results to `tab-1-data`; every chart below reads that store.
`theme-store` is deliberately not an input to any of them -- the clientside
renderer merges the current theme over the options it is given, so flipping to
dark mode must not re-run the queries.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO

import pandas as pd
from dash import Dash, Input, Output, html
from dash.exceptions import PreventUpdate

from dashboard.callbacks.cache import (
    TOP_ALBUMS_CACHE,
    TOP_GENRES_WRAPPED_CACHE,
    make_wrapped_cache_key,
)
from dashboard.callbacks.helpers import (
    compute_left_margin,
    empty_chart,
    register_highcharts_renderer,
    wrap_or_truncate_labels,
)
from dashboard.components.graphs import create_daily_top_playcount_grid
from dashboard.conn import db_cursor
from src.metrics.metrics import (
    get_most_played_artists,
    get_most_played_tracks,
    get_playcount_by_day,
    get_top_albums,
    get_top_artist_genres,
)
from src.preprocessing import get_filtered_plays


def register(app: Dash) -> None:
    """Register the Wrapped tab's data store, figures and chart renderers."""
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

        with db_cursor() as con:
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

            w_key = make_wrapped_cache_key(
                user_id,
                selected_year,
                exclude_december,
                remove_incognito,
                excluded_tracks or [],
                excluded_artists or [],
                excluded_albums or [],
                excluded_genres or [],
            )
            cached_albums = TOP_ALBUMS_CACHE.get(w_key)
            if cached_albums is None:
                top_albums = get_top_albums(filtered, con=con)
                TOP_ALBUMS_CACHE.set(w_key, top_albums)
            else:
                top_albums = cached_albums

            cached_top_genres = TOP_GENRES_WRAPPED_CACHE.get(w_key)
            if cached_top_genres is None:
                top_genres = get_top_artist_genres(filtered, con=con)
                TOP_GENRES_WRAPPED_CACHE.set(w_key, top_genres)
            else:
                top_genres = cached_top_genres

            # Build hierarchical rows for sunburst (parents + children)
            try:
                from src.metrics.metrics import get_genre_sunburst_rows

                sb_rows = get_genre_sunburst_rows(filtered, con=con)
            except Exception:
                sb_rows = pd.DataFrame(columns=["parent", "child", "value"])

            daily_counts = get_playcount_by_day(filtered, con=con, top_only=False)

            # Stats summary (preformatted to avoid shipping large DataFrame)
            from dashboard.components.stats import MS_PER_HOUR, format_stat

            total_ms = filtered["ms_played"].sum()
            total_hours = total_ms / MS_PER_HOUR
            unique_tracks = filtered["master_metadata_track_name"].nunique()
            unique_artists = filtered["master_metadata_album_artist_name"].nunique()
            # Two denominators, each labelled, because they answer different
            # questions and used to be conflated: the Trends tab divided by
            # calendar days while this divided by days with any listening, so
            # two "average per day" figures on one dashboard disagreed.
            dates = filtered["ts"].dt.date
            active_days = dates.nunique()
            span_days = (dates.max() - dates.min()).days + 1 if active_days else 0
            avg_per_calendar_day = total_hours / span_days if span_days > 0 else 0.0
            avg_per_active_day = total_hours / active_days if active_days > 0 else 0.0
            daily_ms = filtered.groupby(dates)["ms_played"].sum()
            most_active_day = daily_ms.idxmax().strftime("%Y-%m-%d") if not daily_ms.empty else "—"
            stats_summary = {
                "Total Listening Time": f"{format_stat(total_hours)} hours",
                "Unique Tracks": format_stat(unique_tracks),
                "Unique Artists": format_stat(unique_artists),
                "Average per Day": f"{format_stat(avg_per_calendar_day)} hours",
                "Average per Day Listened": f"{format_stat(avg_per_active_day)} hours",
                "Days Listened": f"{format_stat(active_days)} of {format_stat(span_days)}",
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

    # Per-figure callbacks (Wrapped)

    @app.callback(
        Output("top-tracks-options", "data"),
        # theme-store is deliberately NOT an Input here: the clientside
        # renderer applies the theme and merges it over these options, so a
        # dark-mode flip must not re-run this server callback (it would
        # re-parse the whole filter payload and rebuild every chart).
        [Input("tab-1-data", "data")],
    )
    def render_top_tracks_figure(data):
        """Build Highcharts options for Most Played Tracks and store them.

        The actual rendering happens clientside into the 'top-tracks-container'.
        """
        if not data or "top_tracks" not in data:
            return empty_chart("No track data for this selection")
        df = pd.read_json(StringIO(data["top_tracks"]), orient="split")
        if df.empty:
            return empty_chart("No track data for this selection")
        dff = df.head(10)
        y_vals = dff["track_name"].astype(str).tolist()
        y_ticktext = wrap_or_truncate_labels(y_vals)
        left_margin = compute_left_margin(y_ticktext)

        artists = (
            dff.get("track_artist")
            if "track_artist" in dff.columns
            else (dff.get("artist") if "artist" in dff.columns else pd.Series([""] * len(dff)))
        )

        series_data = [
            {
                "y": int(plays) if pd.notna(plays) else 0,
                "custom": {"track": str(track), "artist": str(artist)},
            }
            for track, artist, plays in zip(
                y_vals, artists.fillna(""), dff["play_count"].fillna(0), strict=False
            )
        ]

        # Highcharts options
        options = {
            "chart": {
                "type": "bar",
                "height": 440,
                "marginLeft": left_margin,
                "style": {"fontFamily": "Segoe UI, sans-serif"},
            },
            "title": {"text": None},
            "credits": {"enabled": False},
            "legend": {"enabled": False},
            "colors": ["#1DB954"],
            "xAxis": {
                "categories": y_ticktext,
                "title": {"text": None},
                "gridLineWidth": 0,
                "labels": {"style": {"fontSize": "10px"}},
            },
            "yAxis": {
                "title": {"text": "Play Count"},
                "gridLineWidth": 1,
            },
            "plotOptions": {
                "series": {
                    "dataLabels": {
                        "enabled": True,
                        "style": {
                            "textOutline": "none",
                        },
                        "crop": False,
                        "overflow": "allow",
                    }
                }
            },
            "tooltip": {
                "useHTML": True,
                "pointFormat": "Track: {point.custom.track}<br/>Artist: {point.custom.artist}<br/>Plays: {point.y}",
            },
            "series": [
                {
                    "name": "Plays",
                    "data": series_data,
                }
            ],
        }
        return options

    @app.callback(
        Output("top-artists-options", "data"),
        # theme-store is deliberately NOT an Input here: the clientside
        # renderer applies the theme and merges it over these options, so a
        # dark-mode flip must not re-run this server callback (it would
        # re-parse the whole filter payload and rebuild every chart).
        [Input("tab-1-data", "data")],
    )
    def render_top_artists_figure(data):
        if not data or "top_artists" not in data:
            return empty_chart("No artist data for this selection")
        df = pd.read_json(StringIO(data["top_artists"]), orient="split")
        if df.empty:
            return empty_chart("No artist data for this selection")
        dff = df.head(10)
        labels = dff["artist"].astype(str).tolist()
        ticktext = wrap_or_truncate_labels(labels)
        left_margin = compute_left_margin(ticktext)
        series_data = [
            {
                "y": int(pc) if pd.notna(pc) else 0,
                "custom": {"artist": str(a), "unique": int(ut) if pd.notna(ut) else 0},
            }
            for a, pc, ut in zip(
                labels,
                dff["play_count"].fillna(0),
                dff.get("unique_tracks", pd.Series([0] * len(dff))).fillna(0),
                strict=False,
            )
        ]
        return {
            "chart": {
                "type": "bar",
                "height": 440,
                "marginLeft": left_margin,
                "style": {"fontFamily": "Segoe UI, sans-serif"},
            },
            "title": {"text": None},
            "credits": {"enabled": False},
            "legend": {"enabled": False},
            "colors": ["#1DB954"],
            "xAxis": {
                "categories": ticktext,
                "labels": {"style": {"fontSize": "10px"}},
            },
            "yAxis": {
                "title": {"text": "Play Count"},
                "gridLineWidth": 1,
            },
            "plotOptions": {
                "series": {
                    "dataLabels": {
                        "enabled": True,
                        "style": {
                            "textOutline": "none",
                        },
                        "crop": False,
                        "overflow": "allow",
                    }
                }
            },
            "tooltip": {
                "useHTML": True,
                "pointFormat": "Artist: {point.custom.artist}<br/>Unique Tracks: {point.custom.unique}<br/>Plays: {point.y}",
            },
            "series": [{"name": "Plays", "data": series_data}],
        }

    @app.callback(
        Output("top-albums-options", "data"),
        # theme-store is deliberately NOT an Input here: the clientside
        # renderer applies the theme and merges it over these options, so a
        # dark-mode flip must not re-run this server callback (it would
        # re-parse the whole filter payload and rebuild every chart).
        [Input("tab-1-data", "data")],
    )
    def render_top_albums_figure(data):
        if not data or "top_albums" not in data:
            return empty_chart("No album data for this selection")
        df = pd.read_json(StringIO(data["top_albums"]), orient="split")
        if df.empty:
            return empty_chart("No album data for this selection")

        # Sort by album_score (already sorted, but ensure)
        if "album_score" in df.columns:
            df_sorted = df.sort_values("album_score", ascending=False)
        else:
            # Fallback if album_score not available
            return empty_chart("No albums met the minimum track count")

        dff = df_sorted.head(10)
        labels = dff["album_name"].astype(str).tolist()
        ticktext = wrap_or_truncate_labels(labels)
        left_margin = compute_left_margin(ticktext)

        # Get album_score values
        album_scores = (
            dff["album_score"].fillna(0.0)
            if "album_score" in dff.columns
            else pd.Series([0.0] * len(dff))
        )

        # Get MQPC for tooltip
        mqpc_values = (
            dff["mqpc"].fillna(0.0) if "mqpc" in dff.columns else pd.Series([0.0] * len(dff))
        )

        # Extract track_details if available
        track_details_list = []
        if "track_details" in dff.columns:
            for td in dff["track_details"]:
                if isinstance(td, list):
                    track_details_list.append(td)
                else:
                    track_details_list.append([])
        else:
            track_details_list = [[] for _ in range(len(dff))]

        series_data = [
            {
                "y": float(score) if pd.notna(score) else 0.0,
                "custom": {
                    "album": str(al),
                    "artist": str(ar),
                    "mqpc": float(mqpc) if pd.notna(mqpc) else 0.0,
                    "track_details": tracks,
                },
            }
            for al, ar, score, mqpc, tracks in zip(
                labels,
                dff.get("artist", pd.Series([""] * len(dff))).fillna(""),
                album_scores,
                mqpc_values,
                track_details_list,
                strict=False,
            )
        ]
        return {
            "chart": {
                "type": "bar",
                "height": 440,
                "marginLeft": left_margin,
                "style": {"fontFamily": "Segoe UI, sans-serif"},
            },
            "title": {"text": None},
            "credits": {"enabled": False},
            "legend": {"enabled": False},
            "colors": ["#1DB954"],
            "xAxis": {
                "categories": ticktext,
                "labels": {"style": {"fontSize": "10px"}},
            },
            "yAxis": {
                "title": {"text": "Album Score"},
                "gridLineWidth": 1,
            },
            "plotOptions": {
                "series": {
                    "cursor": "pointer",
                    "point": {
                        "events": {
                            "click": """function() {
                                var albumName = this.custom.album;
                                var artist = this.custom.artist;
                                var trackDetails = this.custom.track_details || [];

                                var detailsDiv = document.getElementById('album-track-details');
                                var titleDiv = document.getElementById('album-track-details-title');
                                var tableDiv = document.getElementById('album-track-details-table');

                                if (!detailsDiv || !titleDiv || !tableDiv) return;

                                // Toggle if clicking same album
                                if (window.__SELECTED_ALBUM === albumName && detailsDiv.style.display !== 'none') {
                                    detailsDiv.style.display = 'none';
                                    window.__SELECTED_ALBUM = null;
                                    return;
                                }

                                window.__SELECTED_ALBUM = albumName;
                                titleDiv.textContent = albumName + ' - ' + artist;

                                // Build table HTML
                                var html = '<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">';
                                html += '<thead><tr style="border-bottom: 2px solid #1DB954;">';
                                html += '<th style="text-align: left; padding: 8px;">Track</th>';
                                html += '<th style="text-align: right; padding: 8px;">Play Count</th>';
                                html += '</tr></thead><tbody>';

                                trackDetails.forEach(function(track, idx) {
                                    var bgColor = idx % 2 === 0 ? 'rgba(29, 185, 84, 0.05)' : 'transparent';
                                    html += '<tr style="background-color: ' + bgColor + ';">';
                                    html += '<td style="padding: 8px;">' + track.track_name + '</td>';
                                    html += '<td style="text-align: right; padding: 8px;">' + track.play_count + '</td>';
                                    html += '</tr>';
                                });

                                html += '</tbody></table>';
                                tableDiv.innerHTML = html;
                                detailsDiv.style.display = 'block';
                            }"""
                        }
                    },
                    "dataLabels": {
                        "enabled": True,
                        "style": {
                            "textOutline": "none",
                        },
                        "crop": False,
                        "overflow": "allow",
                        "format": "{y:.1f}",
                    },
                }
            },
            "tooltip": {
                "useHTML": True,
                "pointFormat": "Album: {point.custom.album}<br/>Artist: {point.custom.artist}<br/>Album Score: {point.y:.1f}<br/>MQPC: {point.custom.mqpc:.1f}",
            },
            "series": [{"name": "Album Score", "data": series_data}],
        }

    @app.callback(
        Output("top-genres-options", "data"),
        # theme-store is deliberately NOT an Input here: the clientside
        # renderer applies the theme and merges it over these options, so a
        # dark-mode flip must not re-run this server callback (it would
        # re-parse the whole filter payload and rebuild every chart).
        [Input("tab-1-data", "data")],
    )
    def render_top_genres_figure(data):
        if not data or ("top_genres" not in data and "top_genres_sunburst" not in data):
            return empty_chart("No genre data for this selection")
        # Prefer precomputed hierarchical rows when available
        if "top_genres_sunburst" in data:
            sb = pd.read_json(StringIO(data["top_genres_sunburst"]), orient="split")
        else:
            # Fallback: derive rows from flat top_genres + taxonomy
            df = pd.read_json(StringIO(data["top_genres"]), orient="split")
            if df.empty:
                return empty_chart("No genre data for this selection")
            with db_cursor() as con:
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
            return empty_chart("No genre data for this selection")

        def slug(s: str) -> str:
            return (s or "").lower().replace(" ", "-")

        data_points: list[dict] = []
        data_points.append({"id": "root", "name": "Genres"})
        for p in sorted({str(x) for x in sb["parent"].unique()}):
            if not p:
                continue
            data_points.append({"id": f"p::{slug(p)}", "parent": "root", "name": p})
        for _, r in sb.iterrows():
            p = str(r.get("parent", ""))
            c = str(r.get("child", ""))
            v = int(r.get("value", 0) or 0)
            if not p or not c or v <= 0:
                continue
            point = {
                "id": f"c::{slug(p)}::{slug(c)}",
                "parent": f"p::{slug(p)}",
                "name": c,
                "value": v,
            }
            if "top_artists" in sb.columns and pd.notna(r.get("top_artists")):
                try:
                    ta = r.get("top_artists")
                    ta_str = ""
                    if isinstance(ta, str):
                        ta_str = ta
                    elif isinstance(ta, list | tuple):
                        ta_str = ", ".join([str(x) for x in ta[:5]])
                    else:
                        try:
                            from collections.abc import Iterable

                            if isinstance(ta, Iterable) and not isinstance(ta, str | bytes | dict):
                                seq = list(ta)
                                ta_str = ", ".join([str(x) for x in seq[:5]])
                            elif ta is not None:
                                ta_str = str(ta)
                        except (TypeError, ValueError):
                            pass  # Leave top_artists off this wedge.
                    if ta_str:
                        point["custom"] = {"top_artists": ta_str}
                except (TypeError, ValueError, KeyError):
                    pass  # Tooltip extra only; the wedge still renders.
            data_points.append(point)
        return {
            "chart": {"height": 450},
            "title": {"text": None},
            "credits": {"enabled": False},
            "colors": ["#1DB954", "#1ed760", "#21e065", "#5eb859", "#7dd069", "#9be082", "#b5e8a3"],
            "tooltip": {
                "useHTML": True,
                # "Tagged plays", not "Plays": an artist carrying three genre
                # tags contributes a play to all three wedges, so these values
                # sum to more than the number of plays.
                "formatter": "function(){var ta=(this.point.custom&&this.point.custom.top_artists)?('<br/>Top Artists: '+this.point.custom.top_artists):'';return this.point.name+'<br/>Tagged plays: '+this.point.value+ta;}",
            },
            "series": [
                {
                    "type": "sunburst",
                    "data": data_points,
                    "allowDrillToNode": True,
                    "levels": [
                        {"level": 1, "dataLabels": {"rotationMode": "perpendicular"}},
                        {"level": 2, "colorByPoint": True},
                    ],
                }
            ],
        }

    @app.callback(
        Output("detailed-stats", "children"),
        Input("tab-1-data", "data"),
    )
    def render_stats_table(data):
        if not data or "stats_summary" not in data:
            return []

        stats = data["stats_summary"]
        return html.Table(
            [
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                html.Tbody([html.Tr([html.Td(k), html.Td(v)]) for k, v in stats.items()]),
            ],
            className="stats-table",
        )

    @app.callback(
        Output("daily-song-heatmap-options", "data"),
        # theme-store is deliberately NOT an Input here: the clientside
        # renderer applies the theme and merges it over these options, so a
        # dark-mode flip must not re-run this server callback (it would
        # re-parse the whole filter payload and rebuild every chart).
        [Input("tab-1-data", "data")],
    )
    def render_daily_heatmap(data):
        if not data or "daily_counts" not in data:
            return empty_chart("No daily play data for this selection")
        df = pd.read_json(StringIO(data["daily_counts"]), orient="split")
        if df.empty:
            return empty_chart("No daily play data for this selection")

        # Convert to line chart: aggregate by date and show top song play count per day
        # Also track all top songs (ties) for each day
        if "date" in df.columns and "play_count" in df.columns:
            # Get all top songs per date (all tracks with max play_count for that date)
            if "track" in df.columns and "artist" in df.columns:
                # Find max play_count per date
                max_plays_per_date = df.groupby("date")["play_count"].max().reset_index()
                max_plays_per_date.columns = ["date", "max_play_count"]

                # Merge to find all songs with max play_count for each date
                df_with_max = df.merge(max_plays_per_date, on="date", how="left")
                top_songs_df = df_with_max[
                    df_with_max["play_count"] == df_with_max["max_play_count"]
                ].copy()

                # Create track-artist pairs, handling NaN values
                def format_track_artist(row):
                    track = str(row["track"]) if pd.notna(row["track"]) else ""
                    artist = str(row["artist"]) if pd.notna(row["artist"]) else ""
                    if track and artist:
                        return f"{track} - {artist}"
                    elif track:
                        return track
                    elif artist:
                        return artist
                    else:
                        return ""

                top_songs_df["track_artist"] = top_songs_df.apply(format_track_artist, axis=1)

                # Filter out empty strings and group by date to collect all unique top songs
                top_songs_df = top_songs_df[top_songs_df["track_artist"].str.strip() != ""].copy()

                # Group by date and collect all unique top songs (as track-artist pairs)
                top_songs_by_date = (
                    top_songs_df.groupby("date")["track_artist"]
                    .apply(lambda x: list(x.unique()))
                    .reset_index()
                )
                top_songs_by_date.columns = ["date", "top_songs"]

                # Use max_play_count as the play count (top song's play count, not total)
                # This represents how many times the top song(s) were played
                daily_totals = max_plays_per_date.copy()
                daily_totals.rename(columns={"max_play_count": "play_count"}, inplace=True)

                # Merge to get top songs info
                daily_totals = daily_totals.merge(top_songs_by_date, on="date", how="left")

                # Filter: only show days where top song had more than 1 play
                daily_totals = daily_totals[daily_totals["play_count"] > 1].copy()
            else:
                # No track/artist info, just aggregate totals
                daily_totals = df.groupby("date")["play_count"].sum().reset_index()
                daily_totals["track"] = [[] for _ in range(len(daily_totals))]
                daily_totals["artist"] = [[] for _ in range(len(daily_totals))]
                daily_totals["max_play_count"] = daily_totals["play_count"]
                # Filter: only show days with more than 1 play
                daily_totals = daily_totals[daily_totals["play_count"] > 1].copy()

            daily_totals = daily_totals.sort_values("date")

            # Convert dates to timestamps for Highcharts
            daily_totals["timestamp"] = pd.to_datetime(daily_totals["date"]).astype(int) // 10**6

            # Create data points with custom info for tooltip
            data_pts = []
            for _idx, row in daily_totals.iterrows():
                # Get top songs (already formatted as "Track - Artist")
                top_songs = row.get("top_songs", [])

                # Handle different data types
                if top_songs is None:
                    top_songs = []
                elif hasattr(top_songs, "__len__") and not isinstance(top_songs, str | list):
                    # Handle pandas Series, numpy arrays, etc.
                    try:
                        if hasattr(top_songs, "tolist"):
                            top_songs = top_songs.tolist()
                        elif hasattr(top_songs, "values"):
                            top_songs = (
                                top_songs.values.tolist()
                                if hasattr(top_songs.values, "tolist")
                                else list(top_songs.values)
                            )
                        else:
                            top_songs = list(top_songs)
                    except Exception:
                        top_songs = []
                elif isinstance(top_songs, str) or not isinstance(top_songs, list):
                    top_songs = [top_songs] if top_songs else []

                # Check for NaN values in the list
                if isinstance(top_songs, list):
                    top_songs = [
                        s
                        for s in top_songs
                        if s is not None and not (isinstance(s, float) and pd.isna(s))
                    ]

                # Filter out empty strings and join
                top_songs_list = [
                    str(s) for s in top_songs if s and str(s).strip() and str(s) != "nan - nan"
                ]
                top_songs_str = ", ".join(top_songs_list) if top_songs_list else "N/A"

                point = {
                    "x": int(row["timestamp"]),
                    "y": int(row["play_count"]),
                    "custom": {
                        "top_songs": top_songs_str,
                    },
                }
                data_pts.append(point)
        else:
            # Fallback: use the grid data and aggregate
            grid_df = create_daily_top_playcount_grid(df)
            if "date" in grid_df.columns and "play_count" in grid_df.columns:
                # Get max play_count per date to filter
                max_plays_per_date = grid_df.groupby("date")["play_count"].max().reset_index()
                max_plays_per_date.columns = ["date", "max_play_count"]

                # Get all top tracks per date (all tracks with max play_count)
                def get_top_tracks(group):
                    max_plays = group["play_count"].max()
                    top_tracks = group[group["play_count"] == max_plays]["track"].tolist()
                    return top_tracks

                top_tracks_by_date = grid_df.groupby("date").apply(get_top_tracks).reset_index()
                top_tracks_by_date.columns = ["date", "top_tracks"]

                # Aggregate total plays per date
                daily_totals = grid_df.groupby("date")["play_count"].sum().reset_index()
                # Merge to get top tracks and max play count
                daily_totals = daily_totals.merge(top_tracks_by_date, on="date", how="left")
                daily_totals = daily_totals.merge(max_plays_per_date, on="date", how="left")

                # Filter: only show days where top song had more than 1 play
                daily_totals = daily_totals[daily_totals["max_play_count"] > 1].copy()
                daily_totals = daily_totals.sort_values("date")
                daily_totals["timestamp"] = (
                    pd.to_datetime(daily_totals["date"]).astype(int) // 10**6
                )

                data_pts = []
                for _, row in daily_totals.iterrows():
                    top_tracks = row.get("top_tracks", [])
                    if isinstance(top_tracks, list):
                        top_songs_str = ", ".join([str(t) for t in top_tracks if t])
                    else:
                        top_songs_str = str(top_tracks) if top_tracks else ""

                    point = {
                        "x": int(row["timestamp"]),
                        "y": int(row["play_count"]),
                        "custom": {
                            "top_songs": top_songs_str,
                        },
                    }
                    data_pts.append(point)
            else:
                data_pts = []

        return {
            "chart": {
                "type": "line",
                "height": 400,
            },
            "title": {"text": "Daily Song Plays Over Time"},
            "credits": {"enabled": False},
            "xAxis": {
                "type": "datetime",
                "title": {"text": "Date"},
            },
            "yAxis": {
                "title": {"text": "Top Song Play Count"},
            },
            "tooltip": {
                "useHTML": True,
                "formatter": "function(){var custom=this.point.custom||{};var topSongs=custom.top_songs||'N/A';return '<b>Date:</b> '+Highcharts.dateFormat('%Y-%m-%d', this.x)+'<br/><b>Top Song Play Count:</b> '+this.y+'<br/><b>Top Song(s):</b> '+topSongs;}",
            },
            "plotOptions": {
                "line": {
                    "color": "#1DB954",
                    "lineWidth": 2,
                    "marker": {
                        "enabled": True,
                        "radius": 3,
                    },
                }
            },
            "series": [
                {
                    "name": "Daily Plays",
                    "data": data_pts,
                }
            ],
        }

    # Each options store above is drawn by the same clientside renderer.
    for container_id, options_id in (
        ("top-tracks-container", "top-tracks-options"),
        ("top-artists-container", "top-artists-options"),
        ("top-albums-container", "top-albums-options"),
        ("top-genres-container", "top-genres-options"),
        ("daily-song-heatmap-container", "daily-song-heatmap-options"),
    ):
        register_highcharts_renderer(app, container_id, options_id)
