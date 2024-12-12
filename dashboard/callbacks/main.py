from datetime import datetime

import pandas as pd
import plotly.express as px
from dash import Input, Output

from dashboard.components.graphs import create_graph_style
from dashboard.components.stats import create_stats_table
from src.metrics.metrics import (
    get_most_played_artists,
    get_most_played_tracks,
    get_top_albums,
    get_top_artist_genres,
)
from src.metrics.trends import get_listening_time_by_month
from src.preprocessing import filter_songs


def register_callbacks(app, df: pd.DataFrame, spotify_data):
    @app.callback(
        [
            Output("top-tracks-graph", "figure"),
            Output("top-artists-graph", "figure"),
            Output("top-albums-graph", "figure"),
            Output("top-genres-graph", "figure"),
            Output("detailed-stats", "children"),
        ],
        [
            Input("year-dropdown", "value"),
            Input("exclude-december-tab-one", "value"),
            Input("remove-incognito-tab-one", "value"),
        ],
    )
    def update_dashboard(selected_year, exclude_december, remove_incognito):
        # Check if a valid year is selected
        if not selected_year:
            # Return empty figures if no year is selected
            empty_figure = {"data": [], "layout": create_graph_style()}
            return empty_figure, empty_figure, empty_figure, empty_figure, []

        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            start_date=pd.Timestamp(datetime(selected_year, 1, 1)),
            end_date=pd.Timestamp(datetime(selected_year, 12, 31)),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
        )

        # Get top tracks
        top_tracks = get_most_played_tracks(filtered_df)
        tracks_fig = {
            "data": [
                {
                    "type": "bar",
                    "x": top_tracks["play_count"].head(10),
                    "y": top_tracks["track_name"].head(10),
                    "orientation": "h",
                    "text": top_tracks["play_count"].head(10),
                    "marker": {"color": "#1DB954"},
                }
            ],
            "layout": {
                **create_graph_style(),
                "xaxis": {"gridcolor": "#eee", "title": "Play Count"},
                "yaxis": {"gridcolor": "#eee", "title": ""},
            },
        }

        # Get top artists
        top_artists = get_most_played_artists(filtered_df)
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
                    "hovertemplate": "Artist: %{y}<br>Plays: %{x}<br>Unique Tracks: %{customdata}<extra></extra>",
                }
            ],
            "layout": {
                **create_graph_style(),
                "xaxis": {"gridcolor": "#eee", "title": "Play Count"},
                "yaxis": {"gridcolor": "#eee", "title": ""},
            },
        }

        # Get top albums
        top_albums = get_top_albums(filtered_df, spotify_data)
        albums_fig = {
            "data": [
                {
                    "type": "bar",
                    "x": top_albums["median_plays"].head(10),
                    "y": top_albums["album_name"].head(10),
                    "orientation": "h",
                    "text": top_albums["median_plays"].round(1).head(10),
                    "customdata": top_albums[
                        ["artist", "tracks_played", "total_tracks"]
                    ]
                    .head(10)
                    .values,
                    "marker": {"color": "#1DB954"},
                    "hovertemplate": "Album: %{y}<br>Median Plays: %{x}<br>Artist: %{customdata[0]}<br>Tracks Played: %{customdata[1]}/{customdata[2]}<extra></extra>",
                }
            ],
            "layout": {
                **create_graph_style(),
                "xaxis": {"gridcolor": "#eee", "title": "Median Plays"},
                "yaxis": {"gridcolor": "#eee", "title": ""},
            },
        }

        # Get top genres
        top_genres = get_top_artist_genres(filtered_df, spotify_data)
        genres_fig = {
            "data": [
                {
                    "type": "bar",
                    "x": top_genres["track_count"].head(10),
                    "y": top_genres["genre"].head(10),
                    "orientation": "h",
                    "text": [f"{x:.1f}%" for x in top_genres["percentage"].head(10)],
                    "customdata": top_genres[["percentage", "top_artists"]]
                    .head(10)
                    .values,
                    "marker": {"color": "#1DB954"},
                    "hovertemplate": "Genre: %{y}<br>Tracks: %{x}<br>Percentage: %{customdata[0]:.1f}%<br>Top Artists: %{customdata[1]}<extra></extra>",
                }
            ],
            "layout": {
                **create_graph_style(),
                "xaxis": {"gridcolor": "#eee", "title": "Track Count"},
                "yaxis": {"gridcolor": "#eee", "title": ""},
            },
        }

        # Create stats table
        stats_table = create_stats_table(filtered_df)

        return tracks_fig, artists_fig, albums_fig, genres_fig, stats_table

    @app.callback(
        Output("trends-graph", "figure"),
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december-tab-two", "value"),
            Input("remove-incognito-tab-two", "value"),
            Input("metric-dropdown", "value"),
        ],
    )
    def update_trend_dashboard(
        start_date, end_date, exclude_december, remove_incognito, selected_metric
    ):
        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
        )

        # Calculate monthly statistics
        monthly_stats = get_listening_time_by_month(filtered_df)

        # Create the figure using plotly express
        metric_labels = {
            "total_hours": "Total Listening Hours",
            "unique_tracks": "Unique Tracks",
            "unique_artists": "Unique Artists",
            "avg_hours_per_day": "Average Hours per Day",
        }

        fig = px.line(
            monthly_stats,
            x="month",
            y=selected_metric,
            labels={"month": "Month", selected_metric: metric_labels[selected_metric]},
            title=f"Monthly {metric_labels[selected_metric]}",
        )

        # Update layout
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Segoe UI, sans-serif"},
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
            xaxis={"gridcolor": "#eee"},
            yaxis={"gridcolor": "#eee"},
            showlegend=False,
        )

        # Update line color to match Spotify theme
        fig.update_traces(line_color="#1DB954")

        return fig
