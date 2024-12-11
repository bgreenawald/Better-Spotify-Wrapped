from datetime import datetime

import pandas as pd
from dash import Input, Output

from dashboard.components.graphs import create_graph_style
from dashboard.components.stats import create_stats_table
from src.metrics import (
    get_most_played_artists,
    get_most_played_tracks,
    get_top_albums,
    get_top_artist_genres,
)
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
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
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
