import pandas as pd
from dash import Input, Output

from dashboard.components.graphs import create_top_albums_graph, create_top_tracks_graph
from dashboard.components.stats import create_stats_table
from src.metrics import get_most_played_tracks, get_top_albums
from src.preprocessing import filter_songs


def register_callbacks(app, df: pd.DataFrame, spotify_data):
    @app.callback(
        [
            Output("top-tracks-graph", "figure"),
            Output("top-albums-graph", "figure"),
            Output("detailed-stats", "children"),
        ],
        [
            Input("year-dropdown", "value"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
        ],
    )
    def update_dashboard(selected_year, exclude_december, remove_incognito):
        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            year=selected_year,
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
        )

        # Get top tracks
        top_tracks = get_most_played_tracks(filtered_df).head(10)
        tracks_fig = create_top_tracks_graph(top_tracks).children[1].figure

        # Get top albums
        top_albums = get_top_albums(filtered_df, spotify_data).head(10)
        albums_fig = create_top_albums_graph(top_albums).children[1].figure

        # Create stats table
        stats_table = create_stats_table(filtered_df)

        return tracks_fig, albums_fig, stats_table
