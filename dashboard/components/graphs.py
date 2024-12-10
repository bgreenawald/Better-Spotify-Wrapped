import pandas as pd
import plotly.express as px
from dash import dcc, html


def create_top_tracks_graph(top_tracks: pd.DataFrame):
    fig = px.bar(
        top_tracks,
        x="play_count",
        y="track_name",
        text="play_count",
        orientation="h",
        title="Top 10 Most Played Tracks",
        labels={"track_name": "Track", "play_count": "Play Count"},
        hover_data=["artist"],
    )
    return html.Div(
        [html.H3("Most Played Tracks"), dcc.Graph(id="top-tracks-graph", figure=fig)],
        style={"width": "48%", "display": "inline-block", "marginRight": "2%"},
    )


def create_top_albums_graph(top_albums: pd.DataFrame):
    fig = px.bar(
        top_albums,
        x="median_plays",
        y="album_name",
        text="median_plays",
        orientation="h",
        title="Top 10 Albums by Median Plays",
        labels={"album_name": "Album", "median_plays": "Median Plays"},
        hover_data=["artist", "tracks_played", "total_tracks"],
    )
    return html.Div(
        [html.H3("Top Albums"), dcc.Graph(id="top-albums-graph", figure=fig)],
        style={"width": "48%", "display": "inline-block"},
    )


def create_graphs_section():
    return html.Div(
        [
            html.Div(
                [html.H3("Most Played Tracks"), dcc.Graph(id="top-tracks-graph")],
                style={"width": "48%", "display": "inline-block", "marginRight": "2%"},
            ),
            html.Div(
                [html.H3("Top Albums"), dcc.Graph(id="top-albums-graph")],
                style={"width": "48%", "display": "inline-block"},
            ),
        ]
    )
