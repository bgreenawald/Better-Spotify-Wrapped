import pandas as pd
import plotly.express as px
from dash import dcc, html


def create_graph_style():
    return {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Segoe UI, sans-serif"},
        "margin": {"t": 30, "b": 30, "l": 30, "r": 30},
    }


def create_top_tracks_graph(top_tracks: pd.DataFrame = None):
    layout = {
        **create_graph_style(),
        "xaxis": {"gridcolor": "#eee"},
        "yaxis": {"gridcolor": "#eee"},
    }

    if top_tracks is not None:
        fig = px.bar(
            top_tracks,
            x="play_count",
            y="track_name",
            text="play_count",
            orientation="h",
            labels={"track_name": "Track", "play_count": "Play Count"},
            hover_data=["artist"],
        )
        fig.update_layout(**layout)
        fig.update_traces(marker_color="#1DB954")  # Spotify green

    return html.Div(
        [
            html.H3("Most Played Tracks", className="card-title"),
            dcc.Graph(
                id="top-tracks-graph",
                figure=fig if top_tracks is not None else {},
                config={"displayModeBar": False},
            ),
        ],
        className="graph-card card",
    )


def create_top_albums_graph(top_albums: pd.DataFrame = None):
    layout = {
        **create_graph_style(),
        "xaxis": {"gridcolor": "#eee"},
        "yaxis": {"gridcolor": "#eee"},
    }

    if top_albums is not None:
        fig = px.bar(
            top_albums,
            x="median_plays",
            y="album_name",
            text="median_plays",
            orientation="h",
            labels={"album_name": "Album", "median_plays": "Median Plays"},
            hover_data=["artist", "tracks_played", "total_tracks"],
        )
        fig.update_layout(**layout)
        fig.update_traces(marker_color="#1DB954")  # Spotify green

    return html.Div(
        [
            html.H3("Top Albums", className="card-title"),
            dcc.Graph(
                id="top-albums-graph",
                figure=fig if top_albums is not None else {},
                config={"displayModeBar": False},
            ),
        ],
        className="graph-card card",
    )


def create_graphs_section():
    return html.Div(
        [create_top_tracks_graph(), create_top_albums_graph()],
        className="graph-container",
    )
