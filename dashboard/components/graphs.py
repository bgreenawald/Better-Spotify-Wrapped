import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html


def create_graph_style():
    return {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Segoe UI, sans-serif"},
        "margin": {"t": 30, "b": 30, "l": 200, "r": 30},  # Increased left margin
        "height": 450,  # Fixed height to accommodate more items
        "yaxis": {
            "gridcolor": "#eee",
            "automargin": True,  # Automatically adjust margin for labels
            "tickfont": {"size": 11},  # Slightly smaller font size for labels
        },
        "xaxis": {"gridcolor": "#eee", "automargin": True},
    }


def create_top_tracks_graph(top_tracks: pd.DataFrame = None):
    layout = create_graph_style()

    if top_tracks is not None:
        fig = px.bar(
            top_tracks.head(10),
            x="play_count",
            y="track_name",
            text="play_count",
            orientation="h",
            labels={"track_name": "Track", "play_count": "Play Count"},
            hover_data=["artist"],
        )
        fig.update_layout(**layout)
        fig.update_traces(marker_color="#1DB954")
        # Truncate long track names
        fig.update_yaxes(
            ticktext=[
                f"{text[:50]}..." if len(text) > 50 else text
                for text in top_tracks["track_name"].head(10)
            ],
            tickvals=list(range(len(top_tracks.head(10)))),
        )

    return html.Div(
        [
            dcc.Loading(
                [
                    html.H3("Most Played Tracks", className="card-title"),
                    dcc.Graph(
                        id="top-tracks-graph",
                        figure=fig if top_tracks is not None else {},
                        config={"displayModeBar": False},
                    ),
                ]
            )
        ],
        className="graph-card card",
    )


def create_top_artists_graph(top_artists: pd.DataFrame = None):
    layout = create_graph_style()

    if top_artists is not None:
        fig = px.bar(
            top_artists.head(10),
            x="play_count",
            y="artist",
            text="play_count",
            orientation="h",
            labels={"artist": "Artist", "play_count": "Play Count"},
            hover_data=["unique_tracks"],
        )
        fig.update_layout(**layout)
        fig.update_traces(marker_color="#1DB954")

    return html.Div(
        [
            dcc.Loading(
                [
                    html.H3("Top Artists", className="card-title"),
                    dcc.Graph(
                        id="top-artists-graph",
                        figure=fig if top_artists is not None else {},
                        config={"displayModeBar": False},
                    ),
                ]
            )
        ],
        className="graph-card card",
    )


def create_top_genres_graph(top_genres: pd.DataFrame = None):
    layout = create_graph_style()

    if top_genres is not None:
        fig = px.bar(
            top_genres.head(10),
            x="track_count",
            y="genre",
            text=["{}%".format(x) for x in top_genres["percentage"].head(10)],
            orientation="h",
            labels={"genre": "Genre", "track_count": "Track Count"},
            hover_data=["top_artists", "percentage"],
        )
        fig.update_layout(**layout)
        fig.update_traces(marker_color="#1DB954")

    return html.Div(
        [
            dcc.Loading(
                [
                    html.H3("Top Genres", className="card-title"),
                    dcc.Graph(
                        id="top-genres-graph",
                        figure=fig if top_genres is not None else {},
                        config={"displayModeBar": False},
                    ),
                ],
            )
        ],
        className="graph-card card",
    )


def create_top_albums_graph(top_albums: pd.DataFrame = None):
    layout = create_graph_style()

    if top_albums is not None:
        fig = px.bar(
            top_albums.head(10),
            x="median_plays",
            y="album_name",
            text="median_plays",
            orientation="h",
            labels={"album_name": "Album", "median_plays": "Median Plays"},
            hover_data=["artist", "tracks_played", "total_tracks"],
        )
        fig.update_layout(**layout)
        fig.update_traces(marker_color="#1DB954")
        # Truncate long album names
        fig.update_yaxes(
            ticktext=[
                f"{text[:50]}..." if len(text) > 50 else text
                for text in top_albums["album_name"].head(10)
            ],
            tickvals=list(range(len(top_albums.head(10)))),
        )

    return html.Div(
        [
            dcc.Loading(
                [
                    html.H3("Top Albums", className="card-title"),
                    dcc.Graph(
                        id="top-albums-graph",
                        figure=fig if top_albums is not None else {},
                        config={"displayModeBar": False},
                    ),
                ]
            )
        ],
        className="graph-card card",
    )


def create_graphs_section_tab_one():
    return html.Div(
        [
            # Row 1: Tracks and Artists
            html.Div(
                [create_top_tracks_graph(), create_top_artists_graph()],
                className="graph-container",
            ),
            # Row 2: Albums and Genres
            html.Div(
                [create_top_albums_graph(), create_top_genres_graph()],
                className="graph-container",
            ),
        ]
    )


def create_daily_top_playcount_grid(daily_playcounts: pd.DataFrame) -> pd.DataFrame:
    df = daily_playcounts.copy()
    df["date"] = pd.to_datetime(df["date"])
    top = (
        df.sort_values(["date", "play_count"], ascending=[True, False])
        .drop_duplicates("date")
        .set_index("date")[["track", "artist", "play_count"]]
    )

    full_idx = pd.date_range(top.index.min(), top.index.max(), freq="D")
    top = top.reindex(full_idx)
    top["play_count"] = top["play_count"].fillna(0).astype(int)
    top["track"] = top["track"].fillna("")
    top["artist"] = top["artist"].fillna("")
    top["track_artist"] = top["track"] + " - " + top["artist"]

    n = len(top)
    cols = 10
    out = pd.DataFrame(
        {
            "date": full_idx,
            "track": top["track_artist"].values,
            "play_count": top["play_count"].values,
        }
    )
    out["pos"] = np.arange(n)
    out["row"] = out["pos"] // cols
    out["col"] = out["pos"] % cols

    return out[["row", "col", "date", "track", "play_count"]]


def create_daily_top_heatmap(
    daily_playcounts: pd.DataFrame = None,
    title: str = "Daily Top-Track Play Count Heatmap",
):
    layout = create_graph_style()
    if daily_playcounts is None or daily_playcounts.empty:
        return html.Div("No data available", className="graph-card card")

    grid_df = create_daily_top_playcount_grid(daily_playcounts)

    # Pivot into matrices using keyword args
    z_mat = grid_df.pivot(index="row", columns="col", values="play_count").values
    date_mat = (
        grid_df.pivot(index="row", columns="col", values="date").astype(str).values
    )
    track_mat = grid_df.pivot(index="row", columns="col", values="track").values

    # Build a 3D customdata array of shape (rows, cols, 2)
    customdata = np.dstack([date_mat, track_mat])

    heat = go.Heatmap(
        z=z_mat,
        x=list(range(z_mat.shape[1])),
        y=list(range(z_mat.shape[0])),
        colorscale="Greens",
        customdata=customdata,
        hovertemplate=(
            "Date: %{customdata[0]}<br>"
            "Track: %{customdata[1]}<br>"
            "Plays: %{z}<extra></extra>"
        ),
        colorbar=dict(title="Plays"),
    )
    fig = go.Figure(heat)
    fig.update_layout(**layout)

    return fig
