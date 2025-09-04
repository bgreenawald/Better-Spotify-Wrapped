import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html


def create_graph_style():
    """Return a standardized Plotly layout dictionary."""
    return {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Segoe UI, sans-serif"},
        "margin": {"t": 30, "b": 30, "l": 200, "r": 30},
        "height": 450,
        "yaxis": {
            "gridcolor": "#eee",
            "automargin": True,
            "tickfont": {"size": 11},
        },
        "xaxis": {"gridcolor": "#eee", "automargin": True},
    }


def create_top_tracks_graph(top_tracks=None):
    """Create a horizontal bar chart of the most played tracks.

    Args:
        top_tracks (pd.DataFrame, optional): DataFrame with columns
            'track_name', 'play_count', and 'artist'. Defaults to None.

    Returns:
        html.Div: Dash container with the bar chart.
    """
    graph_layout = create_graph_style()

    if top_tracks is not None:
        df = top_tracks.head(10)
        fig = px.bar(
            df,
            x="play_count",
            y="track_name",
            text="play_count",
            orientation="h",
            labels={"track_name": "Track", "play_count": "Play Count"},
            hover_data=["artist"],
        )
        fig.update_layout(**graph_layout)
        fig.update_traces(marker_color="#1DB954")

        # Truncate long track names for readability
        truncated = [f"{name[:50]}..." if len(name) > 50 else name for name in df["track_name"]]
        fig.update_yaxes(
            ticktext=truncated,
            tickvals=list(range(len(df))),
        )

    return html.Div(
        className="graph-card card",
        children=[
            dcc.Loading(
                children=[
                    html.H3("Most Played Tracks", className="card-title"),
                    dcc.Graph(
                        id="top-tracks-graph",
                        figure=fig if top_tracks is not None else {},
                        config={"displayModeBar": False},
                    ),
                ]
            )
        ],
    )


def create_top_artists_graph(top_artists=None):
    """Create a horizontal bar chart of the top artists by play count.

    Args:
        top_artists (pd.DataFrame, optional): DataFrame with columns
            'artist', 'play_count', and 'unique_tracks'. Defaults to None.

    Returns:
        html.Div: Dash container with the bar chart.
    """
    graph_layout = create_graph_style()

    if top_artists is not None:
        df = top_artists.head(10)
        fig = px.bar(
            df,
            x="play_count",
            y="artist",
            text="play_count",
            orientation="h",
            labels={"artist": "Artist", "play_count": "Play Count"},
            hover_data=["unique_tracks"],
        )
        fig.update_layout(**graph_layout)
        fig.update_traces(marker_color="#1DB954")

    return html.Div(
        className="graph-card card",
        children=[
            dcc.Loading(
                children=[
                    html.H3("Top Artists", className="card-title"),
                    dcc.Graph(
                        id="top-artists-graph",
                        figure=fig if top_artists is not None else {},
                        config={"displayModeBar": False},
                    ),
                ]
            )
        ],
    )


def create_top_genres_graph(top_genres=None):
    """Create a two-level sunburst of genres using parent hierarchy.

    Notes:
        - Inner ring: parent genres (level 0 in DDL).
        - Outer ring: level 1 sub-genres. Sub-genres mapped to multiple
          parents are duplicated (counted once per parent).

    Args:
        top_genres (pd.DataFrame, optional): If provided, should contain columns
            'genre' and 'play_count' (or 'track_count'). When absent, the
            figure is supplied by the callbacks.

    Returns:
        html.Div: Dash container with the sunburst chart.
    """
    graph_layout = create_graph_style()

    fig = {}
    if top_genres is not None and not getattr(top_genres, "empty", True):
        # Build a simple fallback sunburst directly from flat counts in case this
        # helper is used outside of callbacks. This does not resolve DB taxonomy.
        df = top_genres.copy()
        # Normalize column name
        if "track_count" in df.columns and "play_count" not in df.columns:
            df = df.rename(columns={"track_count": "play_count"})
        # Use a flat ring by treating each genre as its own parent; callbacks will
        # provide the full two-level taxonomy-aware figure.
        sb_df = pd.DataFrame(
            {
                "parent": df["genre"],
                "child": df["genre"],
                "value": df["play_count"],
            }
        )
        fig = px.sunburst(sb_df, path=["parent", "child"], values="value")
        fig.update_layout(**graph_layout)

    return html.Div(
        className="graph-card card",
        children=[
            dcc.Loading(
                children=[
                    html.H3("Top Genres", className="card-title"),
                    dcc.Graph(
                        id="top-genres-graph",
                        figure=fig,
                        config={"displayModeBar": False},
                    ),
                ]
            )
        ],
    )


def create_top_albums_graph(top_albums=None):
    """Create a horizontal bar chart of the top albums by median plays.

    Args:
        top_albums (pd.DataFrame, optional): DataFrame with columns
            'album_name', 'median_plays', 'artist', 'tracks_played',
            and 'total_tracks'. Defaults to None.

    Returns:
        html.Div: Dash container with the bar chart.
    """
    graph_layout = create_graph_style()

    if top_albums is not None:
        df = top_albums.head(10)
        fig = px.bar(
            df,
            x="median_plays",
            y="album_name",
            text="median_plays",
            orientation="h",
            labels={"album_name": "Album", "median_plays": "Median Plays"},
            hover_data=["artist", "tracks_played", "total_tracks"],
        )
        fig.update_layout(**graph_layout)
        fig.update_traces(marker_color="#1DB954")

        # Truncate long album names for readability
        truncated = [f"{name[:50]}..." if len(name) > 50 else name for name in df["album_name"]]
        fig.update_yaxes(
            ticktext=truncated,
            tickvals=list(range(len(df))),
        )

    return html.Div(
        className="graph-card card",
        children=[
            dcc.Loading(
                children=[
                    html.H3("Top Albums", className="card-title"),
                    dcc.Graph(
                        id="top-albums-graph",
                        figure=fig if top_albums is not None else {},
                        config={"displayModeBar": False},
                    ),
                ]
            )
        ],
    )


def create_graphs_section_tab_one():
    """Compose the first tab section with tracks, artists, albums, and genres."""
    return html.Div(
        children=[
            # Row 1: Tracks and Artists
            html.Div(
                children=[
                    create_top_tracks_graph(),
                    create_top_artists_graph(),
                ],
                className="graph-container",
            ),
            # Row 2: Albums and Genres
            html.Div(
                children=[
                    create_top_albums_graph(),
                    create_top_genres_graph(),
                ],
                className="graph-container",
            ),
        ]
    )


def create_daily_top_playcount_grid(daily_playcounts):
    """Generate a grid layout DataFrame for daily top track play counts.

    Args:
        daily_playcounts (pd.DataFrame): DataFrame with columns 'date',
            'track', 'artist', and 'play_count'.

    Returns:
        pd.DataFrame: DataFrame with 'row', 'col', 'date', 'track', and
            'play_count' for heatmap plotting.
    """
    df = daily_playcounts.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Get the top track per date
    top_per_date = (
        df.sort_values(["date", "play_count"], ascending=[True, False])
        .drop_duplicates("date")
        .set_index("date")[["track", "artist", "play_count"]]
    )

    # Ensure a continuous date index
    full_index = pd.date_range(
        start=top_per_date.index.min(),
        end=top_per_date.index.max(),
        freq="D",
    )
    top_per_date = top_per_date.reindex(full_index)

    # Fill missing values
    top_per_date["play_count"] = top_per_date["play_count"].fillna(0).astype(int)
    top_per_date["track"] = top_per_date["track"].fillna("")
    top_per_date["artist"] = top_per_date["artist"].fillna("")

    # Combine track and artist for display
    top_per_date["track_artist"] = top_per_date["track"] + " - " + top_per_date["artist"]

    # Create grid positions
    total_days = len(top_per_date)
    columns = 10
    output = pd.DataFrame(
        {
            "date": full_index,
            "track": top_per_date["track_artist"].values,
            "play_count": top_per_date["play_count"].values,
        }
    )
    output["position"] = np.arange(total_days)
    output["row"] = output["position"] // columns
    output["col"] = output["position"] % columns

    return output[["row", "col", "date", "track", "play_count"]]


def create_daily_top_heatmap(
    daily_playcounts=None,
    theme=None,
    title="Daily Top-Track Play Count Heatmap",  # noqa: ARG001
):
    """Create a heatmap figure of daily top track play counts.

    Args:
        daily_playcounts (pd.DataFrame, optional): DataFrame with daily
            play count data. Defaults to None.
        theme (dict, optional): Theme configuration for the graph. Defaults to None.
        title (str, optional): Title of the heatmap (unused). Defaults to
            "Daily Top-Track Play Count Heatmap".

    Returns:
        go.Figure or html.Div: Plotly Figure or a Div with a message if no
            data is available.
    """
    if daily_playcounts is None or daily_playcounts.empty:
        return html.Div(children="No data available", className="graph-card card")

    # Prepare data grid
    grid_df = create_daily_top_playcount_grid(daily_playcounts)

    # Pivot data into matrices
    z_matrix = grid_df.pivot(index="row", columns="col", values="play_count").values
    date_matrix = grid_df.pivot(index="row", columns="col", values="date").astype(str).values
    track_matrix = grid_df.pivot(index="row", columns="col", values="track").values

    # Build customdata for hover information
    customdata = np.dstack([date_matrix, track_matrix])

    # Determine colorscale based on theme
    is_dark = theme and theme.get("template") == "plotly_dark"
    colorscale = "Viridis" if is_dark else "Greens"

    heatmap = go.Heatmap(
        z=z_matrix,
        x=list(range(z_matrix.shape[1])),
        y=list(range(z_matrix.shape[0])),
        colorscale=colorscale,
        customdata=customdata,
        hovertemplate=(
            "Date: %{customdata[0]}<br>Track: %{customdata[1]}<br>Plays: %{z}<extra></extra>"
        ),
        colorbar={
            "title": "Plays",
            "tickfont": {"color": "#e0e0e0" if is_dark else "#000000"},
            "titlefont": {"color": "#e0e0e0" if is_dark else "#000000"},
        },
    )

    fig = go.Figure(heatmap)

    # Apply theme or default style
    if theme:
        # Create a deep copy of theme to avoid modifying the original
        layout_update = copy.deepcopy(theme)
        # Override specific axis settings for heatmap
        layout_update["xaxis"] = {
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "gridcolor": theme.get("xaxis", {}).get("gridcolor", "#333"),
        }
        layout_update["yaxis"] = {
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "gridcolor": theme.get("yaxis", {}).get("gridcolor", "#333"),
        }
        layout_update["height"] = 400
        layout_update["margin"] = {"t": 30, "b": 30, "l": 30, "r": 80}
        fig.update_layout(**layout_update)
    else:
        layout_style = create_graph_style()
        fig.update_layout(**layout_style)

    return fig
