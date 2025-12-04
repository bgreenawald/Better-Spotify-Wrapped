import numpy as np
import pandas as pd
from dash import dcc, html
from dash.development.base_component import Component


def create_top_tracks_graph(_top_tracks: pd.DataFrame | None = None) -> Component:
    """Create the container and store for 'Most Played Tracks' (Highcharts).

    This replaces the previous Plotly Graph with a simple container div that
    will be rendered clientside via Highcharts, and a Store that carries the
    Highcharts options produced by the server callback.
    """
    # For initial load, we no longer create a Plotly figure. The Highcharts
    # options will be populated into the Store by a callback and rendered
    # clientside into the container below.
    return html.Div(
        className="graph-card card",
        children=[
            html.H3("Most Played Tracks", className="card-title"),
            dcc.Store(id="top-tracks-options"),
            dcc.Loading(
                children=html.Div(
                    id="top-tracks-container",
                    children=html.Div(id="top-tracks-container-root"),
                    style={"minHeight": "440px"},
                ),
                delay_show=0,
                overlay_style={
                    "visibility": "visible",
                    "backgroundColor": "rgba(0,0,0,0.15)",
                },
                type="default",
            ),
        ],
    )


def create_top_artists_graph(_top_artists: pd.DataFrame | None = None) -> Component:
    """Create container + store for Top Artists (Highcharts)."""
    return html.Div(
        className="graph-card card",
        children=[
            html.H3("Top Artists", className="card-title"),
            dcc.Store(id="top-artists-options"),
            dcc.Loading(
                children=html.Div(
                    id="top-artists-container",
                    children=html.Div(id="top-artists-container-root"),
                    style={"minHeight": "440px"},
                ),
                delay_show=0,
                overlay_style={
                    "visibility": "visible",
                    "backgroundColor": "rgba(0,0,0,0.15)",
                },
                type="default",
            ),
        ],
    )


def create_top_genres_graph(_top_genres: pd.DataFrame | None = None) -> Component:
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
    # Highcharts sunburst will be rendered clientside; provide store + container
    return html.Div(
        className="graph-card card",
        children=[
            html.H3("Top Genres", className="card-title"),
            dcc.Store(id="top-genres-options"),
            dcc.Loading(
                children=html.Div(
                    id="top-genres-container",
                    children=html.Div(id="top-genres-container-root"),
                    style={"minHeight": "450px"},
                ),
                delay_show=0,
                overlay_style={
                    "visibility": "visible",
                    "backgroundColor": "rgba(0,0,0,0.15)",
                },
                type="default",
            ),
        ],
    )


def create_top_albums_graph(_top_albums: pd.DataFrame | None = None) -> Component:
    """Create container + store for Top Albums (Highcharts) with expandable track details."""
    return html.Div(
        className="graph-card card",
        children=[
            html.H3("Top Albums", className="card-title"),
            dcc.Store(id="top-albums-options"),
            dcc.Loading(
                children=html.Div(
                    id="top-albums-container",
                    children=html.Div(id="top-albums-container-root"),
                    style={"minHeight": "440px"},
                ),
                delay_show=0,
                overlay_style={
                    "visibility": "visible",
                    "backgroundColor": "rgba(0,0,0,0.15)",
                },
                type="default",
            ),
            html.Div(
                id="album-track-details",
                style={"marginTop": "20px", "display": "none"},
                children=[
                    html.H4(id="album-track-details-title", style={"marginBottom": "10px"}),
                    html.Div(id="album-track-details-table"),
                ],
            ),
        ],
    )


def create_graphs_section_tab_one() -> Component:
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


def create_daily_top_playcount_grid(daily_playcounts: pd.DataFrame) -> pd.DataFrame:
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
