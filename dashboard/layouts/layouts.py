import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.development.base_component import Component

from dashboard.components.filters import (
    create_artist_trends_layout,
    create_genre_trends_layout,
    create_global_settings,
    create_monthly_trend_filter,
    create_track_trends_layout,
    create_trend_filters_section,
    create_wrapped_filters_section,
)
from dashboard.components.graphs import create_graphs_section_tab_one


def create_layout(df: pd.DataFrame, spotify_data: pd.DataFrame) -> Component:
    """Generate the main dashboard layout.

    The layout includes a header, a global settings panel, and two tabs:
    Wrapped and Trends.

    Args:
        df (pd.DataFrame): User listening history DataFrame.
        spotify_data (pd.DataFrame): Additional Spotify data for trends.

    Returns:
        Component: Dash HTML component for the dashboard layout.
    """
    return html.Div(
        [
            html.Div(
                [
                    # Store for theme state
                    dcc.Store(id="theme-store", storage_type="local"),
                    # Header
                    html.Div(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H1(
                                            "Spotify Listening History",
                                            className="dashboard-title",
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "🌙",
                                                    className="theme-toggle-label",
                                                    id="theme-icon",
                                                ),
                                                dbc.Switch(
                                                    id="theme-toggle",
                                                    value=False,
                                                    className="theme-toggle-switch",
                                                ),
                                            ],
                                            className="theme-toggle-container",
                                        ),
                                    ],
                                    className="header-content",
                                ),
                            ],
                            className="container",
                        ),
                        className="dashboard-header",
                    ),
                    # Global settings collapsible panel
                    html.Div(
                        [
                            dbc.Button(
                                "Global Settings",
                                id="collapse-button",
                                className="mb-3 global-settings-button",
                                color="primary",
                                n_clicks=0,
                            ),
                            dbc.Collapse(
                                dbc.Card(dbc.CardBody(create_global_settings(df))),
                                id="collapse",
                                is_open=False,
                            ),
                        ],
                        className="global-settings-container",
                    ),
                    # Main content tabs
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="Wrapped",
                                children=[create_tab_one_layout(df)],
                            ),
                            dcc.Tab(
                                label="Trends",
                                children=[create_tab_two_layout(df, spotify_data)],
                            ),
                        ]
                    ),
                ],
                id="app-container",
                className="",
            )
        ]
    )


def create_tab_one_layout(df: pd.DataFrame) -> Component:
    """Create the layout for the 'Wrapped' tab.

    This includes filters, graphs, detailed stats, and a daily heatmap.

    Args:
        df (pd.DataFrame): User listening history DataFrame.

    Returns:
        Component: Dash HTML component for the Wrapped tab.
    """
    return html.Div(
        [
            # Filters section
            html.Div(
                create_wrapped_filters_section(df),
                className="card",
            ),
            # Graphs section with loading indicator
            dcc.Loading(
                children=create_graphs_section_tab_one(),
                overlay_style={
                    "visibility": "visible",
                    "filter": "blur(2px)",
                },
                delay_show=2000,
            ),
            # Detailed statistics
            html.Div(
                [
                    html.H3(
                        "Detailed Statistics",
                        className="card-title",
                    ),
                    html.Div(id="detailed-stats"),
                ],
                className="card",
            ),
            # Daily song heatmap
            dcc.Loading(
                children=html.Div(
                    [
                        html.H3(
                            "Daily Song Heatmap",
                            className="card-title",
                        ),
                        dcc.Graph(
                            id="daily-song-heatmap",
                            figure={},
                            config={"displayModeBar": False},
                        ),
                    ],
                    className="card",
                ),
                overlay_style={
                    "visibility": "visible",
                    "filter": "blur(2px)",
                },
                delay_show=2000,
            ),
        ],
        className="container",
    )


def create_tab_two_layout(
    df: pd.DataFrame,
    spotify_data: pd.DataFrame,
) -> Component:
    """Create the layout for the 'Trends' tab.

    This includes filters, several trend graphs (listening, genres,
    artists, tracks), and a hidden store for intermediate data.

    Args:
        df (pd.DataFrame): User listening history DataFrame.
        spotify_data (pd.DataFrame): Additional Spotify data for trends.

    Returns:
        Component: Dash HTML component for the Trends tab.
    """
    return html.Div(
        [
            # Trend filters
            html.Div(
                create_trend_filters_section(df),
                className="card",
            ),
            dcc.Loading(
                overlay_style={
                    "visibility": "visible",
                    "filter": "blur(2px)",
                },
                delay_show=2000,
                children=[
                    # Listening over time section
                    html.Div(
                        [
                            html.H3(
                                "Listening Over Time",
                                className="card-title",
                            ),
                            create_monthly_trend_filter(),
                            html.Div(
                                [
                                    html.H3(
                                        "Listening Trends",
                                        className="card-title",
                                    ),
                                    dcc.Loading(
                                        children=dcc.Graph(
                                            id="trends-graph",
                                            figure={},
                                            config={"displayModeBar": False},
                                        )
                                    ),
                                ]
                            ),
                        ],
                        className="card",
                    ),
                    # Genres over time section
                    html.Div(
                        [
                            html.H3(
                                "Genres Over Time",
                                className="card-title",
                            ),
                            create_genre_trends_layout(df, spotify_data),
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
                                ]
                            ),
                        ],
                        className="card",
                    ),
                    # Artists over time section
                    html.Div(
                        [
                            html.H3(
                                "Artists Over Time",
                                className="card-title",
                            ),
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
                                ]
                            ),
                        ],
                        className="card",
                    ),
                    # Tracks over time section
                    html.Div(
                        [
                            html.H3(
                                "Tracks Over Time",
                                className="card-title",
                            ),
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
                                ]
                            ),
                        ],
                        className="card",
                    ),
                    # Store intermediate tab-2 data
                    dcc.Store(id="tab-2-data"),
                ],
            ),
        ],
        className="container",
    )
