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
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "Spotify Listening History", className="dashboard-title"
                            )
                        ],
                        className="container",
                    )
                ],
                className="dashboard-header",
            ),
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
                        dbc.Card(dbc.CardBody([create_global_settings(df)])),
                        id="collapse",
                        is_open=False,
                    ),
                ],
                className="global-settings-container",
            ),
            # Main content
            dcc.Tabs(
                children=[
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
        ]
    )


def create_tab_one_layout(df: pd.DataFrame) -> Component:
    return html.Div(
        [
            # Filters Card
            html.Div([create_wrapped_filters_section(df)], className="card"),
            # Graphs Section
            dcc.Loading(
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                delay_show=2000,
                children=create_graphs_section_tab_one(),
            ),
            # Stats Card
            html.Div(
                [
                    html.H3("Detailed Statistics", className="card-title"),
                    html.Div(id="detailed-stats"),
                ],
                className="card",
            ),
            # Daily song heat-map
            dcc.Loading(
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                delay_show=2000,
                children=[
                    html.Div(
                        [
                            html.H3("Daily Song Heatmap", className="card-title"),
                            dcc.Graph(
                                id="daily-song-heatmap",
                                figure={},
                                config={"displayModeBar": False},
                            ),
                        ],
                        className="card",
                    )
                ],
            ),
        ],
        className="container",
    )


def create_tab_two_layout(df: pd.DataFrame, spotify_data: pd.DataFrame) -> Component:
    return html.Div(
        [
            # Main filters cards
            html.Div([create_trend_filters_section(df)], className="card"),
            dcc.Loading(
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                delay_show=2000,
                children=[
                    # Graphs Section
                    html.Div(
                        [
                            html.H3("Listening Over Time", className="card-title"),
                            create_monthly_trend_filter(),
                            html.Div(
                                [
                                    html.H3("Listening Trends", className="card-title"),
                                    dcc.Loading(
                                        [
                                            dcc.Graph(
                                                id="trends-graph",
                                                figure={},
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.H3("Genres Over Time", className="card-title"),
                            create_genre_trends_layout(df, spotify_data),
                            dcc.Loading(
                                [
                                    html.Div(
                                        [
                                            # Main Trend Graph
                                            dcc.Graph(
                                                id="genre-trends-graph",
                                                config={"displayModeBar": False},
                                            )
                                        ],
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                id="genre-trends-table",
                                                className="table-container",
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.H3("Artists Over Time", className="card-title"),
                            create_artist_trends_layout(df),
                            dcc.Loading(
                                [
                                    html.Div(
                                        [
                                            # Main Trend Graph
                                            dcc.Graph(
                                                id="artist-trends-graph",
                                                config={"displayModeBar": False},
                                            )
                                        ],
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                id="artist-trends-table",
                                                className="table-container",
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Tracks Over Time",
                                        className="card-title",
                                    ),
                                    create_track_trends_layout(df),
                                    dcc.Loading(
                                        [
                                            # Main Trend Graph
                                            dcc.Graph(
                                                id="track-trends-graph",
                                                config={"displayModeBar": False},
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        id="track-trends-table",
                                                        className="table-container",
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                        ],
                        className="card",
                    ),
                    # dcc.Store stores the intermediate value
                    dcc.Store(id="tab-2-data"),
                ],
            ),
        ],
        className="container",
    )
