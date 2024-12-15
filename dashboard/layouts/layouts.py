import pandas as pd
from dash import dcc, html

from dashboard.components.filters import (
    create_artist_trends_layout,
    create_genre_trends_layout,
    create_monthly_trend_filter,
    create_track_trends_layout,
    create_trend_filters_section,
    create_wrapped_filters_section,
)
from dashboard.components.graphs import create_graphs_section_tab_one


def create_layout(df: pd.DataFrame):
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
            # Main content
            dcc.Tabs(
                children=[
                    dcc.Tab(
                        label="Wrapped",
                        children=[create_tab_one_layout(df)],
                    ),
                    dcc.Tab(
                        label="Trends",
                        children=[create_tab_two_layout(df)],
                    ),
                ]
            ),
        ]
    )


def create_tab_one_layout(df: pd.DataFrame):
    return html.Div(
        [
            # Filters Card
            html.Div([create_wrapped_filters_section(df)], className="card"),
            # Graphs Section
            create_graphs_section_tab_one(),
            # Stats Card
            html.Div(
                [
                    html.H3("Detailed Statistics", className="card-title"),
                    html.Div(id="detailed-stats"),
                ],
                className="card",
            ),
        ],
        className="container",
    )


def create_tab_two_layout(df: pd.DataFrame):
    return html.Div(
        [
            # Main filters cards
            html.Div([create_trend_filters_section(df)], className="card"),
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
                    create_genre_trends_layout(),
                    html.Div(
                        [
                            # Main Trend Graph
                            dcc.Loading(
                                [
                                    dcc.Graph(
                                        id="genre-trends-graph",
                                        config={"displayModeBar": False},
                                    )
                                ],
                            ),
                        ],
                    ),
                    html.Div([dcc.Loading([html.Div(id="genre-trends-table")])]),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H3("Artists Over Time", className="card-title"),
                    create_artist_trends_layout(),
                    html.Div(
                        [
                            # Main Trend Graph
                            dcc.Loading(
                                [
                                    dcc.Graph(
                                        id="artist-trends-graph",
                                        config={"displayModeBar": False},
                                    )
                                ],
                            ),
                        ],
                    ),
                    html.Div([dcc.Loading([html.Div(id="artist-trends-table")])]),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H3("Tracks Over Time", className="card-title"),
                    create_track_trends_layout(),
                    html.Div(
                        [
                            # Main Trend Graph
                            dcc.Loading(
                                [
                                    dcc.Graph(
                                        id="track-trends-graph",
                                        config={"displayModeBar": False},
                                    )
                                ],
                            ),
                        ],
                    ),
                    html.Div([dcc.Loading([html.Div(id="track-trends-table")])]),
                ],
                className="card",
            ),
        ],
        className="container",
    )
