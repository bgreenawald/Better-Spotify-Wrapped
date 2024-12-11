import pandas as pd
from dash import dcc, html

from dashboard.components.filters import create_filters_section
from dashboard.components.graphs import create_graphs_section


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
                        label="Testing",
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
            html.Div([create_filters_section(df)], className="card"),
            # Graphs Section
            create_graphs_section(),
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
    return html.Div(["Testing"], className="container")
