import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.development.base_component import Component

from dashboard.components.filters import (
    create_global_settings,
    create_trend_filters_section,
    create_wrapped_filters_section,
)
from dashboard.components.graphs import create_graphs_section_tab_one


def _create_user_selector(df: pd.DataFrame) -> Component:
    users = sorted(df.get("user_id", pd.Series(dtype=str)).dropna().unique())
    default_user = users[0] if users else None
    return html.Div(
        [
            html.Label("User", className="filter-label"),
            dcc.Dropdown(
                id="user-id-dropdown",
                options=[{"label": u, "value": u} for u in users],
                value=default_user,
                clearable=False,
                className="dropdown",
            ),
        ],
        className="user-selector",
        style={"minWidth": "240px"},
    )


def create_layout(df: pd.DataFrame) -> Component:
    """Generate the main dashboard layout.

    The layout includes a header, a global settings panel, and two tabs:
    Wrapped and Trends.

    Args:
        df (pd.DataFrame): User listening history DataFrame.

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
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            _create_user_selector(df),
                                            create_global_settings(df),
                                        ]
                                    )
                                ),
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
                                children=[create_tab_two_layout(df)],
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
            # Graphs section (each graph handles its own loading state)
            create_graphs_section_tab_one(),
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
            # Daily song heatmap (local loading)
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
                    "backgroundColor": "rgba(0,0,0,0.15)",
                },
                delay_show=0,
                type="default",
            ),
            # Store precomputed Wrapped tab data
            dcc.Store(id="tab-1-data"),
        ],
        className="container",
    )


def create_tab_two_layout(
    df: pd.DataFrame,
) -> Component:
    """Create the layout for the 'Trends' tab.

    This includes filters, several trend graphs (listening, genres,
    artists, tracks), and a hidden store for intermediate data.

    Args:
        df (pd.DataFrame): User listening history DataFrame.

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
            # Chart selector for Trends tab
            html.Div(
                [
                    html.H3("Select Chart", className="card-title"),
                    dcc.RadioItems(
                        id="tab-2-chart-selector",
                        options=[
                            {"label": "Listening Over Time", "value": "listening"},
                            {"label": "Genres Over Time", "value": "genres"},
                            {"label": "Artists Over Time", "value": "artists"},
                            {"label": "Tracks Over Time", "value": "tracks"},
                        ],
                        value="listening",
                        className="radio-group",
                    ),
                ],
                className="card",
            ),
            # Tab-2 content with outer loader to avoid blank flash on mount
            dcc.Loading(
                children=html.Div(id="tab-2-content"),
                delay_show=0,
                overlay_style={
                    "visibility": "visible",
                    "backgroundColor": "rgba(0,0,0,0.15)",
                },
                type="default",
            ),
            # Store intermediate tab-2 data
            dcc.Store(id="tab-2-data"),
            # Store cached genre options to avoid recompute
            dcc.Store(id="genre-options-store"),
        ],
        className="container",
    )
