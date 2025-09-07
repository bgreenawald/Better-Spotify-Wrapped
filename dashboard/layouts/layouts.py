import dash_bootstrap_components as dbc
import dash_mantine_components as dmc  # type: ignore
import pandas as pd
from dash import dcc, html
from dash.development.base_component import Component

from dashboard.components.filters import (
    create_global_settings,
    create_social_date_range_filter,
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
                placeholder="Select user…",
                persistence=True,
                persistence_type="local",
                className="dropdown",
            ),
        ],
        className="user-selector",
        style={"minWidth": "240px", "marginBottom": "30px"},
    )


def _create_social_user_selector(df: pd.DataFrame) -> Component:
    users = sorted(df.get("user_id", pd.Series(dtype=str)).dropna().unique())
    return html.Div(
        [
            html.Label("Select Users (2–3)", className="filter-label"),
            dcc.Dropdown(
                id="social-users-dropdown",
                options=[{"label": u, "value": u} for u in users],
                value=[],
                multi=True,
                placeholder="Choose users to compare…",
                persistence=True,
                persistence_type="local",
                className="dropdown",
            ),
        ],
        className="filter-item",
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
                                                # Theme toggle
                                                dmc.Switch(
                                                    id="theme-toggle",
                                                    checked=False,
                                                    size="md",
                                                    color="green",
                                                    onLabel="☀️",
                                                    offLabel="🌙",
                                                    className="theme-toggle-switch",
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                # Global settings hamburger toggle (moved into header)
                                                dbc.Button(
                                                    html.Span("☰"),
                                                    id="collapse-button",
                                                    title="Global settings",
                                                    className="header-settings-button",
                                                    color="light",
                                                    size="sm",
                                                    n_clicks=0,
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
                            dbc.Collapse(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            _create_user_selector(df),
                                            create_global_settings(),
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
                        id="main-tabs",
                        value="wrapped",
                        children=[
                            dcc.Tab(
                                label="🎁 Wrapped",
                                value="wrapped",
                                children=[create_tab_one_layout(df)],
                            ),
                            dcc.Tab(
                                label="📈 Trends",
                                value="trends",
                                children=[create_tab_two_layout(df)],
                            ),
                            dcc.Tab(
                                label="👥 Social",
                                value="social",
                                children=[create_tab_social_layout(df)],
                            ),
                        ],
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
            # Daily song heatmap (Highcharts)
            dcc.Loading(
                children=html.Div(
                    [
                        html.H3(
                            "Daily Song Heatmap",
                            className="card-title",
                        ),
                        dcc.Store(id="daily-song-heatmap-options"),
                        html.Div(
                            id="daily-song-heatmap-container",
                            className="card",
                            children=html.Div(id="daily-song-heatmap-container-root"),
                            style={"minHeight": "420px"},
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
                        inputClassName="radio-pill",
                        labelClassName="radio-pill-label",
                        persistence=True,
                        persistence_type="local",
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
            # Store for genre hide-level0 toggle (to avoid referencing
            # dynamic controls as Inputs/State in other callbacks)
            dcc.Store(id="genre-hide-level0-store"),
        ],
        className="container",
    )


def create_tab_social_layout(df: pd.DataFrame) -> Component:
    """Create layout for the Social tab (multi-user comparison via Venn).

    Contains: local date range, user multiselect, mode radio, venn display, and details.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Filters", className="card-title"),
                    html.Div(
                        [
                            _create_social_user_selector(df),
                            # Local date range (independent of Trends tab)
                            create_social_date_range_filter(df),
                            html.Div(
                                [
                                    html.Label("Mode", className="filter-label"),
                                    dcc.RadioItems(
                                        id="social-mode",
                                        options=[
                                            {"label": "Tracks", "value": "tracks"},
                                            {"label": "Artists", "value": "artists"},
                                            {"label": "Genres", "value": "genres"},
                                        ],
                                        value="tracks",
                                        className="radio-group",
                                        inputClassName="radio-pill",
                                        labelClassName="radio-pill-label",
                                        persistence=True,
                                        persistence_type="local",
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Exclude Parent Genres (Level 0)",
                                        className="filter-label",
                                    ),
                                    dcc.RadioItems(
                                        id="social-genre-hide-level0-radio",
                                        options=[
                                            {"label": "Yes", "value": True},
                                            {"label": "No", "value": False},
                                        ],
                                        value=False,
                                        className="radio-group",
                                        inputClassName="radio-pill",
                                        labelClassName="radio-pill-label",
                                        persistence=True,
                                        persistence_type="local",
                                    ),
                                ],
                                className="filter-item",
                                id="social-genre-hide-level0-container",
                            ),
                        ],
                        className="filters-section",
                    ),
                ],
                className="card",
            ),
            # Content
            html.Div(
                [
                    html.H3("Comparison", className="card-title"),
                    dcc.Loading(
                        children=dcc.Graph(id="social-venn-graph", figure={}),
                        delay_show=0,
                        overlay_style={
                            "visibility": "visible",
                            "backgroundColor": "rgba(0,0,0,0.15)",
                        },
                        type="default",
                    ),
                    html.Div(id="social-region-lists"),
                ],
                className="card",
            ),
            # Store for computed social data
            dcc.Store(id="social-data"),
            # Store for selected region (to filter lists)
            dcc.Store(id="social-selected-region"),
        ],
        className="container",
    )
