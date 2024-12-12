from datetime import datetime

import pandas as pd
from dash import dcc, html


def create_year_dropdown(df: pd.DataFrame):
    available_years = sorted(df["ts"].dt.year.unique())
    return html.Div(
        [
            html.Label("Select Year", className="filter-label"),
            dcc.Dropdown(
                id="year-dropdown",
                options=[
                    {"label": str(year), "value": year} for year in available_years
                ],
                value=available_years[-1],
                className="dropdown",
            ),
        ],
        className="filter-item",
    )


def create_december_toggle(tab_id: str):
    return html.Div(
        [
            html.Label("Exclude December", className="filter-label"),
            dcc.RadioItems(
                id=f"exclude-december-{tab_id}",
                options=[
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ],
                value=True,
                className="radio-group",
            ),
        ],
        className="filter-item",
    )


def create_incognito_toggle(tab_id: str):
    return html.Div(
        [
            html.Label("Remove Incognito", className="filter-label"),
            dcc.RadioItems(
                id=f"remove-incognito-{tab_id}",
                options=[
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ],
                value=True,
                className="radio-group",
            ),
        ],
        className="filter-item",
    )


def create_wrapped_filters_section(df: pd.DataFrame):
    return html.Div(
        [
            html.H3("Filters", className="card-title"),
            html.Div(
                [
                    create_year_dropdown(df),
                    create_december_toggle("tab-one"),
                    create_incognito_toggle("tab-one"),
                ],
                className="filters-section",
            ),
        ]
    )


def create_year_range_filter(df: pd.DataFrame) -> html.Div:
    min_date = datetime(df["ts"].min().year, df["ts"].min().month, df["ts"].min().day)
    max_date = datetime(df["ts"].max().year, df["ts"].max().month, df["ts"].max().day)

    return html.Div(
        [
            html.Label("Select Date Range", className="filter-label"),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
                className="datepicker",
            ),
        ],
        className="filter-item",
    )


def create_genre_trends_layout():
    """Create the layout for the genre trends analysis section"""
    return html.Div(
        [
            # Genre Filter
            html.Div(
                [
                    html.Label("Select Genres", className="filter-label"),
                    dcc.Dropdown(
                        id="genre-filter-dropdown", multi=True, className="dropdown"
                    ),
                ],
                className="filter-item",
            ),
            # Top N Genres Slider
            html.Div(
                [
                    html.Label("Number of Top Genres", className="filter-label"),
                    dcc.Slider(
                        id="top-genres-slider",
                        min=3,
                        max=15,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(3, 16, 3)},
                        className="slider",
                    ),
                ],
                className="filter-item",
            ),
            # Display Type Selector
            html.Div(
                [
                    html.Label("Display Type", className="filter-label"),
                    dcc.RadioItems(
                        id="display-type-radio",
                        options=[
                            {"label": "Percentage", "value": "percentage"},
                            {"label": "Play Count", "value": "play_count"},
                        ],
                        value="percentage",
                        className="radio-group",
                    ),
                ],
                className="filter-item",
            ),
        ],
        className="filters-section",
    )


def create_monthly_trend_filter() -> html.Div:
    return html.Div(
        [
            html.Label("Select Metric", className="filter-label"),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[
                    {"label": "Total Hours", "value": "total_hours"},
                    {"label": "Unique Tracks", "value": "unique_tracks"},
                    {"label": "Unique Artists", "value": "unique_artists"},
                    {"label": "Average Hours per Day", "value": "avg_hours_per_day"},
                ],
                value="total_hours",
                className="dropdown",
            ),
        ],
        className="filter-item",
    )


def create_trend_filters_section(df: pd.DataFrame):
    return html.Div(
        [
            html.H3("Filters", className="card-title"),
            html.Div(
                [
                    create_year_range_filter(df),
                    create_december_toggle("tab-two"),
                    create_incognito_toggle("tab-two"),
                ],
                className="filters-section",
            ),
        ]
    )
