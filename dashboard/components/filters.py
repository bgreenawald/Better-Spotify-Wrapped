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


def create_december_toggle():
    return html.Div(
        [
            html.Label("Exclude December", className="filter-label"),
            dcc.RadioItems(
                id="exclude-december",
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


def create_incognito_toggle():
    return html.Div(
        [
            html.Label("Remove Incognito", className="filter-label"),
            dcc.RadioItems(
                id="remove-incognito",
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


def create_filters_section(df: pd.DataFrame):
    return html.Div(
        [
            html.H3("Filters", className="card-title"),
            html.Div(
                [
                    create_year_dropdown(df),
                    create_december_toggle(),
                    create_incognito_toggle(),
                ],
                className="filters-section",
            ),
        ]
    )
