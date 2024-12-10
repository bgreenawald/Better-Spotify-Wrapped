import pandas as pd
from dash import dcc, html


def create_year_dropdown(df: pd.DataFrame):
    available_years = sorted(df["ts"].dt.year.unique())
    return html.Div(
        [
            html.Label("Select Year"),
            dcc.Dropdown(
                id="year-dropdown",
                options=[
                    {"label": str(year), "value": year} for year in available_years
                ],
                value=available_years[-1],
            ),
        ],
        style={"width": "30%", "display": "inline-block", "marginRight": "20px"},
    )


def create_december_toggle():
    return html.Div(
        [
            html.Label("Exclude December"),
            dcc.RadioItems(
                id="exclude-december",
                options=[
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ],
                value=True,
            ),
        ],
        style={"width": "30%", "display": "inline-block", "marginRight": "20px"},
    )


def create_incognito_toggle():
    return html.Div(
        [
            html.Label("Remove Incognito"),
            dcc.RadioItems(
                id="remove-incognito",
                options=[
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ],
                value=True,
            ),
        ],
        style={"width": "30%", "display": "inline-block"},
    )


def create_filters_section(df: pd.DataFrame):
    return html.Div(
        [create_year_dropdown(df), create_december_toggle(), create_incognito_toggle()],
        style={"marginBottom": "30px"},
    )
